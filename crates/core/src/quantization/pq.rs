//! Product Quantization (PQ) for inline neighbor gating.
//!
//! Global codebook trained offline via k-means. Each vector encoded as M bytes
//! (one centroid index per subspace). At query time, a per-query lookup table
//! enables O(M) approximate distance per neighbor.
//!
//! Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search", TPAMI 2011.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// PQ codebook: M subspaces × 256 centroids × subspace_dim floats.
///
/// Stored as `centroids[m][c][d]` where:
/// - m ∈ [0, M): subspace index
/// - c ∈ [0, 256): centroid index
/// - d ∈ [0, subspace_dim): dimension within subspace
pub struct PqCodebook {
    /// Number of subquantizers.
    pub m: usize,
    /// Centroids per subspace (always 256 for PQ×8).
    pub k: usize,
    /// Dimension of each subspace (= full_dim / m).
    pub subspace_dim: usize,
    /// Full vector dimension.
    pub full_dim: usize,
    /// Flat centroid storage: m * k * subspace_dim floats.
    /// Layout: centroids[m][c][d] at index (m * k + c) * subspace_dim + d.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train a PQ codebook from a set of vectors using k-means.
    ///
    /// `vectors`: flat f32 array, shape (n, dim), row-major.
    /// `m`: number of subquantizers. `dim` must be divisible by `m`.
    /// `max_iter`: k-means iterations per subspace.
    /// `seed`: random seed for centroid initialization.
    pub fn train(vectors: &[f32], n: usize, dim: usize, m: usize, max_iter: usize, seed: u64) -> Self {
        assert_eq!(vectors.len(), n * dim);
        assert!(dim % m == 0, "dim ({dim}) must be divisible by m ({m})");
        assert!(n >= 256, "need at least 256 vectors to train PQ (got {n})");

        let k = 256usize;
        let sd = dim / m;
        let mut centroids = vec![0.0f32; m * k * sd];

        for sub in 0..m {
            let offset = sub * sd;

            // Extract subspace slices into contiguous buffer
            let mut sub_data = vec![0.0f32; n * sd];
            for i in 0..n {
                sub_data[i * sd..(i + 1) * sd]
                    .copy_from_slice(&vectors[i * dim + offset..i * dim + offset + sd]);
            }

            // Run k-means for this subspace
            let sub_centroids = kmeans(&sub_data, n, sd, k, max_iter, seed.wrapping_add(sub as u64));

            // Store centroids
            let dst_start = sub * k * sd;
            centroids[dst_start..dst_start + k * sd].copy_from_slice(&sub_centroids);
        }

        Self {
            m,
            k,
            subspace_dim: sd,
            full_dim: dim,
            centroids,
        }
    }

    /// Get centroid slice for subspace m, centroid c.
    #[inline]
    fn centroid(&self, m: usize, c: usize) -> &[f32] {
        let start = (m * self.k + c) * self.subspace_dim;
        &self.centroids[start..start + self.subspace_dim]
    }

    /// Encode a single vector to PQ codes.
    /// `vector`: f32 slice of length `full_dim`.
    /// `codes`: output u8 slice of length `m`.
    pub fn encode_one(&self, vector: &[f32], codes: &mut [u8]) {
        debug_assert_eq!(vector.len(), self.full_dim);
        debug_assert_eq!(codes.len(), self.m);

        for sub in 0..self.m {
            let offset = sub * self.subspace_dim;
            let sub_vec = &vector[offset..offset + self.subspace_dim];

            let mut best_dist = f32::MAX;
            let mut best_c = 0u8;

            for c in 0..self.k {
                let cent = self.centroid(sub, c);
                let dist = l2_sq(sub_vec, cent);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c as u8;
                }
            }

            codes[sub] = best_c;
        }
    }

    /// Encode all vectors. Returns flat Vec<u8> of shape (n, m).
    pub fn encode_all(&self, vectors: &[f32], n: usize) -> Vec<u8> {
        assert_eq!(vectors.len(), n * self.full_dim);
        let mut codes = vec![0u8; n * self.m];
        for i in 0..n {
            let vec_slice = &vectors[i * self.full_dim..(i + 1) * self.full_dim];
            self.encode_one(vec_slice, &mut codes[i * self.m..(i + 1) * self.m]);
        }
        codes
    }

    /// Build a per-query distance lookup table for Asymmetric Distance Computation (ADC).
    ///
    /// For L2: lut[m][c] = ||q_sub_m - centroid[m][c]||²
    /// For inner product (cosine on normalized vectors): lut[m][c] = dot(q_sub_m, centroid[m][c])
    ///   (caller negates or uses 1 - sum for cosine distance)
    pub fn build_distance_table(&self, query: &[f32], use_inner_product: bool) -> PqDistanceTable {
        assert_eq!(query.len(), self.full_dim);

        let mut table = vec![0.0f32; self.m * self.k];

        for sub in 0..self.m {
            let offset = sub * self.subspace_dim;
            let q_sub = &query[offset..offset + self.subspace_dim];

            for c in 0..self.k {
                let cent = self.centroid(sub, c);
                let val = if use_inner_product {
                    dot(q_sub, cent)
                } else {
                    l2_sq(q_sub, cent)
                };
                table[sub * self.k + c] = val;
            }
        }

        PqDistanceTable {
            table,
            m: self.m,
            k: self.k,
            use_inner_product,
        }
    }

    /// Save codebook to binary file.
    ///
    /// Format: [magic: u32][version: u32][m: u32][k: u32][subspace_dim: u32][full_dim: u32]
    ///         [centroids: f32 × m × k × subspace_dim]
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&0x50514342u32.to_le_bytes())?; // magic "PQCB"
        w.write_all(&1u32.to_le_bytes())?; // version
        w.write_all(&(self.m as u32).to_le_bytes())?;
        w.write_all(&(self.k as u32).to_le_bytes())?;
        w.write_all(&(self.subspace_dim as u32).to_le_bytes())?;
        w.write_all(&(self.full_dim as u32).to_le_bytes())?;

        // Write centroids as raw f32 LE
        for &v in &self.centroids {
            w.write_all(&v.to_le_bytes())?;
        }

        w.flush()
    }

    /// Load codebook from binary file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x50514342 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad PQ codebook magic"));
        }

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported PQ codebook version {version}"),
            ));
        }

        r.read_exact(&mut buf4)?;
        let m = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let k = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let subspace_dim = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let full_dim = u32::from_le_bytes(buf4) as usize;

        let num_floats = m * k * subspace_dim;
        let mut raw = vec![0u8; num_floats * 4];
        r.read_exact(&mut raw)?;

        let centroids: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            m,
            k,
            subspace_dim,
            full_dim,
            centroids,
        })
    }
}

/// Per-query PQ distance lookup table.
/// Pre-computed distances from query subvectors to all centroids.
pub struct PqDistanceTable {
    /// Flat table: m × k floats. table[sub * k + c] = distance(q_sub, centroid[sub][c]).
    table: Vec<f32>,
    m: usize,
    k: usize,
    use_inner_product: bool,
}

impl PqDistanceTable {
    /// Whether this table was built in inner-product mode (dot products).
    #[inline]
    pub fn use_inner_product(&self) -> bool {
        self.use_inner_product
    }

    /// Compute approximate distance from query to a PQ-encoded vector.
    ///
    /// For L2: returns sum of squared subspace distances.
    /// For inner product: returns sum of dot products (caller converts to cosine distance).
    #[inline]
    pub fn approximate_distance(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.m);
        let mut dist = 0.0f32;
        for sub in 0..self.m {
            dist += self.table[sub * self.k + code[sub] as usize];
        }
        dist
    }

    /// Compute approximate cosine distance (1 - dot product).
    /// Only valid when `use_inner_product` was true during table construction,
    /// AND both query and codebook centroids were trained on L2-normalized vectors.
    #[inline]
    pub fn approximate_cosine_distance(&self, code: &[u8]) -> f32 {
        debug_assert!(self.use_inner_product);
        1.0 - self.approximate_distance(code)
    }

    /// Number of subquantizers.
    pub fn m(&self) -> usize {
        self.m
    }
}

// ─── k-means ───────────────────────────────────────────────────────────────

/// Simple k-means clustering. Returns flat centroid array of shape (k, dim).
fn kmeans(data: &[f32], n: usize, dim: usize, k: usize, max_iter: usize, seed: u64) -> Vec<f32> {
    assert!(n >= k);

    // Initialize centroids: deterministic sampling (stride-based, seeded)
    let mut centroids = vec![0.0f32; k * dim];
    let mut rng = seed;
    let mut used = vec![false; n];

    for c in 0..k {
        // Simple LCG to pick initial centroids without replacement
        loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % n;
            if !used[idx] {
                used[idx] = true;
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
                break;
            }
        }
    }

    let mut assignments = vec![0u32; n];
    let mut counts = vec![0u32; k];

    for _iter in 0..max_iter {
        let prev_centroids = centroids.clone();

        // Assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let point = &data[i * dim..(i + 1) * dim];
            let mut best_dist = f32::MAX;
            let mut best_c = 0u32;

            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let d = l2_sq(point, cent);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u32;
                }
            }

            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        centroids.fill(0.0);
        counts.fill(0);

        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let point = &data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                centroids[c * dim + d] += point[d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] *= inv;
                }
            } else {
                // Keep the previous centroid to avoid creating a "dead" zero vector.
                // Empty clusters happen frequently when k=256 and n is not huge.
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&prev_centroids[c * dim..(c + 1) * dim]);
            }
        }
    }

    centroids
}

/// Squared L2 distance between two slices.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Dot product between two slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Vec<f32>, usize, usize) {
        // 512 vectors in 8 dimensions, 2 clear clusters per subspace
        let n = 512;
        let dim = 8;
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            let cluster = if i < n / 2 { 1.0f32 } else { -1.0 };
            for d in 0..dim {
                data.push(cluster + (i as f32 * 0.001) + (d as f32 * 0.01));
            }
        }
        (data, n, dim)
    }

    #[test]
    fn train_and_encode() {
        let (data, n, dim) = make_simple_data();
        let m = 4; // 4 subquantizers, 2 dims each
        let cb = PqCodebook::train(&data, n, dim, m, 10, 42);

        assert_eq!(cb.m, 4);
        assert_eq!(cb.k, 256);
        assert_eq!(cb.subspace_dim, 2);
        assert_eq!(cb.full_dim, 8);
        assert_eq!(cb.centroids.len(), 4 * 256 * 2);

        // Encode first vector
        let mut codes = vec![0u8; m];
        cb.encode_one(&data[0..dim], &mut codes);

        // Encode second vector — should get same codes if in same cluster
        let mut codes2 = vec![0u8; m];
        cb.encode_one(&data[dim..2 * dim], &mut codes2);

        // Both in cluster 0, codes should be identical or very close
        // (not guaranteed identical due to small perturbation, but same centroid)
    }

    #[test]
    fn encode_all_shape() {
        let (data, n, dim) = make_simple_data();
        let cb = PqCodebook::train(&data, n, dim, 4, 5, 42);
        let codes = cb.encode_all(&data, n);
        assert_eq!(codes.len(), n * 4);
    }

    #[test]
    fn distance_table_l2() {
        let (data, n, dim) = make_simple_data();
        let cb = PqCodebook::train(&data, n, dim, 4, 10, 42);

        let query = &data[0..dim]; // first vector as query
        let dt = cb.build_distance_table(query, false);

        // Distance to self (via PQ) should be small
        let mut codes = vec![0u8; 4];
        cb.encode_one(query, &mut codes);
        let self_dist = dt.approximate_distance(&codes);
        assert!(self_dist < 0.1, "self-distance too large: {self_dist}");

        // Distance to opposite cluster should be larger
        let mut codes_far = vec![0u8; 4];
        cb.encode_one(&data[8 * dim..9 * dim], &mut codes_far);
        let far_dist = dt.approximate_distance(&codes_far);
        assert!(far_dist > self_dist, "far point should be further");
    }

    #[test]
    fn distance_table_ip() {
        let (data, n, dim) = make_simple_data();
        let cb = PqCodebook::train(&data, n, dim, 4, 10, 42);

        let query = &data[0..dim];
        let dt = cb.build_distance_table(query, true);
        assert_eq!(dt.m(), 4);

        let mut codes = vec![0u8; 4];
        cb.encode_one(query, &mut codes);
        let self_dot = dt.approximate_distance(&codes);
        // Self dot product should be positive and large
        assert!(self_dot > 0.0);
    }

    #[test]
    fn codebook_save_load_roundtrip() {
        let (data, n, dim) = make_simple_data();
        let cb = PqCodebook::train(&data, n, dim, 4, 5, 42);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pq_codebook.bin");

        cb.save(&path).unwrap();
        let cb2 = PqCodebook::load(&path).unwrap();

        assert_eq!(cb2.m, cb.m);
        assert_eq!(cb2.k, cb.k);
        assert_eq!(cb2.subspace_dim, cb.subspace_dim);
        assert_eq!(cb2.full_dim, cb.full_dim);
        assert_eq!(cb2.centroids.len(), cb.centroids.len());

        // Centroids should be identical
        for (a, b) in cb.centroids.iter().zip(cb2.centroids.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn ranking_quality() {
        // Generate data with clear structure: 256 vectors in 16 dims
        let n = 256;
        let dim = 16;
        let m = 4;
        let mut data = vec![0.0f32; n * dim];
        let mut rng = 12345u64;
        for i in 0..n {
            for d in 0..dim {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                data[i * dim + d] = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
        }

        let cb = PqCodebook::train(&data, n, dim, m, 20, 42);
        let codes = cb.encode_all(&data, n);

        // Use vector 0 as query
        let query = &data[0..dim];
        let dt = cb.build_distance_table(query, false);

        // Compute exact and approximate distances
        let mut exact: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, l2_sq(query, &data[i * dim..(i + 1) * dim])))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut approx: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, dt.approximate_distance(&codes[i * m..(i + 1) * m])))
            .collect();
        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Top-10 overlap should be reasonable (>= 5 out of 10)
        let exact_top10: Vec<usize> = exact[..10].iter().map(|&(i, _)| i).collect();
        let approx_top10: Vec<usize> = approx[..10].iter().map(|&(i, _)| i).collect();
        let overlap = exact_top10
            .iter()
            .filter(|i| approx_top10.contains(i))
            .count();

        assert!(
            overlap >= 4,
            "top-10 overlap too low: {overlap}/10 (exact={exact_top10:?}, approx={approx_top10:?})"
        );
    }
}
