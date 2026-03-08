//! Ada-ef v0: Query-adaptive (ef, stall_limit, drain_budget) estimation.
//!
//! Based on "Distribution-Aware Exploration for Adaptive HNSW Search" (SIGMOD 2026),
//! adapted for disk-based search: uses entry_set seed distances (pure DRAM, zero IO)
//! instead of the paper's 2-hop graph traversal.
//!
//! v0 uses diagonal-only variance (O(d) per query), ignoring cross-dimension correlations.

/// Precomputed per-dimension statistics of the normalized dataset.
/// Computed once offline from all vectors.
pub struct AdaEfStats {
    /// Per-dimension mean of normalized vectors: E[v̂ᵢ] for i in 0..dim
    pub mean: Vec<f64>,
    /// Per-dimension variance of normalized vectors: Var(v̂ᵢ) for i in 0..dim
    pub variance: Vec<f64>,
    pub dim: usize,
}

impl AdaEfStats {
    /// Compute stats from a flat array of pre-normalized vectors.
    /// `vectors`: flat &[f32] of length n * dim (row-major, already L2-normalized for cosine).
    ///
    /// **IMPORTANT**: Vectors MUST be L2-normalized before calling this.
    /// For cosine distance, FDL theory requires normalized vectors.
    /// Use `from_raw_vectors_cosine()` if your vectors are not pre-normalized.
    pub fn from_normalized_vectors(vectors: &[f32], n: usize, dim: usize) -> Self {
        assert_eq!(vectors.len(), n * dim);

        let mut mean = vec![0.0f64; dim];
        let mut m2 = vec![0.0f64; dim]; // running sum of squared deviations

        // Online Welford's algorithm for numerical stability
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let nf = (i + 1) as f64;
            for j in 0..dim {
                let x = row[j] as f64;
                let delta = x - mean[j];
                mean[j] += delta / nf;
                let delta2 = x - mean[j];
                m2[j] += delta * delta2;
            }
        }

        let variance: Vec<f64> = m2.iter().map(|&s| s / (n as f64 - 1.0)).collect();

        Self { mean, variance, dim }
    }

    /// Compute stats from raw (un-normalized) vectors for cosine distance.
    /// Normalizes each vector inline before computing mean/variance.
    pub fn from_raw_vectors_cosine(vectors: &[f32], n: usize, dim: usize) -> Self {
        assert_eq!(vectors.len(), n * dim);

        // Normalize a copy row-by-row
        let mut normalized = vectors.to_vec();
        for i in 0..n {
            let row = &mut normalized[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }

        Self::from_normalized_vectors(&normalized, n, dim)
    }

    /// Estimate FDL parameters (μ_CD, σ_CD) for a cosine-distance query.
    /// `query`: raw query vector (will be L2-normalized internally).
    /// Returns (mu_cd, sigma_cd) for the cosine distance distribution.
    pub fn estimate_fdl_params(&self, query: &[f32]) -> (f64, f64) {
        assert_eq!(query.len(), self.dim);

        // Normalize query
        let norm: f64 = query.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return (1.0, 0.01); // degenerate query
        }

        let mut mu_cs = 0.0f64;
        let mut sigma2_cs = 0.0f64;

        for i in 0..self.dim {
            let q_hat = query[i] as f64 / norm;
            mu_cs += q_hat * self.mean[i];
            sigma2_cs += q_hat * q_hat * self.variance[i];
        }

        // Cosine distance = 1 - cosine similarity
        let mu_cd = 1.0 - mu_cs;
        let sigma_cd = sigma2_cs.max(1e-18).sqrt();

        (mu_cd, sigma_cd)
    }
}

/// Parameters assigned to a query based on its difficulty score.
#[derive(Debug, Clone, Copy)]
pub struct AdaEfParams {
    pub ef: usize,
    pub stall_limit: u32,
    pub drain_budget: u32,
}

/// A row in the ef-estimation table: score → params.
#[derive(Debug, Clone)]
struct TableRow {
    score_threshold: f64, // queries with score >= this get these params
    params: AdaEfParams,
}

/// Lookup table mapping query difficulty score to search parameters.
/// Built offline by sweeping ef on sampled queries grouped by score.
pub struct AdaEfTable {
    /// Rows sorted by descending score_threshold.
    /// First match (score >= threshold) wins.
    rows: Vec<TableRow>,
    /// Floor: minimum params regardless of score.
    floor: AdaEfParams,
}

impl AdaEfTable {
    /// Create a simple table from score thresholds.
    /// `entries`: (score_threshold, ef, stall_limit, drain_budget) sorted by descending score.
    /// `floor`: minimum params (WAE equivalent).
    pub fn new(entries: &[(f64, usize, u32, u32)], floor: AdaEfParams) -> Self {
        let mut rows: Vec<TableRow> = entries
            .iter()
            .map(|&(score, ef, sl, db)| TableRow {
                score_threshold: score,
                params: AdaEfParams {
                    ef,
                    stall_limit: sl,
                    drain_budget: db,
                },
            })
            .collect();
        // Sort descending by score so first match wins
        rows.sort_by(|a, b| b.score_threshold.partial_cmp(&a.score_threshold).unwrap());
        Self { rows, floor }
    }

    /// Look up params for a given score.
    pub fn lookup(&self, score: f64) -> AdaEfParams {
        for row in &self.rows {
            if score >= row.score_threshold {
                return row.params;
            }
        }
        // Score below all thresholds → hardest query → use floor (largest ef)
        self.floor
    }
}

/// Number of quantile bins.
const NUM_BINS: usize = 5;
/// Quantile width per bin (0.001 = 0.1% of distribution per bin).
const DELTA: f64 = 0.001;
/// Exponential decay weights: w_i = 100 * e^{-(i-1)}.
const BIN_WEIGHTS: [f64; NUM_BINS] = [
    100.0,                          // e^0
    100.0 * 0.367879441171442,      // e^-1
    100.0 * 0.135335283236613,      // e^-2
    100.0 * 0.049787068367864,      // e^-3
    100.0 * 0.018315638888734,      // e^-4
];

/// Estimate query difficulty and return adaptive search parameters.
///
/// `seed_distances`: distances from query to entry_set vectors (computed in DRAM).
/// `stats`: precomputed dataset statistics.
/// `query`: the query vector.
/// `table`: score-to-params lookup table.
pub fn estimate_ada_ef(
    seed_distances: &[f32],
    stats: &AdaEfStats,
    query: &[f32],
    table: &AdaEfTable,
) -> AdaEfParams {
    let (mu, sigma) = stats.estimate_fdl_params(query);

    // Compute bin thresholds: θ_i = μ + σ * Φ⁻¹(δ * i)
    let mut thresholds = [0.0f64; NUM_BINS];
    for i in 0..NUM_BINS {
        let quantile = DELTA * (i + 1) as f64;
        thresholds[i] = mu + sigma * inv_normal_cdf(quantile);
    }

    // Count seed distances in each bin
    let mut counts = [0u32; NUM_BINS];
    for &d in seed_distances {
        let d = d as f64;
        for (bin, &thresh) in thresholds.iter().enumerate() {
            if d <= thresh {
                counts[bin] += 1;
                break; // each distance counted in first (lowest) matching bin only
            }
        }
    }

    // Compute weighted score
    let n = seed_distances.len() as f64;
    if n < 1.0 {
        return table.floor;
    }
    let score: f64 = counts
        .iter()
        .zip(BIN_WEIGHTS.iter())
        .map(|(&c, &w)| w * c as f64 / n)
        .sum();

    table.lookup(score)
}

/// Rational approximation of the inverse normal CDF (probit function).
/// Public for diagnostic use in benchmarks.
/// Uses the Abramowitz & Stegun approximation (formula 26.2.23).
/// Accurate to ~4.5e-4 for 0 < p < 1.
pub fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -6.0; // clamp
    }
    if p >= 1.0 {
        return 6.0; // clamp
    }
    if p == 0.5 {
        return 0.0;
    }

    // Use symmetry: for p > 0.5, compute for 1-p and negate
    let (sign, pp) = if p > 0.5 {
        (1.0, 1.0 - p)
    } else {
        (-1.0, p)
    };

    let t = (-2.0 * pp.ln()).sqrt();

    // Coefficients for rational approximation
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let result = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);

    sign * result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_normal_cdf_basic() {
        // Φ⁻¹(0.5) = 0
        assert!((inv_normal_cdf(0.5)).abs() < 1e-6);
        // Φ⁻¹(0.001) ≈ -3.09
        assert!((inv_normal_cdf(0.001) - (-3.09)).abs() < 0.02);
        // Φ⁻¹(0.999) ≈ 3.09
        assert!((inv_normal_cdf(0.999) - 3.09).abs() < 0.02);
        // Φ⁻¹(0.025) ≈ -1.96
        assert!((inv_normal_cdf(0.025) - (-1.96)).abs() < 0.01);
    }

    #[test]
    fn stats_from_unit_vectors() {
        // 4 vectors in 2d, already normalized
        let vecs: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let stats = AdaEfStats::from_normalized_vectors(&vecs, 4, 2);
        assert_eq!(stats.dim, 2);
        // Mean should be (0, 0)
        assert!(stats.mean[0].abs() < 1e-10);
        assert!(stats.mean[1].abs() < 1e-10);
        // Variance should be 2/3 (sample variance of [1,0,-1,0])
        assert!((stats.variance[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((stats.variance[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn table_lookup_ordering() {
        let table = AdaEfTable::new(
            &[
                (10.0, 50, 4, 8),   // easy: score >= 10
                (5.0, 100, 4, 16),  // medium: score >= 5
                (2.0, 150, 8, 16),  // hard: score >= 2
            ],
            AdaEfParams { ef: 200, stall_limit: 0, drain_budget: 0 },
        );

        let easy = table.lookup(15.0);
        assert_eq!(easy.ef, 50);

        let medium = table.lookup(7.0);
        assert_eq!(medium.ef, 100);

        let hard = table.lookup(3.0);
        assert_eq!(hard.ef, 150);

        let hardest = table.lookup(1.0);
        assert_eq!(hardest.ef, 200); // floor
    }

    #[test]
    fn estimate_produces_valid_params() {
        let vecs: Vec<f32> = vec![
            0.6, 0.8,
            0.8, 0.6,
            -0.6, 0.8,
            -0.8, 0.6,
        ];
        let stats = AdaEfStats::from_normalized_vectors(&vecs, 4, 2);

        let table = AdaEfTable::new(
            &[
                (5.0, 50, 4, 8),
                (1.0, 100, 4, 16),
            ],
            AdaEfParams { ef: 200, stall_limit: 0, drain_budget: 0 },
        );

        let query = vec![0.6f32, 0.8];
        let seed_distances = vec![0.0f32, 0.04, 0.96, 1.04]; // cosine distances

        let params = estimate_ada_ef(&seed_distances, &stats, &query, &table);
        // Should produce valid params
        assert!(params.ef >= 50);
    }
}
