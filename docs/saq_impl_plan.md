# SAQ Implementation Plan for Divergence (v3)

## Strategy: Full Rust-Native SAQ, Validated via C++ Cross-Check

SAQ (PCA/rotation, DP segmentation, CAQ encoding, distance estimation) will be
implemented natively in Rust. A quick C++ export experiment validates correctness
before we trust the Rust implementation for benchmarks.

**Two tracks running in parallel:**
- **Track A (fast validation)**: C++ export → Rust loader → proxy recall test.
  Answers "does SAQ proxy + refine work?" in days, not weeks.
- **Track B (production)**: Full Rust encoder + estimator.
  Track A's exported data serves as ground truth for Track B's correctness.

---

## SAQ Correct Formulas (from SAQ repo)

### Quantization Range (per vector, per segment)

**Symmetric around zero**:
```
v_mx = max_j(|o[j]|)
v_mi = -v_mx
Δ = (v_mx - v_mi) / 2^b = 2·v_mx / 2^b
code_max = 2^b - 1
```

### Encoding
```
code[j] = floor((o[j] - v_mi) / Δ),  clamped to [0, code_max]
```

### Reconstruction (mid-bin correction)
```
o_a[j] = (code[j] + 0.5) · Δ + v_mi
```

### Per-vector factors (per segment)

Computed from codes in closed form (no explicit reconstruction):
```
ip_o_code = Σ_j code[j] · o[j]
sum_o     = Σ_j o[j]

⟨o, o_a⟩ = ip_o_code · Δ + (v_mi + 0.5·Δ) · sum_o
|o_a|²   = Δ² · Σ code[j]² + (Δ² + 2·Δ·v_mi) · Σ code[j]
           + (0.25·Δ² + Δ·v_mi + v_mi²) · D

fac_rescale = |o|² / ⟨o, o_a⟩
fac_error   = |o|² · 1.9 · sqrt( (|o|²·|o_a|² / ⟨o,o_a⟩² - 1) / (D-1) )
```

### vmx-to-1 normalization (after encoding)

Rescale so v_mx=1, v_mi=-1. Codes unchanged; `fac_rescale` absorbs `v_mx`:
```
sq_delta = 2.0 / 2^b    (constant per segment — same for all vectors)
```

### Distance estimation (accurate, per segment)
```
ip_oa_q = Σ_j code[j]·q_rot[j] · sq_delta + (-1 + sq_delta/2) · sum_q_rot
ip_o_q  = fac_rescale · ip_oa_q
```

Sum across segments: `total_ip = Σ_seg ip_o_q_seg`

For L2: `dist = |o|² + |q|² - 2·total_ip`
For cosine (normalized): `dist = 1 - total_ip`

### Per-segment rotation

Each segment has its own random orthogonal matrix `P` (D_seg × D_seg).
Applied as `o_rotated = o_seg · P` before quantization. Query applies same rotation.
Decorrelates dimensions so uniform quantization is more effective.

### Code adjustment (CAQ)

Greedy coordinate descent, r rounds (default 6). Per-dimension, try code[j] ± 1.
Accept if `⟨o, o_a⟩² / |o_a|²` improves (equivalent to maximizing cosine similarity).
Incremental update: O(1) per trial (update dot product and norm² with delta).

---

## Per-Segment Data Layout

Each segment is an independent CAQ instance:

| Field | Per-vector bytes | Description |
|-------|-----------------|-------------|
| codes | D_seg × b / 8 | Packed b-bit codes (or 1 byte/dim unpacked in v0) |
| ExFactor.rescale | 4 | `fac_rescale = \|o\|²/⟨o,o_a⟩` |
| ExFactor.error | 4 | Error bound for pruning |
| o_l2norm | 4 | `\|o_seg\|` for L2 distance |

Total per vector per segment: `D_seg·b/8 + 12` bytes.

QuantPlanT = `Vec<(dim_len, bits)>` — one entry per segment.

---

## Track A: Quick C++ Validation Experiment

### A1. C++ Export Script (`scripts/export_saq.cpp`)

Small main that calls SAQ encoder on Cohere 100K, dumps binary:

```
Header:
  magic: u32 = 0x53415131 ("SAQ1")
  version: u32 = 1
  n: u32, full_dim: u32, num_segments: u32
  Per segment: dim_len: u32, bits: u8, rotation: f32[dim_len × dim_len]

Per-vector (contiguous):
  Per segment: codes: u8[dim_len], fac_rescale: f32, fac_error: f32, o_l2norm: f32
```

Codes stored unpacked (1 byte/dim) for v0 simplicity.

### A2. Rust Loader (`SaqData::load_exported`)

Reads the C++ binary, populates `SaqData`. This is a temporary code path —
once Track B is done, we load from our own format.

### A3. Correctness Cross-Check

For 10 vectors: C++ distance vs Rust distance, must match to ~1e-5.
For 100 queries: ranking correlation Kendall τ > 0.999.

### A4. Proxy Recall Test (`exp_saq_proxy_recall`)

Oracle test: SAQ distance to all N vectors, measure recall@k in top-R.
Proves SAQ proxy quality independent of graph.

### A5. Full Pipeline Test (`exp_saq_v4_refine`)

SAQ proxy graph traversal + FP32 disk refine vs PQ96 at same R values.

---

## Track B: Full Rust-Native SAQ

### B1. Random Orthogonal Rotation (`crates/core/src/quantization/saq.rs`)

Generate D×D random orthogonal matrix (Gram-Schmidt on random Gaussian columns).
Apply as matrix-vector multiply on segment dimensions.

```rust
/// Generate a random orthogonal matrix of size dim × dim.
/// Uses QR decomposition of a random Gaussian matrix.
pub fn random_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    // Fill dim×dim with N(0,1) using seed
    // QR decomposition (Householder reflections)
    // Return Q as flat Vec<f32>
}
```

No external dependency needed — Householder QR on a random matrix is ~50 LOC.

### B2. PCA Training (`crates/core/src/quantization/pca.rs`)

Note: SAQ uses random rotation per segment, NOT PCA. However, SAQ's DP segmentation
requires per-dimension variances, which come from PCA-ordering the dimensions.
The paper's pipeline: PCA projection → DP segmentation → random rotation per segment.

```rust
pub struct PcaTransform {
    pub dim: usize,
    pub mean: Vec<f32>,           // per-dim mean
    pub components: Vec<f32>,     // dim × dim rotation matrix (eigenvectors)
    pub variances: Vec<f64>,      // per-dim variance (eigenvalues, descending)
}

impl PcaTransform {
    /// Train from n vectors of dimension dim.
    pub fn train(vectors: &[f32], n: usize, dim: usize) -> Self {
        // 1. Compute mean
        // 2. Covariance: (1/n) Σ (x-μ)(x-μ)^T  (dim × dim)
        // 3. Eigendecomposition (symmetric → Jacobi or Householder + QR iteration)
        // 4. Sort by descending eigenvalue
    }

    /// Project a single vector: (v - mean) · components^T
    pub fn project(&self, vector: &[f32]) -> Vec<f32> { ... }
}
```

Eigendecomposition of 768×768: ~100ms with a good implementation. Offline-only.
Options: (a) `nalgebra` crate, (b) hand-rolled symmetric eigen. Start with nalgebra,
can remove later if we want zero deps.

### B3. DP Dimension Segmentation

```rust
/// DP segmentation: partition PCA dimensions into S segments with optimal bit allocation.
///
/// Objective: minimize Σ_s (Σ_{i∈seg_s} σ²_i) / 2^{b_s}
/// Constraint: Σ b_s·|seg_s| = avg_bits × D (total bit budget)
/// Segment boundaries must be multiples of 64 (padding alignment).
pub fn dp_segmentation(
    variances: &[f64],
    dim: usize,
    num_segments: usize,
    avg_bits: f64,
    max_bits: u8,
) -> Vec<(usize, u8)>  // Vec<(dim_len, bits)>
```

Reference: `saqlib/quantization/saq_data.hpp`, `compute_segmentation()`.

### B4. CAQ Encoder

```rust
pub struct CaqEncoder {
    pub dim: usize,
    pub bits: u8,
    pub rotation: Vec<f32>,     // dim × dim orthogonal matrix
    pub adjustment_rounds: u8,  // default 6
}

impl CaqEncoder {
    /// Encode a single segment of a vector.
    /// Input: rotated segment vector (dim floats).
    /// Output: codes (dim u8s, unpacked), fac_rescale, fac_error, o_l2norm.
    pub fn encode(&self, rotated: &[f32]) -> CaqEncoded {
        let dim = self.dim;
        let b = self.bits;

        // 1. Symmetric range
        let v_mx = rotated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let v_mi = -v_mx;
        let delta = (v_mx - v_mi) / (1u32 << b) as f32;
        let code_max = (1u32 << b) - 1;

        // 2. Initial quantization
        let mut codes = vec![0u8; dim];
        for j in 0..dim {
            let c = ((rotated[j] - v_mi) / delta).floor() as i32;
            codes[j] = c.clamp(0, code_max as i32) as u8;
        }

        // 3. Compute running ⟨o, o_a⟩ and |o_a|² from codes (closed form)
        // 4. Code adjustment: r rounds of greedy ±1 per dimension
        //    Accept if ⟨o, o_a_new⟩² · |o_a_old|² > ⟨o, o_a_old⟩² · |o_a_new|²
        //    (incremental update, O(1) per trial)
        // 5. vmx-to-1 normalization: fac_rescale absorbs v_mx
        // 6. Return codes + factors
    }
}
```

### B5. SaqCodebook (ties everything together)

```rust
pub struct SaqCodebook {
    pub full_dim: usize,
    pub pca: PcaTransform,                    // PCA projection (offline)
    pub segments: Vec<SaqSegmentConfig>,       // DP segmentation result
    pub encoders: Vec<CaqEncoder>,            // one per segment (with rotation)
}

pub struct SaqSegmentConfig {
    pub dim_offset: usize,     // start dim in PCA-projected space
    pub dim_len: usize,        // segment dimension
    pub bits: u8,
}

impl SaqCodebook {
    /// Full training pipeline.
    pub fn train(
        vectors: &[f32], n: usize, dim: usize,
        num_segments: usize, avg_bits: f64, adj_rounds: u8, seed: u64,
    ) -> Self {
        // 1. PCA: compute transform, get variances
        // 2. DP segmentation: allocate bits to segments
        // 3. Create per-segment CaqEncoder with random rotation
    }

    /// Encode all vectors.
    pub fn encode_all(&self, vectors: &[f32], n: usize) -> SaqCodes {
        // For each vector:
        //   1. PCA project
        //   2. For each segment: rotate, CAQ encode
        //   3. Store codes + factors
    }
}
```

### B6. SaqCodes + SaqQueryState + SaqVectorBank

Same as Track A's Rust query-side code (Phases 1–3 from v2 plan), but now
also used with Rust-encoded data, not just C++ exports.

### B7. Serialization

```rust
impl SaqCodebook { pub fn save/load(&self, path) -> io::Result<()>; }
impl SaqCodes { pub fn save/load(&self, path) -> io::Result<()>; }
```

---

## File Layout

| File | Contents | ~LOC |
|------|----------|------|
| `crates/core/src/quantization/saq.rs` | SaqCodebook, SaqSegmentConfig, CaqEncoder, SaqCodes, SaqFactor, SaqData, SaqQueryState, encode, estimate_ip, dp_segmentation, serialization | ~600 |
| `crates/core/src/quantization/pca.rs` | PcaTransform (train, project), random_orthogonal_matrix | ~200 |
| `crates/core/src/quantization/mod.rs` | `pub mod saq; pub mod pca;` + re-exports | ~5 |
| `crates/core/src/distance.rs` | SaqVectorBank | ~60 |
| `crates/core/Cargo.toml` | `nalgebra` (optional, for PCA eigen) | ~1 |
| `scripts/export_saq.cpp` | C++ export for Track A validation | ~150 |
| `crates/engine/tests/disk_search.rs` | exp_saq_proxy_recall, exp_saq_v4_refine | ~150 |

---

## DRAM Budget (Cohere 100K, dim=768, avg_bits=4)

| Component | Per-vector | 100K total |
|-----------|-----------|------------|
| Codes (unpacked, 1 byte/dim) | 768 B | 73.2 MB |
| Codes (packed, 4 bits/dim) | 384 B | 36.6 MB |
| Factors (3 × f32 × S=4 segs) | 48 B | 4.6 MB |
| Rotations (shared, S × D_seg²) | — | ~2.3 MB |
| **Total (unpacked)** | | **~80 MB** |
| **Total (packed)** | | **~44 MB** |

vs PQ96: 96 B/vec = 9.4 MB. SAQ is 4–8× more DRAM but should give much
better proxy recall at this dimension.

---

## Simplifications for v0

1. **Unpacked codes** (1 byte/dim) — bit-packing is a later optimization
2. **No progressive distance** (1-bit fast scan → B-bit accurate) — future
3. **No SIMD fastscan** — scalar loops first
4. **No clustering** — single cluster (centroid = 0). Per-cluster SAQ is future
5. **No variance-based pruning** — compute full accurate distance for every candidate
6. **nalgebra for PCA** — can replace with hand-rolled eigen later

---

## Implementation Order

**Track A (validation, ~2 days):**
1. `scripts/export_saq.cpp` — encode Cohere 100K with SAQ C++
2. `saq.rs` — SaqData::load_exported() + SaqQueryState::estimate_ip()
3. `distance.rs` — SaqVectorBank (query-side only)
4. Cross-check: Rust estimator vs C++ reference distances
5. `exp_saq_proxy_recall` + `exp_saq_v4_refine` on EC2

**Track B (production, ~1 week):**
1. `pca.rs` — PcaTransform::train() + project() + random_orthogonal_matrix()
2. `saq.rs` — dp_segmentation() + unit tests
3. `saq.rs` — CaqEncoder::encode() (symmetric range, mid-bin, code adjustment)
4. `saq.rs` — SaqCodebook::train() + encode_all()
5. Cross-check: Rust-encoded data vs C++ export (codes must match, factors to ~1e-5)
6. `saq.rs` — save/load serialization
7. Delete Track A's load_exported(), switch tests to Rust-native encode

Track A gives us the answer fast. Track B makes it production-ready.
Track A's C++ export serves as ground truth for Track B's encoder correctness.
