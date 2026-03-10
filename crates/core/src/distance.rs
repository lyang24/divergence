use std::cell::RefCell;

use crate::MetricType;
use crate::quantization::pq::{PqCodebook, PqDistanceTable};
use half::f16;
use half::slice::HalfFloatSliceExt;

/// Computes distances between vectors.
pub trait DistanceComputer: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]);
}

/// Create a distance computer for the given metric.
pub fn create_distance_computer(metric: MetricType) -> Box<dyn DistanceComputer> {
    match metric {
        MetricType::L2 => Box::new(L2Distance),
        MetricType::Cosine => Box::new(CosineDistance),
        MetricType::InnerProduct => Box::new(InnerProductDistance),
    }
}

// ---------------------------------------------------------------------------
// VectorBank — bundles vectors + distance into one interface
// ---------------------------------------------------------------------------

/// Abstraction over vector storage + distance computation.
///
/// Bundles DRAM-resident vectors with their distance function, abstracting
/// over precision (FP32, FP16, int8) and metric type. The search function
/// calls `bank.distance(query, vid)` without knowing the underlying format.
pub trait VectorBank {
    /// Compute distance between an f32 query and the stored vector at `vid`.
    fn distance(&self, query: &[f32], vid: usize) -> f32;
    /// Number of vectors in the bank.
    fn num_vectors(&self) -> usize;
    /// Vector dimensionality.
    fn dimension(&self) -> usize;
}

/// FP32 vector bank: flat f32 array + any DistanceComputer.
pub struct FP32VectorBank<'a> {
    vectors: &'a [f32],
    dim: usize,
    distance: &'a dyn DistanceComputer,
}

impl<'a> FP32VectorBank<'a> {
    pub fn new(vectors: &'a [f32], dim: usize, distance: &'a dyn DistanceComputer) -> Self {
        Self { vectors, dim, distance }
    }
}

impl VectorBank for FP32VectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        let offset = vid * self.dim;
        self.distance.distance(query, &self.vectors[offset..offset + self.dim])
    }
    fn num_vectors(&self) -> usize {
        self.vectors.len() / self.dim
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

/// FP16 vector bank: flat f16 array, fused convert+compute SIMD kernel.
///
/// Query is always f32. Stored vectors are f16. On x86_64 with f16c+FMA,
/// the distance kernel fuses f16→f32 conversion with L2 computation in
/// a single SIMD loop — no scratch buffer, half the cache line loads.
///
/// Fallback: batch convert to scratch buffer (pre-allocated) on platforms
/// without f16c.
pub struct FP16VectorBank<'a> {
    vectors: &'a [f16],
    dim: usize,
    metric: MetricType,
    /// Fallback scratch buffer for non-SIMD path.
    scratch: RefCell<Vec<f32>>,
}

impl<'a> FP16VectorBank<'a> {
    pub fn new(vectors: &'a [f16], dim: usize, metric: MetricType) -> Self {
        Self {
            vectors,
            dim,
            metric,
            scratch: RefCell::new(vec![0.0f32; dim]),
        }
    }
}

impl VectorBank for FP16VectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        let offset = vid * self.dim;
        let v = &self.vectors[offset..offset + self.dim];

        // Fast path: fused SIMD on x86_64 with f16c + FMA (L2 only for now)
        #[cfg(target_arch = "x86_64")]
        if self.metric == MetricType::L2
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: we checked f16c+fma, and f16 has same repr as u16
            return unsafe { l2_fp16_fused_avx(query, v) };
        }

        // Fallback: batch convert + compute
        let mut scratch = self.scratch.borrow_mut();
        v.convert_to_f32_slice(&mut scratch);
        match self.metric {
            MetricType::L2 => l2_f32(query, &scratch),
            MetricType::InnerProduct => ip_f32(query, &scratch),
            MetricType::Cosine => cosine_f32(query, &scratch),
        }
    }
    fn num_vectors(&self) -> usize {
        self.vectors.len() / self.dim
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Fused f16→f32 convert + L2 SIMD kernel (AVX2 + f16c + FMA)
// ---------------------------------------------------------------------------

/// Fused L2 squared distance: f32 query × f16 vector, no scratch buffer.
///
/// Each iteration: load 8×f16 (16 bytes) → vcvtph2ps → sub with query → FMA accumulate.
/// Half the cache line loads compared to f32 vectors.
///
/// # Safety
/// Requires x86_64 with f16c + FMA. Caller must verify with `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn l2_fp16_fused_avx(query: &[f32], vec_fp16: &[f16]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = query.len();
        let q_ptr = query.as_ptr();
        // f16 is repr(transparent) over u16, safe to cast
        let v_ptr = vec_fp16.as_ptr() as *const u16;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks = dim / 16;
        let mut i = 0usize;

        // Main loop: 16 elements per iteration (2×8 for better ILP)
        for _ in 0..chunks {
            // Block 0: 8 elements
            let fp16_0 = _mm_loadu_si128(v_ptr.add(i) as *const _);
            let vec_0 = _mm256_cvtph_ps(fp16_0);
            let q_0 = _mm256_loadu_ps(q_ptr.add(i));
            let diff_0 = _mm256_sub_ps(q_0, vec_0);
            acc0 = _mm256_fmadd_ps(diff_0, diff_0, acc0);

            // Block 1: next 8 elements
            let fp16_1 = _mm_loadu_si128(v_ptr.add(i + 8) as *const _);
            let vec_1 = _mm256_cvtph_ps(fp16_1);
            let q_1 = _mm256_loadu_ps(q_ptr.add(i + 8));
            let diff_1 = _mm256_sub_ps(q_1, vec_1);
            acc1 = _mm256_fmadd_ps(diff_1, diff_1, acc1);

            i += 16;
        }

        // Merge two accumulators
        acc0 = _mm256_add_ps(acc0, acc1);

        // Handle remaining 8-element chunk (if dim % 16 >= 8)
        if i + 8 <= dim {
            let fp16 = _mm_loadu_si128(v_ptr.add(i) as *const _);
            let vec = _mm256_cvtph_ps(fp16);
            let q = _mm256_loadu_ps(q_ptr.add(i));
            let diff = _mm256_sub_ps(q, vec);
            acc0 = _mm256_fmadd_ps(diff, diff, acc0);
            i += 8;
        }

        // Horizontal sum of 8 f32 lanes
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Scalar tail (dim % 8 remaining elements)
        while i < dim {
            let qv = *q_ptr.add(i);
            let vv = f16::from_bits(*v_ptr.add(i)).to_f32();
            let diff = qv - vv;
            result += diff * diff;
            i += 1;
        }

        result
    }
}

/// L2 squared distance: pure f32, auto-vectorizes with SSE/AVX.
#[inline]
fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

// ---------------------------------------------------------------------------
// Hand-written FP32 L2 SIMD kernel (AVX2 + FMA) — fair baseline
// ---------------------------------------------------------------------------

/// Hand-written FP32 L2 squared distance using AVX2 + FMA.
/// Structurally identical to `l2_fp16_fused_avx` but loads f32 directly.
/// This is the fair baseline for comparing FP16 bandwidth savings.
///
/// # Safety
/// Requires x86_64 with AVX2 + FMA. Caller must verify with `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_fp32_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks = dim / 16;
        let mut i = 0usize;

        // Main loop: 16 elements per iteration (2×8 for ILP)
        for _ in 0..chunks {
            let a0 = _mm256_loadu_ps(a_ptr.add(i));
            let b0 = _mm256_loadu_ps(b_ptr.add(i));
            let diff0 = _mm256_sub_ps(a0, b0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
            let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
            let diff1 = _mm256_sub_ps(a1, b1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            i += 16;
        }

        acc0 = _mm256_add_ps(acc0, acc1);

        if i + 8 <= dim {
            let a0 = _mm256_loadu_ps(a_ptr.add(i));
            let b0 = _mm256_loadu_ps(b_ptr.add(i));
            let diff = _mm256_sub_ps(a0, b0);
            acc0 = _mm256_fmadd_ps(diff, diff, acc0);
            i += 8;
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Scalar tail
        while i < dim {
            let diff = *a_ptr.add(i) - *b_ptr.add(i);
            result += diff * diff;
            i += 1;
        }

        result
    }
}

/// FP32 vector bank with hand-written AVX2+FMA kernels.
/// Supports L2, Cosine, and InnerProduct metrics.
pub struct FP32SimdVectorBank<'a> {
    vectors: &'a [f32],
    dim: usize,
    metric: MetricType,
    /// Pre-computed norms for cosine: ||v_i||^2 for each vector.
    /// Avoids recomputing norm_b on every distance call.
    norms: Option<Vec<f32>>,
}

impl<'a> FP32SimdVectorBank<'a> {
    pub fn new(vectors: &'a [f32], dim: usize, metric: MetricType) -> Self {
        let norms = if metric == MetricType::Cosine {
            let n = vectors.len() / dim;
            let mut norms = Vec::with_capacity(n);
            for i in 0..n {
                let offset = i * dim;
                let v = &vectors[offset..offset + dim];
                let norm_sq: f32 = v.iter().map(|x| x * x).sum();
                norms.push(norm_sq);
            }
            Some(norms)
        } else {
            None
        };
        Self { vectors, dim, metric, norms }
    }
}

impl VectorBank for FP32SimdVectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        let offset = vid * self.dim;
        let v = &self.vectors[offset..offset + self.dim];

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return match self.metric {
                MetricType::L2 => unsafe { l2_fp32_avx(query, v) },
                MetricType::Cosine => {
                    let dot = unsafe { dot_fp32_avx(query, v) };
                    let norm_a = unsafe { dot_fp32_avx(query, query) };
                    let norm_b = self.norms.as_ref().unwrap()[vid];
                    let denom = (norm_a * norm_b).sqrt();
                    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
                }
                MetricType::InnerProduct => {
                    let dot = unsafe { dot_fp32_avx(query, v) };
                    -dot
                }
            };
        }

        match self.metric {
            MetricType::L2 => l2_f32(query, v),
            MetricType::Cosine => cosine_f32(query, v),
            MetricType::InnerProduct => ip_f32(query, v),
        }
    }
    fn num_vectors(&self) -> usize {
        self.vectors.len() / self.dim
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Hand-written FP32 dot product SIMD kernel (AVX2 + FMA)
// ---------------------------------------------------------------------------

/// Hand-written dot product using AVX2 + FMA.
/// Used by cosine (dot / norms) and inner product (-dot).
///
/// # Safety
/// Requires x86_64 with AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_fp32_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks = dim / 16;
        let mut i = 0usize;

        for _ in 0..chunks {
            let a0 = _mm256_loadu_ps(a_ptr.add(i));
            let b0 = _mm256_loadu_ps(b_ptr.add(i));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);

            let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
            let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);

            i += 16;
        }

        acc0 = _mm256_add_ps(acc0, acc1);

        if i + 8 <= dim {
            let a0 = _mm256_loadu_ps(a_ptr.add(i));
            let b0 = _mm256_loadu_ps(b_ptr.add(i));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            i += 8;
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        while i < dim {
            result += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Int8 dot product SIMD kernel (AVX2) — AD-5
// ---------------------------------------------------------------------------

/// Hand-written int8 dot product using AVX2.
///
/// Uses the widen-then-madd chain (AD-5):
///   load 32×i8 → split → cvtepi8_epi16 → madd_epi16 → add_epi32
///
/// 4× unroll (128 elements/iteration), 4 independent i32 accumulators
/// to break loop-carried dependency chains. Uses ~12 of 16 ymm registers.
/// Safe to ~66K dimensions (i32 overflow limit).
///
/// # Safety
/// Requires x86_64 with AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = a.len();
        let a_ptr = a.as_ptr() as *const u8;
        let b_ptr = b.as_ptr() as *const u8;

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        let chunks = dim / 128;
        let mut i = 0usize;

        // Main loop: 128 elements per iteration (4×32, AD-5 unroll 4×)
        for _ in 0..chunks {
            // Block 0: 32 elements → acc0
            let va0 = _mm256_lddqu_si256(a_ptr.add(i) as *const _);
            let vb0 = _mm256_lddqu_si256(b_ptr.add(i) as *const _);
            let a0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va0));
            let a0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va0, 1));
            let b0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb0));
            let b0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb0, 1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0_lo, b0_lo));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0_hi, b0_hi));

            // Block 1: next 32 → acc1
            let va1 = _mm256_lddqu_si256(a_ptr.add(i + 32) as *const _);
            let vb1 = _mm256_lddqu_si256(b_ptr.add(i + 32) as *const _);
            let a1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va1));
            let a1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va1, 1));
            let b1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb1));
            let b1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb1, 1));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1_lo, b1_lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1_hi, b1_hi));

            // Block 2: next 32 → acc2
            let va2 = _mm256_lddqu_si256(a_ptr.add(i + 64) as *const _);
            let vb2 = _mm256_lddqu_si256(b_ptr.add(i + 64) as *const _);
            let a2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va2));
            let a2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va2, 1));
            let b2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb2));
            let b2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb2, 1));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(a2_lo, b2_lo));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(a2_hi, b2_hi));

            // Block 3: next 32 → acc3
            let va3 = _mm256_lddqu_si256(a_ptr.add(i + 96) as *const _);
            let vb3 = _mm256_lddqu_si256(b_ptr.add(i + 96) as *const _);
            let a3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va3));
            let a3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va3, 1));
            let b3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb3));
            let b3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb3, 1));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(a3_lo, b3_lo));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(a3_hi, b3_hi));

            i += 128;
        }

        // Handle remaining 32-element chunks
        if i + 32 <= dim {
            let va = _mm256_lddqu_si256(a_ptr.add(i) as *const _);
            let vb = _mm256_lddqu_si256(b_ptr.add(i) as *const _);
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo, b_lo));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_hi, b_hi));
            i += 32;
        }

        // Merge four accumulators
        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc0 = _mm256_add_epi32(acc0, acc2);

        // Horizontal reduction: 8×i32 → scalar i32
        let hi128 = _mm256_extracti128_si256(acc0, 1);
        let lo128 = _mm256_castsi256_si128(acc0);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b_01_00_11_10));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b_00_00_00_01));
        let mut result = _mm_cvtsi128_si32(sum32);

        // Scalar tail
        while i < dim {
            result += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
            i += 1;
        }

        result
    }
}

/// Scalar fallback for int8 dot product.
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

/// Compute int8 dot product, dispatching to SIMD or scalar.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

// ---------------------------------------------------------------------------
// Int8 VectorBank — pre-quantized i8 vectors + SIMD dot kernel
// ---------------------------------------------------------------------------

/// Quantize a single f32 vector to i8. Pre-normalized vectors only (AD-4).
#[inline]
fn quantize_f32_to_i8(src: &[f32], dst: &mut [i8]) {
    for i in 0..src.len() {
        let v = (src[i] * 127.0).round();
        dst[i] = v.clamp(-127.0, 127.0) as i8;
    }
}

/// Int8 vector bank for cheap cosine distance on pre-normalized vectors.
///
/// Stores pre-quantized i8 vectors. Does NOT cache the quantized query —
/// use `Int8PreparedBank` for batch distance calls within a single query.
///
/// The `VectorBank::distance()` implementation quantizes the query inline
/// every call (correct but slow). For search, use `prepare()` to get an
/// `Int8PreparedBank` that pre-quantizes once.
pub struct Int8VectorBank<'a> {
    codes: &'a [i8],
    dim: usize,
}

impl<'a> Int8VectorBank<'a> {
    pub fn new(codes: &'a [i8], dim: usize) -> Self {
        Self { codes, dim }
    }

    /// Prepare a query for efficient batch distance computation.
    /// Quantizes the f32 query to i8 once. The returned `Int8PreparedBank`
    /// implements `VectorBank` and uses the pre-quantized query.
    ///
    /// Lifetime: tied to this bank AND the search call. Create once per query,
    /// drop after search completes. Never reuse across different queries.
    pub fn prepare<'b>(&'b self, query: &[f32]) -> Int8PreparedBank<'b>
    where
        'a: 'b,
    {
        let mut query_i8 = vec![0i8; self.dim];
        quantize_f32_to_i8(query, &mut query_i8);
        Int8PreparedBank {
            codes: self.codes,
            dim: self.dim,
            query_i8,
        }
    }

    /// Prepare from a pre-quantized i8 query (e.g., per-dim quantized externally).
    /// Caller is responsible for ensuring query_i8 was quantized with the same
    /// scheme used for the stored codes.
    pub fn prepare_raw<'b>(&'b self, query_i8: &[i8]) -> Int8PreparedBank<'b>
    where
        'a: 'b,
    {
        debug_assert_eq!(query_i8.len(), self.dim);
        Int8PreparedBank {
            codes: self.codes,
            dim: self.dim,
            query_i8: query_i8.to_vec(),
        }
    }

    /// Raw int8 dot product between a pre-quantized query and stored vector.
    #[inline]
    pub fn distance_raw(&self, query_i8: &[i8], vid: usize) -> i32 {
        let offset = vid * self.dim;
        let v = &self.codes[offset..offset + self.dim];
        dot_i8(query_i8, v)
    }
}

impl VectorBank for Int8VectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        // Slow path: quantizes query every call. Use prepare() for batch.
        let mut query_i8 = vec![0i8; self.dim];
        quantize_f32_to_i8(query, &mut query_i8);
        let dot = self.distance_raw(&query_i8, vid);
        // Raw integer negate — no float division (AD-3 ranking optimization)
        -(dot as f32)
    }

    fn num_vectors(&self) -> usize {
        self.codes.len() / self.dim
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Pre-quantized int8 bank for efficient batch distance within a single query.
///
/// Created via `Int8VectorBank::prepare(query)`. Holds the pre-quantized query
/// and implements `VectorBank` without re-quantizing on each `distance()` call.
///
/// Distance = -(dot_i8 as f32). Pure integer ranking — no float division.
/// The scale factor (127²) is a monotonic constant that doesn't affect ordering.
pub struct Int8PreparedBank<'a> {
    codes: &'a [i8],
    dim: usize,
    query_i8: Vec<i8>,
}

impl Int8PreparedBank<'_> {
    /// Raw int8 dot product (exposed for callers that need the integer value).
    #[inline]
    pub fn distance_raw(&self, vid: usize) -> i32 {
        let offset = vid * self.dim;
        let v = &self.codes[offset..offset + self.dim];
        dot_i8(&self.query_i8, v)
    }
}

impl VectorBank for Int8PreparedBank<'_> {
    fn distance(&self, _query: &[f32], vid: usize) -> f32 {
        -(self.distance_raw(vid) as f32)
    }

    fn num_vectors(&self) -> usize {
        self.codes.len() / self.dim
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

// ─── PQ-backed vector bank (codes in DRAM, LUT per query) ─────────────────

/// PQ vector bank: holds flat PQ codes (N × M bytes) and codebook reference.
///
/// Two usage modes:
/// 1. **Prepared** (efficient): call `prepare(query)` once per query, use the
///    returned `PqPreparedBank` as `VectorBank`. O(M) distance per neighbor.
/// 2. **Direct** (convenient): use `PqVectorBank` itself as `VectorBank`.
///    Caches the LUT in a `RefCell`, auto-rebuilds when query changes.
///    Slightly slower due to RefCell borrow overhead, but compatible with
///    functions that take `&dyn VectorBank` for multiple queries.
pub struct PqVectorBank<'a> {
    /// Flat PQ codes: codes[vid * m .. vid * m + m] = M-byte code for vid.
    codes: &'a [u8],
    /// Number of subquantizers (code length per vector).
    m: usize,
    /// Number of vectors.
    n: usize,
    /// Full vector dimension (for trait compliance).
    dim: usize,
    /// Codebook reference (for building LUT).
    codebook: &'a PqCodebook,
    /// Whether to use inner product (cosine on normalized vectors).
    use_inner_product: bool,
    /// Cached LUT + the query pointer it was built from.
    /// Using raw pointer for query identity check (same slice = same query).
    cached_lut: RefCell<Option<(*const f32, PqDistanceTable)>>,
}

impl<'a> PqVectorBank<'a> {
    pub fn new(
        codes: &'a [u8],
        n: usize,
        dim: usize,
        codebook: &'a PqCodebook,
        use_inner_product: bool,
    ) -> Self {
        let m = codebook.m;
        assert_eq!(codes.len(), n * m, "PQ codes length mismatch: {} != {} * {}", codes.len(), n, m);
        Self { codes, m, n, dim, codebook, use_inner_product, cached_lut: RefCell::new(None) }
    }

    /// Build per-query ADC lookup table. Returns a `PqPreparedBank` that
    /// implements `VectorBank` with O(M) distance per neighbor.
    pub fn prepare<'b>(&'b self, query: &[f32]) -> PqPreparedBank<'b>
    where
        'a: 'b,
    {
        let lut = self.codebook.build_distance_table(query, self.use_inner_product);
        PqPreparedBank {
            codes: self.codes,
            m: self.m,
            n: self.n,
            dim: self.dim,
            lut,
        }
    }

    /// Ensure cached LUT matches the given query. Rebuild if needed.
    fn ensure_lut(&self, query: &[f32]) {
        let query_ptr = query.as_ptr();
        let needs_rebuild = {
            let cached = self.cached_lut.borrow();
            match cached.as_ref() {
                Some((ptr, _)) => *ptr != query_ptr,
                None => true,
            }
        };
        if needs_rebuild {
            let lut = self.codebook.build_distance_table(query, self.use_inner_product);
            *self.cached_lut.borrow_mut() = Some((query_ptr, lut));
        }
    }
}

impl VectorBank for PqVectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        self.ensure_lut(query);
        let cached = self.cached_lut.borrow();
        let (_, lut) = cached.as_ref().unwrap();
        let offset = vid * self.m;
        let code = &self.codes[offset..offset + self.m];
        if lut.use_inner_product() {
            lut.approximate_cosine_distance(code)
        } else {
            lut.approximate_distance(code)
        }
    }

    fn num_vectors(&self) -> usize {
        self.n
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Pre-computed PQ distance bank for a single query.
///
/// Created via `PqVectorBank::prepare(query)`. Holds the per-query ADC lookup
/// table. `distance(_query, vid)` does M table lookups (no float multiply).
///
/// For cosine on L2-normalized vectors with `use_inner_product=true`:
///   distance = 1.0 - sum(lut[m][code[m]])
/// For L2:
///   distance = sum(lut[m][code[m]])
pub struct PqPreparedBank<'a> {
    codes: &'a [u8],
    m: usize,
    n: usize,
    dim: usize,
    lut: PqDistanceTable,
}

impl VectorBank for PqPreparedBank<'_> {
    fn distance(&self, _query: &[f32], vid: usize) -> f32 {
        let offset = vid * self.m;
        let code = &self.codes[offset..offset + self.m];
        if self.lut.use_inner_product() {
            self.lut.approximate_cosine_distance(code)
        } else {
            self.lut.approximate_distance(code)
        }
    }

    fn num_vectors(&self) -> usize {
        self.n
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// SaqVectorBank — SAQ proxy distance (loaded from C++ export)
// ---------------------------------------------------------------------------

/// SAQ proxy distance bank. Caches per-query rotation + precomputed state.
///
/// Distance returned is L2² (same ordering as exact L2² for ranking).
/// For normalized vectors, L2² = 2 - 2·⟨o,q⟩, so ranking is equivalent.
pub struct SaqVectorBank<'a> {
    data: &'a crate::quantization::saq::SaqData,
    cached: RefCell<Option<(*const f32, crate::quantization::saq::SaqQueryState)>>,
}

impl<'a> SaqVectorBank<'a> {
    pub fn new(data: &'a crate::quantization::saq::SaqData) -> Self {
        Self {
            data,
            cached: RefCell::new(None),
        }
    }

    fn ensure_prepared(&self, query: &[f32]) {
        let ptr = query.as_ptr();
        let needs = match self.cached.borrow().as_ref() {
            Some((p, _)) => *p != ptr,
            None => true,
        };
        if needs {
            let state = crate::quantization::saq::SaqQueryState::prepare(query, &self.data.segments);
            *self.cached.borrow_mut() = Some((ptr, state));
        }
    }
}

impl VectorBank for SaqVectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        self.ensure_prepared(query);
        let cached = self.cached.borrow();
        let (_, state) = cached.as_ref().unwrap();
        state.estimate_l2sqr(self.data, vid)
    }

    fn num_vectors(&self) -> usize {
        self.data.n
    }

    fn dimension(&self) -> usize {
        self.data.full_dim
    }
}

//// ---------------------------------------------------------------------------
// SaqPackedVectorBank — packed 4-bit SAQ proxy distance
// ---------------------------------------------------------------------------

/// Packed 4-bit SAQ proxy distance bank.
pub struct SaqPackedVectorBank<'a> {
    data: &'a crate::quantization::saq::SaqPackedData,
    cached: RefCell<Option<(*const f32, crate::quantization::saq::SaqPackedQueryState)>>,
}

impl<'a> SaqPackedVectorBank<'a> {
    pub fn new(data: &'a crate::quantization::saq::SaqPackedData) -> Self {
        Self {
            data,
            cached: RefCell::new(None),
        }
    }

    fn ensure_prepared(&self, query: &[f32]) {
        let ptr = query.as_ptr();
        let needs = match self.cached.borrow().as_ref() {
            Some((p, _)) => *p != ptr,
            None => true,
        };
        if needs {
            let state = crate::quantization::saq::SaqPackedQueryState::prepare(query, self.data);
            *self.cached.borrow_mut() = Some((ptr, state));
        }
    }
}

impl VectorBank for SaqPackedVectorBank<'_> {
    fn distance(&self, query: &[f32], vid: usize) -> f32 {
        self.ensure_prepared(query);
        let cached = self.cached.borrow();
        let (_, state) = cached.as_ref().unwrap();
        state.estimate_l2sqr(self.data, vid)
    }

    fn num_vectors(&self) -> usize {
        self.data.n
    }

    fn dimension(&self) -> usize {
        self.data.dim
    }
}

// Inner product distance: pure f32, negated.
#[inline]
fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    -dot
}

/// Cosine distance: pure f32.
#[inline]
fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

/// Convert f32 vector slice to f16. Allocates.
pub fn fp32_to_fp16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&x| f16::from_f32(x)).collect()
}

pub struct L2Distance;
pub struct CosineDistance;
pub struct InnerProductDistance;

impl DistanceComputer for L2Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

impl DistanceComputer for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

impl DistanceComputer for InnerProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot // negate so smaller = more similar
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors() {
        let d = L2Distance;
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(d.distance(&v, &v), 0.0);
    }

    #[test]
    fn l2_known_distance() {
        let d = L2Distance;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(d.distance(&a, &b), 25.0); // squared L2
    }

    #[test]
    fn cosine_identical_vectors() {
        let d = CosineDistance;
        let v = vec![1.0, 2.0, 3.0];
        assert!((d.distance(&v, &v)).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((d.distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ip_known_value() {
        let d = InnerProductDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(d.distance(&a, &b), -32.0); // -(1*4 + 2*5 + 3*6)
    }

    #[test]
    fn fp16_l2_close_to_fp32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let b_fp16 = fp32_to_fp16(&b);

        let fp32_dist = L2Distance.distance(&a, &b);
        let bank = FP16VectorBank::new(&b_fp16, 4, MetricType::L2);
        let fp16_dist = bank.distance(&a, 0);
        assert!(
            (fp32_dist - fp16_dist).abs() < 0.1,
            "fp32={} fp16={}",
            fp32_dist,
            fp16_dist
        );
    }

    #[test]
    fn fp16_l2_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let a_fp16 = fp32_to_fp16(&a);
        let bank = FP16VectorBank::new(&a_fp16, 3, MetricType::L2);
        let d = bank.distance(&a, 0);
        // Not exactly 0 due to fp16 rounding, but very close
        assert!(d < 1e-4, "d={}", d);
    }

    #[test]
    fn vector_bank_fp32() {
        let vectors = vec![0.0f32, 0.0, 3.0, 4.0]; // 2 vectors, dim=2
        let dist = L2Distance;
        let bank = FP32VectorBank::new(&vectors, 2, &dist);
        assert_eq!(bank.num_vectors(), 2);
        assert_eq!(bank.dimension(), 2);

        let query = vec![0.0, 0.0];
        assert_eq!(bank.distance(&query, 0), 0.0);
        assert_eq!(bank.distance(&query, 1), 25.0);
    }

    #[test]
    fn vector_bank_fp16() {
        let vectors_f32 = vec![0.0f32, 0.0, 3.0, 4.0];
        let vectors_fp16 = fp32_to_fp16(&vectors_f32);
        let bank = FP16VectorBank::new(&vectors_fp16, 2, MetricType::L2);
        assert_eq!(bank.num_vectors(), 2);

        let query = vec![0.0f32, 0.0];
        assert!(bank.distance(&query, 0) < 1e-4);
        assert!((bank.distance(&query, 1) - 25.0).abs() < 0.1);
    }

    #[test]
    fn fp32_to_fp16_roundtrip() {
        let src = vec![1.0f32, -2.5, 0.0, 100.0];
        let fp16 = fp32_to_fp16(&src);
        assert_eq!(fp16.len(), 4);
        for (orig, converted) in src.iter().zip(fp16.iter()) {
            assert!((orig - converted.to_f32()).abs() < 0.01);
        }
    }

    #[test]
    fn fp32_simd_bank_l2_correctness() {
        let vectors = vec![0.0f32, 0.0, 3.0, 4.0];
        let bank = FP32SimdVectorBank::new(&vectors, 2, MetricType::L2);
        let query = vec![0.0, 0.0];
        assert_eq!(bank.distance(&query, 0), 0.0);
        let d1 = bank.distance(&query, 1);
        assert!((d1 - 25.0).abs() < 1e-6, "d1={}", d1);
    }

    #[test]
    fn fp32_simd_bank_cosine_correctness() {
        // Identical vectors → cosine distance = 0
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 vector, dim=4
        let bank = FP32SimdVectorBank::new(&vectors, 4, MetricType::Cosine);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let d = bank.distance(&query, 0);
        assert!(d.abs() < 1e-6, "identical cosine={}", d);

        // Orthogonal vectors → cosine distance = 1
        let vectors = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 vectors, dim=2
        let bank = FP32SimdVectorBank::new(&vectors, 2, MetricType::Cosine);
        let query = vec![1.0, 0.0];
        let d0 = bank.distance(&query, 0); // same direction
        let d1 = bank.distance(&query, 1); // orthogonal
        assert!(d0.abs() < 1e-6, "same direction cosine={}", d0);
        assert!((d1 - 1.0).abs() < 1e-6, "orthogonal cosine={}", d1);

        // Cross-check: SIMD cosine matches scalar cosine for larger dim
        let mut rng = rand::thread_rng();
        let dim = 768;
        let vec_data: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
        let query_data: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
        let scalar_d = CosineDistance.distance(&query_data, &vec_data);
        let bank = FP32SimdVectorBank::new(&vec_data, dim, MetricType::Cosine);
        let simd_d = bank.distance(&query_data, 0);
        assert!(
            (scalar_d - simd_d).abs() < 1e-5,
            "scalar={} simd={} diff={}",
            scalar_d, simd_d, (scalar_d - simd_d).abs()
        );
    }

    #[test]
    fn fp32_simd_bank_ip_correctness() {
        let vectors = vec![4.0f32, 5.0, 6.0];
        let bank = FP32SimdVectorBank::new(&vectors, 3, MetricType::InnerProduct);
        let query = vec![1.0, 2.0, 3.0];
        let d = bank.distance(&query, 0);
        assert!((d - (-32.0)).abs() < 1e-6, "ip={}", d);
    }

    // -----------------------------------------------------------------------
    // Int8 tests
    // -----------------------------------------------------------------------

    #[test]
    fn int8_dot_scalar_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let dot = dot_i8_scalar(&a, &b);
        assert_eq!(dot, 1*5 + 2*6 + 3*7 + 4*8); // 70
    }

    #[test]
    fn int8_dot_avx_matches_scalar() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for dim in [32, 64, 128, 256, 512, 768, 1024] {
            let a: Vec<i8> = (0..dim).map(|_| rng.r#gen::<i8>().max(-127)).collect();
            let b: Vec<i8> = (0..dim).map(|_| rng.r#gen::<i8>().max(-127)).collect();

            let scalar = dot_i8_scalar(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                let avx = unsafe { dot_i8_avx(&a, &b) };
                assert_eq!(
                    avx, scalar,
                    "AVX vs scalar mismatch at dim={}: avx={} scalar={}",
                    dim, avx, scalar
                );
            }
        }
    }

    #[test]
    fn int8_dot_avx_odd_dims() {
        // Test dimensions that aren't multiples of 32 or 64
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for dim in [33, 47, 100, 500, 700] {
            let a: Vec<i8> = (0..dim).map(|_| rng.r#gen::<i8>().max(-127)).collect();
            let b: Vec<i8> = (0..dim).map(|_| rng.r#gen::<i8>().max(-127)).collect();

            let scalar = dot_i8_scalar(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                let avx = unsafe { dot_i8_avx(&a, &b) };
                assert_eq!(avx, scalar, "dim={}: avx={} scalar={}", dim, avx, scalar);
            }
        }
    }

    #[test]
    fn int8_vector_bank_basic() {
        use crate::quantization::ScalarQuantizer;

        // Two pre-normalized vectors, dim=4
        // v0 = [0.6, 0.8, 0, 0] (norm=1), v1 = [0, 0, 0.6, 0.8] (norm=1)
        let dim = 4;
        let sq = ScalarQuantizer::new(dim);
        let v0 = [0.6f32, 0.8, 0.0, 0.0];
        let v1 = [0.0f32, 0.0, 0.6, 0.8];
        let mut codes = vec![0i8; 8];
        sq.encode(&v0, &mut codes[0..4]);
        sq.encode(&v1, &mut codes[4..8]);

        let bank = Int8VectorBank::new(&codes, dim);
        assert_eq!(bank.num_vectors(), 2);

        // Use prepare() for batch — this is the fast path
        let prepared = bank.prepare(&v0);

        // Query = v0 → distance to v0 should be very negative (self-similarity)
        // Query = v0 → distance to v1 should be near 0 (orthogonal)
        let d0 = prepared.distance(&v0, 0);
        let d1 = prepared.distance(&v0, 1);
        // d0 = -(dot of self) = large negative number (high similarity)
        // d1 = -(dot of orthogonal) = near 0
        assert!(d0 < d1, "self should be closer: d0={} d1={}", d0, d1);
        assert!(d0 < -5000.0, "self-distance should be large negative: d0={}", d0);
        assert!(d1.abs() < 100.0, "orthogonal distance should be near 0: d1={}", d1);

        // Verify raw dot values make sense
        let raw0 = prepared.distance_raw(0);
        let raw1 = prepared.distance_raw(1);
        assert!(raw0 > 0, "self dot should be positive: raw0={}", raw0);
        assert!(raw1.abs() < 100, "orthogonal dot should be near 0: raw1={}", raw1);
    }

    #[test]
    fn int8_bank_cosine_correlation() {
        // Verify int8 distances correlate with FP32 cosine distances
        use crate::quantization::{ScalarQuantizer, l2_normalize};
        use rand::Rng;

        let dim = 768;
        let n = 100;
        let mut rng = rand::thread_rng();

        // Generate and normalize vectors
        let mut flat: Vec<f32> = (0..n * dim)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        for chunk in flat.chunks_exact_mut(dim) {
            l2_normalize(chunk);
        }

        // Quantize
        let sq = ScalarQuantizer::new(dim);
        let codes = sq.encode_batch(&flat);

        // Generate and normalize query
        let mut query: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
        l2_normalize(&mut query);

        let int8_bank = Int8VectorBank::new(&codes, dim);
        let prepared = int8_bank.prepare(&query);
        let fp32_bank = FP32SimdVectorBank::new(&flat, dim, MetricType::Cosine);

        // Check rank correlation: int8 and fp32 should agree on ordering
        let mut int8_dists: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, prepared.distance(&query, i)))
            .collect();
        let mut fp32_dists: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, fp32_bank.distance(&query, i)))
            .collect();

        int8_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        fp32_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Top-10 recall: how many of fp32's top-10 appear in int8's top-10
        let fp32_top10: std::collections::HashSet<usize> =
            fp32_dists.iter().take(10).map(|x| x.0).collect();
        let int8_top10: std::collections::HashSet<usize> =
            int8_dists.iter().take(10).map(|x| x.0).collect();
        let recall = fp32_top10.intersection(&int8_top10).count();

        assert!(
            recall >= 7,
            "Int8 top-10 recall vs FP32 too low: {}/10",
            recall
        );
    }

    // -----------------------------------------------------------------------
    // 刀2: Kernel-only microbenchmark — NO IO, NO heap, NO visited
    // -----------------------------------------------------------------------

    /// Pure kernel microbenchmark. 3 variants × 2 dims.
    /// Isolates distance computation from all search overhead.
    /// Run with: cargo test -p divergence-core --release -- kernel_microbench --nocapture
    #[test]
    fn kernel_microbench() {
        use std::time::Instant;

        let dims = [512, 768];
        let n_vectors = 100; // small dataset, all hot in L1/L2
        let n_iters = 100_000;

        eprintln!("\n========== KNIFE 2: KERNEL-ONLY MICROBENCH ==========");
        eprintln!("  {} iterations per measurement, {} vectors", n_iters, n_vectors);
        eprintln!(
            "{:<8} {:<16} {:>10} {:>10} {:>10}",
            "dim", "kernel", "ns/call", "total_ms", "speedup"
        );

        let mut rng = rand::thread_rng();

        for &dim in &dims {
            // Generate data — same data for all kernels
            let query: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
            let flat_f32: Vec<f32> = (0..n_vectors * dim)
                .map(|_| rand::Rng::r#gen::<f32>(&mut rng))
                .collect();
            let flat_f16 = fp32_to_fp16(&flat_f32);

            // --- 1. FP32 autovectorized (current baseline) ---
            let dist = L2Distance;
            let bank_auto = FP32VectorBank::new(&flat_f32, dim, &dist);

            // Warmup
            let mut sink = 0.0f32;
            for vid in 0..n_vectors {
                sink += bank_auto.distance(&query, vid);
            }

            let t = Instant::now();
            for _ in 0..n_iters {
                for vid in 0..n_vectors {
                    sink += bank_auto.distance(&query, vid);
                }
            }
            let auto_ns = t.elapsed().as_nanos() as f64;
            let auto_per_call = auto_ns / (n_iters * n_vectors) as f64;

            // --- 2. FP32 hand-written AVX2+FMA ---
            let bank_simd = FP32SimdVectorBank::new(&flat_f32, dim, MetricType::L2);

            for vid in 0..n_vectors {
                sink += bank_simd.distance(&query, vid);
            }

            let t = Instant::now();
            for _ in 0..n_iters {
                for vid in 0..n_vectors {
                    sink += bank_simd.distance(&query, vid);
                }
            }
            let simd_ns = t.elapsed().as_nanos() as f64;
            let simd_per_call = simd_ns / (n_iters * n_vectors) as f64;

            // --- 3. FP16 fused AVX2+f16c+FMA ---
            let bank_fp16 = FP16VectorBank::new(&flat_f16, dim, MetricType::L2);

            for vid in 0..n_vectors {
                sink += bank_fp16.distance(&query, vid);
            }

            let t = Instant::now();
            for _ in 0..n_iters {
                for vid in 0..n_vectors {
                    sink += bank_fp16.distance(&query, vid);
                }
            }
            let fp16_ns = t.elapsed().as_nanos() as f64;
            let fp16_per_call = fp16_ns / (n_iters * n_vectors) as f64;

            // Print results
            eprintln!(
                "{:<8} {:<16} {:>10.1} {:>10.1} {:>10}",
                dim, "fp32-auto", auto_per_call, auto_ns / 1e6, "-"
            );
            eprintln!(
                "{:<8} {:<16} {:>10.1} {:>10.1} {:>10.2}x",
                dim, "fp32-simd", simd_per_call, simd_ns / 1e6,
                auto_per_call / simd_per_call
            );
            eprintln!(
                "{:<8} {:<16} {:>10.1} {:>10.1} {:>10.2}x  (vs simd: {:.2}x)",
                dim, "fp16-fused", fp16_per_call, fp16_ns / 1e6,
                auto_per_call / fp16_per_call,
                simd_per_call / fp16_per_call
            );
            eprintln!();

            // Prevent dead code elimination
            assert!(sink != 0.0);
        }
    }

    /// Kernel microbench with LARGE dataset (L3 pressure).
    /// 2000 vectors × 768 = 6.1MB FP32, 3.07MB FP16 — L3 territory.
    /// This isolates the bandwidth benefit of FP16.
    #[test]
    fn kernel_microbench_l3() {
        use std::time::Instant;

        let dim = 768;
        let n_vectors = 2000; // ~6MB FP32, ~3MB FP16
        let n_iters = 1000;   // fewer iters since dataset is large

        eprintln!("\n========== KNIFE 2b: KERNEL MICROBENCH (L3 PRESSURE) ==========");
        eprintln!(
            "  dim={}, {} vectors ({:.1}MB fp32 / {:.1}MB fp16), {} iterations",
            dim, n_vectors,
            (n_vectors * dim * 4) as f64 / 1e6,
            (n_vectors * dim * 2) as f64 / 1e6,
            n_iters
        );
        eprintln!(
            "{:<16} {:>10} {:>10} {:>12}",
            "kernel", "ns/call", "total_ms", "speedup"
        );

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
        let flat_f32: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng))
            .collect();
        let flat_f16 = fp32_to_fp16(&flat_f32);

        let mut sink = 0.0f32;

        // --- FP32 autovectorized ---
        let dist = L2Distance;
        let bank_auto = FP32VectorBank::new(&flat_f32, dim, &dist);
        for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        }
        let auto_ns = t.elapsed().as_nanos() as f64;
        let auto_per_call = auto_ns / (n_iters * n_vectors) as f64;

        // --- FP32 hand-written SIMD ---
        let bank_simd = FP32SimdVectorBank::new(&flat_f32, dim, MetricType::L2);
        for vid in 0..n_vectors { sink += bank_simd.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_simd.distance(&query, vid); }
        }
        let simd_ns = t.elapsed().as_nanos() as f64;
        let simd_per_call = simd_ns / (n_iters * n_vectors) as f64;

        // --- FP16 fused ---
        let bank_fp16 = FP16VectorBank::new(&flat_f16, dim, MetricType::L2);
        for vid in 0..n_vectors { sink += bank_fp16.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_fp16.distance(&query, vid); }
        }
        let fp16_ns = t.elapsed().as_nanos() as f64;
        let fp16_per_call = fp16_ns / (n_iters * n_vectors) as f64;

        eprintln!(
            "{:<16} {:>10.1} {:>10.1} {:>12}",
            "fp32-auto", auto_per_call, auto_ns / 1e6, "-"
        );
        eprintln!(
            "{:<16} {:>10.1} {:>10.1} {:>12.2}x vs auto",
            "fp32-simd", simd_per_call, simd_ns / 1e6,
            auto_per_call / simd_per_call
        );
        eprintln!(
            "{:<16} {:>10.1} {:>10.1} {:>12.2}x vs simd",
            "fp16-fused", fp16_per_call, fp16_ns / 1e6,
            simd_per_call / fp16_per_call
        );

        assert!(sink != 0.0);
    }

    /// Cosine-specific microbench: auto vs SIMD, dim=768.
    /// This is the metric we actually use in production.
    #[test]
    fn kernel_microbench_cosine() {
        use std::time::Instant;

        let dim = 768;
        let n_vectors = 2000;
        let n_iters = 1000;

        eprintln!("\n========== COSINE KERNEL MICROBENCH (dim={}, n={}) ==========", dim, n_vectors);
        eprintln!(
            "{:<16} {:>10} {:>12}",
            "kernel", "ns/call", "speedup"
        );

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
        let flat_f32: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng))
            .collect();

        let mut sink = 0.0f32;

        // Cosine auto
        let dist = CosineDistance;
        let bank_auto = FP32VectorBank::new(&flat_f32, dim, &dist);
        for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        }
        let auto_ns = t.elapsed().as_nanos() as f64;
        let auto_per_call = auto_ns / (n_iters * n_vectors) as f64;

        // Cosine SIMD
        let bank_simd = FP32SimdVectorBank::new(&flat_f32, dim, MetricType::Cosine);
        for vid in 0..n_vectors { sink += bank_simd.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_simd.distance(&query, vid); }
        }
        let simd_ns = t.elapsed().as_nanos() as f64;
        let simd_per_call = simd_ns / (n_iters * n_vectors) as f64;

        eprintln!(
            "{:<16} {:>10.1} {:>12}",
            "cosine-auto", auto_per_call, "-"
        );
        eprintln!(
            "{:<16} {:>10.1} {:>12.2}x",
            "cosine-simd", simd_per_call,
            auto_per_call / simd_per_call
        );

        assert!(sink != 0.0);
    }

    /// Int8 kernel microbenchmark: i8 dot vs FP32 SIMD dot.
    /// This is the real comparison — does int8 beat FP32 hand SIMD?
    #[test]
    fn kernel_microbench_int8() {
        use crate::quantization::{ScalarQuantizer, l2_normalize};
        use std::time::Instant;

        let dim = 768;
        let n_vectors = 2000;
        let n_iters = 1000;

        eprintln!("\n========== INT8 KERNEL MICROBENCH (dim={}, n={}) ==========", dim, n_vectors);
        eprintln!(
            "{:<16} {:>10} {:>12}",
            "kernel", "ns/call", "vs fp32-simd"
        );

        let mut rng = rand::thread_rng();

        // Generate normalized vectors
        let mut flat_f32: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng) * 2.0 - 1.0)
            .collect();
        for chunk in flat_f32.chunks_exact_mut(dim) {
            l2_normalize(chunk);
        }

        // Quantize
        let sq = ScalarQuantizer::new(dim);
        let codes = sq.encode_batch(&flat_f32);

        // Generate and normalize query
        let mut query: Vec<f32> = (0..dim)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng) * 2.0 - 1.0)
            .collect();
        l2_normalize(&mut query);

        let mut sink = 0.0f32;

        // --- FP32 SIMD (cosine via dot) ---
        let bank_fp32 = FP32SimdVectorBank::new(&flat_f32, dim, MetricType::Cosine);
        for vid in 0..n_vectors { sink += bank_fp32.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_fp32.distance(&query, vid); }
        }
        let fp32_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        // --- Int8 SIMD (prepared — no per-call quantization) ---
        let bank_i8 = Int8VectorBank::new(&codes, dim);
        let prepared = bank_i8.prepare(&query);
        for vid in 0..n_vectors { sink += prepared.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += prepared.distance(&query, vid); }
        }
        let i8_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        eprintln!(
            "{:<16} {:>10.1} {:>12}",
            "fp32-simd", fp32_per_call, "1.00x"
        );
        eprintln!(
            "{:<16} {:>10.1} {:>12.2}x",
            "int8-simd", i8_per_call,
            fp32_per_call / i8_per_call
        );

        assert!(sink != 0.0);
    }

    /// Regression gate: FP32 SIMD must not regress beyond threshold.
    /// If this fails, someone broke the SIMD kernels.
    #[test]
    fn simd_regression_gate() {
        use std::time::Instant;

        let dim = 768;
        let n_vectors = 100;
        let n_iters = 50_000;

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rand::Rng::r#gen::<f32>(&mut rng)).collect();
        let flat: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng))
            .collect();

        let mut sink = 0.0f32;

        // L2 SIMD
        let bank_l2 = FP32SimdVectorBank::new(&flat, dim, MetricType::L2);
        for vid in 0..n_vectors { sink += bank_l2.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_l2.distance(&query, vid); }
        }
        let l2_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        // Cosine SIMD
        let bank_cos = FP32SimdVectorBank::new(&flat, dim, MetricType::Cosine);
        for vid in 0..n_vectors { sink += bank_cos.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_cos.distance(&query, vid); }
        }
        let cos_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        // L2 autovectorized for sanity baseline
        let dist = L2Distance;
        let bank_auto = FP32VectorBank::new(&flat, dim, &dist);
        for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_auto.distance(&query, vid); }
        }
        let auto_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        eprintln!("\n========== REGRESSION GATE (dim={}) ==========", dim);
        eprintln!("  L2 SIMD:     {:.1} ns/call", l2_per_call);
        eprintln!("  Cosine SIMD: {:.1} ns/call", cos_per_call);
        eprintln!("  L2 auto:     {:.1} ns/call", auto_per_call);
        eprintln!("  SIMD speedup: {:.1}x", auto_per_call / l2_per_call);

        // Gate: kernel-only (L1 hot), SIMD must be ≥5x faster than autovectorized
        // (We measured 6-9x; 5x catches regressions without false positives on CI)
        assert!(
            auto_per_call / l2_per_call >= 5.0,
            "REGRESSION: L2 SIMD only {:.1}x faster than auto (expected >= 5x). \
             L2 SIMD={:.0}ns auto={:.0}ns",
            auto_per_call / l2_per_call, l2_per_call, auto_per_call
        );

        // Gate: Cosine SIMD must be faster than auto too
        // (Cosine has extra norm computation, so lower bar)
        let cos_dist = CosineDistance;
        let bank_cos_auto = FP32VectorBank::new(&flat, dim, &cos_dist);
        for vid in 0..n_vectors { sink += bank_cos_auto.distance(&query, vid); }
        let t = Instant::now();
        for _ in 0..n_iters {
            for vid in 0..n_vectors { sink += bank_cos_auto.distance(&query, vid); }
        }
        let cos_auto_per_call = t.elapsed().as_nanos() as f64 / (n_iters * n_vectors) as f64;

        assert!(
            cos_auto_per_call / cos_per_call >= 2.0,
            "REGRESSION: Cosine SIMD only {:.1}x faster than auto (expected >= 2x). \
             SIMD={:.0}ns auto={:.0}ns",
            cos_auto_per_call / cos_per_call, cos_per_call, cos_auto_per_call
        );

        assert!(sink != 0.0);
    }

    #[test]
    fn pq_vector_bank_basic() {
        // 256 vectors, 16 dims, PQ4 (4 subspaces, 4 dims each)
        let n = 256;
        let dim = 16;
        let m = 4;
        let mut data = vec![0.0f32; n * dim];
        let mut rng = 42u64;
        for i in 0..n * dim {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data[i] = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let cb = PqCodebook::train(&data, n, dim, m, 15, 42);
        let codes = cb.encode_all(&data, n);

        // Test PqVectorBank (auto-LUT caching)
        let bank = PqVectorBank::new(&codes, n, dim, &cb, false);
        let query = &data[0..dim];

        // Self-distance should be small
        let self_dist = bank.distance(query, 0);
        assert!(self_dist < 0.5, "PQ self-distance too large: {self_dist}");

        // Distance to a far vector should be larger
        let far_dist = bank.distance(query, n / 2);
        // (not guaranteed larger for random data, but check it returns a finite value)
        assert!(far_dist.is_finite());

        // num_vectors / dimension
        assert_eq!(bank.num_vectors(), n);
        assert_eq!(bank.dimension(), dim);

        // Test PqPreparedBank
        let prepared = bank.prepare(query);
        let self_dist2 = prepared.distance(query, 0);
        assert!((self_dist - self_dist2).abs() < 1e-6, "Prepared vs direct mismatch");
    }

    #[test]
    fn pq_vector_bank_lut_caching() {
        // Verify that PqVectorBank rebuilds LUT when query changes
        let n = 256;
        let dim = 8;
        let m = 2;
        let mut data = vec![0.0f32; n * dim];
        let mut rng = 123u64;
        for i in 0..n * dim {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data[i] = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let cb = PqCodebook::train(&data, n, dim, m, 10, 42);
        let codes = cb.encode_all(&data, n);
        let bank = PqVectorBank::new(&codes, n, dim, &cb, false);

        let q1 = &data[0..dim];
        let q2 = &data[dim..2 * dim];

        let d1_v0 = bank.distance(q1, 0);
        let d2_v0 = bank.distance(q2, 0);

        // Different queries should give different distances to the same vector
        // (unless by coincidence, which is astronomically unlikely for random data)
        assert!((d1_v0 - d2_v0).abs() > 1e-6,
            "LUT caching broken: same distance for different queries ({d1_v0} vs {d2_v0})");
    }
}
