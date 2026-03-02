use std::cell::RefCell;

use crate::MetricType;
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

/// Inner product distance: pure f32, negated.
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
}
