/// Scalar quantization: f32 → i8 for pre-normalized vectors.
///
/// AD-4: For pre-normalized cosine vectors (all components in [-1, 1]),
/// the quantizer maps directly: code[i] = clamp(round(x[i] * 127), -127, 127).
/// No training step. No scale parameter.
///
/// For non-normalized vectors (future), use percentile-based scale.
pub struct ScalarQuantizer {
    dim: usize,
}

impl ScalarQuantizer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Quantize a single pre-normalized f32 vector to i8.
    /// Caller must ensure `src.len() == dim` and `src` is L2-normalized.
    #[inline]
    pub fn encode(&self, src: &[f32], dst: &mut [i8]) {
        debug_assert_eq!(src.len(), self.dim);
        debug_assert_eq!(dst.len(), self.dim);
        for i in 0..src.len() {
            let v = (src[i] * 127.0).round();
            dst[i] = v.clamp(-127.0, 127.0) as i8;
        }
    }

    /// Quantize a batch of pre-normalized f32 vectors to i8.
    /// `src` is a flat array of `n * dim` floats.
    /// Returns a flat Vec<i8> of `n * dim` codes.
    pub fn encode_batch(&self, src: &[f32]) -> Vec<i8> {
        debug_assert_eq!(src.len() % self.dim, 0);
        let mut dst = vec![0i8; src.len()];
        for chunk in 0..(src.len() / self.dim) {
            let s = chunk * self.dim;
            self.encode(&src[s..s + self.dim], &mut dst[s..s + self.dim]);
        }
        dst
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// L2-normalize a vector in place. Returns the norm before normalization.
pub fn l2_normalize(v: &mut [f32]) -> f32 {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

/// L2-normalize a batch of vectors in place.
/// `vectors` is a flat array of `n * dim` floats.
pub fn l2_normalize_batch(vectors: &mut [f32], dim: usize) {
    debug_assert_eq!(vectors.len() % dim, 0);
    for chunk in vectors.chunks_exact_mut(dim) {
        l2_normalize(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_unit_vector() {
        let sq = ScalarQuantizer::new(3);
        // [1, 0, -1] → [127, 0, -127]
        let src = [1.0f32, 0.0, -1.0];
        let mut dst = [0i8; 3];
        sq.encode(&src, &mut dst);
        assert_eq!(dst, [127, 0, -127]);
    }

    #[test]
    fn encode_clamps_outliers() {
        let sq = ScalarQuantizer::new(2);
        // Values > 1 or < -1 are clamped
        let src = [1.5f32, -2.0];
        let mut dst = [0i8; 2];
        sq.encode(&src, &mut dst);
        assert_eq!(dst, [127, -127]);
    }

    #[test]
    fn encode_normalized_vector() {
        let sq = ScalarQuantizer::new(3);
        // A normalized vector: [1/√3, 1/√3, 1/√3] ≈ [0.577, 0.577, 0.577]
        let v = 1.0 / 3.0f32.sqrt();
        let src = [v, v, v];
        let mut dst = [0i8; 3];
        sq.encode(&src, &mut dst);
        // round(0.577 * 127) = round(73.3) = 73
        assert_eq!(dst, [73, 73, 73]);
    }

    #[test]
    fn encode_batch_roundtrip() {
        let dim = 4;
        let sq = ScalarQuantizer::new(dim);
        let src = [0.5f32, -0.5, 0.25, -0.25, 1.0, -1.0, 0.0, 0.0];
        let codes = sq.encode_batch(&src);
        assert_eq!(codes.len(), 8);
        // First vector: [64, -64, 32, -32] (round(0.5*127)=64, round(0.25*127)=32)
        assert_eq!(codes[0], 64);
        assert_eq!(codes[1], -64);
        assert_eq!(codes[2], 32);
        assert_eq!(codes[3], -32);
        // Second vector: [127, -127, 0, 0]
        assert_eq!(codes[4], 127);
        assert_eq!(codes[5], -127);
        assert_eq!(codes[6], 0);
        assert_eq!(codes[7], 0);
    }

    #[test]
    fn l2_normalize_basic() {
        let mut v = vec![3.0f32, 4.0];
        let norm = l2_normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        // Check unit norm
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        let norm = l2_normalize(&mut v);
        assert_eq!(norm, 0.0);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_batch_basic() {
        let mut data = vec![3.0f32, 4.0, 0.0, 5.0];
        l2_normalize_batch(&mut data, 2);
        // First vector [3,4] → [0.6, 0.8]
        assert!((data[0] - 0.6).abs() < 1e-6);
        assert!((data[1] - 0.8).abs() < 1e-6);
        // Second vector [0,5] → [0, 1]
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }
}
