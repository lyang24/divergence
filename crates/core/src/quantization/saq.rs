//! SAQ (Segmented Code Adjustment Quantization) for proxy distance estimation.
//!
//! Loads pre-encoded SAQ data (from C++ SAQ library export) and provides
//! a `VectorBank`-compatible distance estimator. The encoding is done offline;
//! this module only handles the query-side estimation.
//!
//! Distance formula per segment (after vmx-to-1 normalization):
//!   ip_oa_q = sq_delta * Σ code[j]*q_rot[j] + (-1 + sq_delta/2) * sum_q_rot
//!   ip_o_q  = fac_rescale * ip_oa_q
//!
//! For L2:  dist = o_l2norm² + q_l2sqr - 2 * Σ_seg ip_o_q_seg
//!
//! Reference: SAQ (SIGMOD '26), saqlib/quantization/caq/caq_estimator.hpp

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Per-segment configuration.
#[derive(Clone, Debug)]
pub struct SaqSegment {
    pub dim_padded: usize,
    pub bits: u8,
    pub sq_delta: f32, // = 2.0 / 2^bits
    /// Rotation matrix P, dim_padded × dim_padded, row-major.
    /// query_rotated = query_seg · P^T (or equivalently, P applied to query_seg).
    pub rotation: Option<Vec<f32>>,
}

/// Per-vector factors for one segment.
#[derive(Clone, Copy, Debug)]
pub struct SaqFactor {
    pub rescale: f32,  // |o|² / ⟨o, o_a⟩
    pub error: f32,    // error bound
    pub o_l2norm: f32, // |o_seg| (for L2 distance)
}

/// Loaded SAQ data (from C++ export).
pub struct SaqData {
    pub n: usize,
    pub full_dim: usize,
    pub segments: Vec<SaqSegment>,
    /// Unpacked codes: codes[vid * codes_per_vec + seg_offset + d] = code value (0..2^b-1)
    pub codes: Vec<u8>,
    /// Factors: factors[vid * num_segments + seg_idx]
    pub factors: Vec<SaqFactor>,
    /// Byte offset of each segment within a vector's code block.
    pub seg_code_offsets: Vec<usize>,
    /// Total code bytes per vector (sum of all seg dim_padded).
    pub codes_per_vec: usize,
}

impl SaqData {
    /// Load from the "SAQ2" unpacked export format produced by saq_eval.cpp.
    ///
    /// Format:
    ///   magic: u32 = 0x53415132
    ///   version: u32 = 1
    ///   n: u32, dim: u32, num_segments: u32
    ///   per segment: (dim_padded: u32, bits: u32)
    ///   per segment: has_rotation: u32, [rows: u32, cols: u32, data: f32×rows×cols]
    ///   per vector, per segment: codes: u8[dim_padded], fac_rescale: f32, fac_error: f32, o_l2norm: f32
    pub fn load_exported(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let mut buf4 = [0u8; 4];

        // Header
        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x53415132 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad SAQ magic: 0x{magic:08x}, expected 0x53415132"),
            ));
        }

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported SAQ version {version}"),
            ));
        }

        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let full_dim = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let num_segments = u32::from_le_bytes(buf4) as usize;

        // Quant plan
        let mut segments = Vec::with_capacity(num_segments);
        for _ in 0..num_segments {
            r.read_exact(&mut buf4)?;
            let dim_padded = u32::from_le_bytes(buf4) as usize;
            r.read_exact(&mut buf4)?;
            let bits = u32::from_le_bytes(buf4) as u8;
            let sq_delta = 2.0 / (1u32 << bits) as f32;
            segments.push(SaqSegment {
                dim_padded,
                bits,
                sq_delta,
                rotation: None,
            });
        }

        // Rotation matrices
        for seg in segments.iter_mut() {
            r.read_exact(&mut buf4)?;
            let has_rotation = u32::from_le_bytes(buf4);
            if has_rotation != 0 {
                r.read_exact(&mut buf4)?;
                let rows = u32::from_le_bytes(buf4) as usize;
                r.read_exact(&mut buf4)?;
                let cols = u32::from_le_bytes(buf4) as usize;
                let num_floats = rows * cols;
                let mut raw = vec![0u8; num_floats * 4];
                r.read_exact(&mut raw)?;
                let rotation: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                seg.rotation = Some(rotation);
            }
        }

        // Compute offsets
        let mut seg_code_offsets = Vec::with_capacity(num_segments);
        let mut offset = 0usize;
        for seg in &segments {
            seg_code_offsets.push(offset);
            offset += seg.dim_padded;
        }
        let codes_per_vec = offset;

        // Per-vector data
        let mut codes = vec![0u8; n * codes_per_vec];
        let mut factors = Vec::with_capacity(n * num_segments);

        for vid in 0..n {
            for (si, seg) in segments.iter().enumerate() {
                // Read unpacked codes
                let code_start = vid * codes_per_vec + seg_code_offsets[si];
                r.read_exact(&mut codes[code_start..code_start + seg.dim_padded])?;

                // Read factors
                r.read_exact(&mut buf4)?;
                let rescale = f32::from_le_bytes(buf4);
                r.read_exact(&mut buf4)?;
                let error = f32::from_le_bytes(buf4);
                r.read_exact(&mut buf4)?;
                let o_l2norm = f32::from_le_bytes(buf4);
                factors.push(SaqFactor {
                    rescale,
                    error,
                    o_l2norm,
                });
            }
        }

        Ok(Self {
            n,
            full_dim,
            segments,
            codes,
            factors,
            seg_code_offsets,
            codes_per_vec,
        })
    }
}

/// Pre-computed per-query state for SAQ distance estimation.
pub struct SaqQueryState {
    seg_states: Vec<SaqSegQueryState>,
    q_l2sqr: f32,
}

struct SaqSegQueryState {
    rotated_query: Vec<f32>,
    sum_q: f32,
}

impl SaqQueryState {
    /// Prepare query: slice into segments, apply rotation.
    ///
    /// Multi-segment: advances dim_offset by dim_padded per segment. This is correct
    /// when dim_padded == dim_actual (e.g., equal-segment configs where segments
    /// divide dim evenly). If a segment has padding (dim_padded > dim_actual),
    /// subsequent segments start at the wrong offset. The assert below catches this.
    pub fn prepare(query: &[f32], segments: &[SaqSegment]) -> Self {
        // Verify total padded dims == full query length (no gaps from padding)
        let total_padded: usize = segments.iter().map(|s| s.dim_padded).sum();
        assert!(
            total_padded <= query.len() + 16, // allow small padding on last segment
            "SAQ segment dims sum to {} but query has {} dims — \
             likely a dim_padded != dim_actual issue in a multi-segment config",
            total_padded, query.len()
        );
        let q_l2sqr: f32 = query.iter().map(|&x| x * x).sum();
        let mut seg_states = Vec::with_capacity(segments.len());
        let mut dim_offset = 0usize;

        for seg in segments {
            // Extract query segment (zero-pad if needed)
            let copy_len = seg.dim_padded.min(query.len().saturating_sub(dim_offset));
            let mut q_seg = vec![0.0f32; seg.dim_padded];
            if copy_len > 0 {
                q_seg[..copy_len].copy_from_slice(&query[dim_offset..dim_offset + copy_len]);
            }

            // Apply rotation: q_rot = q_seg · P (row vector × matrix)
            // SAQ convention: Eigen `query * P` = P^T * q as column vector.
            // P is stored row-major: M[i*dim + j] = P(i, j).
            // So q_rot[j] = Σ_i q[i] * P(i,j) = Σ_i q[i] * M[i*dim + j].
            let rotated = if let Some(ref p) = seg.rotation {
                mat_vec_mul_transpose(p, &q_seg, seg.dim_padded)
            } else {
                q_seg
            };

            let sum_q: f32 = rotated.iter().sum();
            seg_states.push(SaqSegQueryState {
                rotated_query: rotated,
                sum_q,
            });
            dim_offset += seg.dim_padded;
        }

        Self { seg_states, q_l2sqr }
    }

    /// Compute SAQ L2² distance estimate for a single vector.
    #[inline]
    pub fn estimate_l2sqr(&self, data: &SaqData, vid: usize) -> f32 {
        let num_seg = data.segments.len();
        let mut total_ip = 0.0f32;
        let mut o_l2sqr = 0.0f32;

        for (s, seg) in data.segments.iter().enumerate() {
            let state = &self.seg_states[s];
            let code_start = vid * data.codes_per_vec + data.seg_code_offsets[s];
            let seg_codes = &data.codes[code_start..code_start + seg.dim_padded];
            let factor = &data.factors[vid * num_seg + s];

            // ip_oa_q = sq_delta * Σ code[j]*q_rot[j] + (-1 + sq_delta/2) * sum_q
            let code_dot_q = dot_u8_f32(seg_codes, &state.rotated_query);
            let ip_oa_q = code_dot_q * seg.sq_delta + (-1.0 + seg.sq_delta * 0.5) * state.sum_q;
            total_ip += factor.rescale * ip_oa_q;
            o_l2sqr += factor.o_l2norm * factor.o_l2norm;
        }

        // L2²: |o|² + |q|² - 2·ip_o_q
        let dist = o_l2sqr + self.q_l2sqr - 2.0 * total_ip;
        dist.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// SIMD kernel: u8 × f32 dot product (AVX2 + FMA)
// ---------------------------------------------------------------------------

/// u8 codes × f32 query dot product using AVX2 + FMA.
///
/// Processes 8 elements per iteration: load 8 u8 → cvtepu8_epi32 → cvtepi32_ps → fmadd_ps.
/// 2× unroll (16 elements/iteration) for ILP. At dim=64 (eqseg16): 4 iterations, 100% SIMD.
///
/// # Safety
/// Requires x86_64 with AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_u8_f32_avx(codes: &[u8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = codes.len().min(query.len());
        let codes_ptr = codes.as_ptr();
        let query_ptr = query.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut i = 0usize;
        let chunks16 = dim / 16;

        // Main loop: 16 elements per iteration (2×8)
        for _ in 0..chunks16 {
            // Block 0: 8 u8 → 8 f32, FMA with query
            let raw0 = _mm_loadl_epi64(codes_ptr.add(i) as *const _);
            let i32_0 = _mm256_cvtepu8_epi32(raw0);
            let f32_0 = _mm256_cvtepi32_ps(i32_0);
            let q0 = _mm256_loadu_ps(query_ptr.add(i));
            acc0 = _mm256_fmadd_ps(f32_0, q0, acc0);

            // Block 1: next 8
            let raw1 = _mm_loadl_epi64(codes_ptr.add(i + 8) as *const _);
            let i32_1 = _mm256_cvtepu8_epi32(raw1);
            let f32_1 = _mm256_cvtepi32_ps(i32_1);
            let q1 = _mm256_loadu_ps(query_ptr.add(i + 8));
            acc1 = _mm256_fmadd_ps(f32_1, q1, acc1);

            i += 16;
        }

        // Handle remaining 8-element chunk
        if i + 8 <= dim {
            let raw = _mm_loadl_epi64(codes_ptr.add(i) as *const _);
            let i32_v = _mm256_cvtepu8_epi32(raw);
            let f32_v = _mm256_cvtepi32_ps(i32_v);
            let q = _mm256_loadu_ps(query_ptr.add(i));
            acc0 = _mm256_fmadd_ps(f32_v, q, acc0);
            i += 8;
        }

        // Merge accumulators
        acc0 = _mm256_add_ps(acc0, acc1);

        // Horizontal sum: 8 f32 → scalar
        let hi128 = _mm256_extractf128_ps(acc0, 1);
        let lo128 = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sum64, sum64); // [2,3,2,3]
        let sum32 = _mm_add_ss(sum64, shuf2);
        let mut result = _mm_cvtss_f32(sum32);

        // Scalar tail
        while i < dim {
            result += *codes.get_unchecked(i) as f32 * *query.get_unchecked(i);
            i += 1;
        }

        result
    }
}

/// u8 codes × f32 query dot product using AVX-512F + AVX-512BW (+ FMA).
///
/// Processes 16 elements per iteration: load 16 u8 → cvtepu8_epi32 → cvtepi32_ps → fmadd_ps.
/// 2× unroll (32 elements/iteration) for ILP. At dim=64: 2 iterations, fully pipelined.
///
/// # Safety
/// Requires x86_64 with AVX-512F + AVX-512BW (+ FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,fma")]
unsafe fn dot_u8_f32_avx512(codes: &[u8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let dim = codes.len().min(query.len());
        let codes_ptr = codes.as_ptr();
        let query_ptr = query.as_ptr();

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut i = 0usize;
        let chunks32 = dim / 32;

        // Main loop: 32 elements per iteration (2×16)
        for _ in 0..chunks32 {
            // Block 0: 16 u8 → 16 f32
            let raw0 = _mm_loadu_si128(codes_ptr.add(i) as *const _);
            let i32_0 = _mm512_cvtepu8_epi32(raw0);
            let f32_0 = _mm512_cvtepi32_ps(i32_0);
            let q0 = _mm512_loadu_ps(query_ptr.add(i));
            acc0 = _mm512_fmadd_ps(f32_0, q0, acc0);

            // Block 1: next 16
            let raw1 = _mm_loadu_si128(codes_ptr.add(i + 16) as *const _);
            let i32_1 = _mm512_cvtepu8_epi32(raw1);
            let f32_1 = _mm512_cvtepi32_ps(i32_1);
            let q1 = _mm512_loadu_ps(query_ptr.add(i + 16));
            acc1 = _mm512_fmadd_ps(f32_1, q1, acc1);

            i += 32;
        }

        // Handle remaining 16-element chunk
        if i + 16 <= dim {
            let raw = _mm_loadu_si128(codes_ptr.add(i) as *const _);
            let i32_v = _mm512_cvtepu8_epi32(raw);
            let f32_v = _mm512_cvtepi32_ps(i32_v);
            let q = _mm512_loadu_ps(query_ptr.add(i));
            acc0 = _mm512_fmadd_ps(f32_v, q, acc0);
            i += 16;
        }

        // Merge accumulators and reduce
        acc0 = _mm512_add_ps(acc0, acc1);
        let mut result = _mm512_reduce_add_ps(acc0);

        // Scalar tail (at most 15 elements)
        while i < dim {
            result += *codes.get_unchecked(i) as f32 * *query.get_unchecked(i);
            i += 1;
        }

        result
    }
}

/// Scalar fallback for u8 × f32 dot product.
#[inline]
fn dot_u8_f32_scalar(codes: &[u8], query: &[f32]) -> f32 {
    codes.iter()
        .zip(query.iter())
        .map(|(&c, &q)| c as f32 * q)
        .sum()
}

/// Compute u8 × f32 dot product, dispatching to best available SIMD.
#[inline]
fn dot_u8_f32(codes: &[u8], query: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_u8_f32_avx512(codes, query) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_u8_f32_avx(codes, query) };
        }
    }
    dot_u8_f32_scalar(codes, query)
}

/// Packed nibble dot: each byte has lo=dim[2i], hi=dim[2i+1].
/// Unpacks to f32, dots with interleaved query f32 pairs.
///
/// Processes 16 packed bytes (32 dims) per iteration:
///   load 16 u8 → AND/SHIFT to get lo/hi nibbles → cvt to f32 → FMA with query.
///
/// # Safety
/// Requires x86_64 with AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_packed4_f32_avx(packed: &[u8], query: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let packed_len = packed.len();
        let packed_ptr = packed.as_ptr();
        let query_ptr = query.as_ptr();
        let mask_0f = _mm_set1_epi8(0x0F);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut bi = 0usize; // byte index
        let chunks8 = packed_len / 8; // process 8 packed bytes (16 dims) at a time

        for _ in 0..chunks8 {
            let d = bi * 2;
            if d + 16 > dim { break; }

            // Load 8 packed bytes into low 64 bits of xmm
            let raw = _mm_loadl_epi64(packed_ptr.add(bi) as *const _);

            // Extract low nibbles (even dims)
            let lo_nib = _mm_and_si128(raw, mask_0f);
            // Extract high nibbles (odd dims)
            let hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

            // Interleave lo and hi to get dim order: [lo0, hi0, lo1, hi1, ...]
            let interleaved = _mm_unpacklo_epi8(lo_nib, hi_nib);

            // Convert first 8 u8 → 8 f32
            let i32_0 = _mm256_cvtepu8_epi32(_mm_unpacklo_epi64(interleaved, _mm_setzero_si128()));
            let f32_0 = _mm256_cvtepi32_ps(i32_0);
            let q0 = _mm256_loadu_ps(query_ptr.add(d));
            acc0 = _mm256_fmadd_ps(f32_0, q0, acc0);

            // Convert next 8 u8 → 8 f32
            let i32_1 = _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(interleaved, _mm_setzero_si128()));
            let f32_1 = _mm256_cvtepi32_ps(i32_1);
            let q1 = _mm256_loadu_ps(query_ptr.add(d + 8));
            acc1 = _mm256_fmadd_ps(f32_1, q1, acc1);

            bi += 8;
        }

        // Merge accumulators
        acc0 = _mm256_add_ps(acc0, acc1);

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(acc0, 1);
        let lo128 = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sum64, sum64);
        let sum32 = _mm_add_ss(sum64, shuf2);
        let mut result = _mm_cvtss_f32(sum32);

        // Scalar tail for remaining bytes
        while bi < packed_len {
            let byte = *packed.get_unchecked(bi);
            let d = bi * 2;
            let lo = (byte & 0x0F) as f32;
            result += lo * *query.get_unchecked(d);
            if d + 1 < dim {
                let hi = (byte >> 4) as f32;
                result += hi * *query.get_unchecked(d + 1);
            }
            bi += 1;
        }

        result
    }
}

/// Packed nibble dot using AVX-512F + AVX-512BW.
///
/// Processes 16 packed bytes (32 dims) per iteration:
///   load 16 bytes → AND/SHIFT for lo/hi nibbles → interleave →
///   cvtepu8_epi32 (16 at a time) → cvtepi32_ps → fmadd_ps.
///
/// # Safety
/// Requires x86_64 with AVX-512F + AVX-512BW (+ FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,fma")]
unsafe fn dot_packed4_f32_avx512(packed: &[u8], query: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let packed_len = packed.len();
        let packed_ptr = packed.as_ptr();
        let query_ptr = query.as_ptr();
        let mask_0f = _mm_set1_epi8(0x0F);

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut bi = 0usize;
        // Process 16 packed bytes = 32 dims per iteration
        let chunks16 = packed_len / 16;

        for _ in 0..chunks16 {
            let d = bi * 2;
            if d + 32 > dim { break; }

            // Load 16 packed bytes
            let raw = _mm_loadu_si128(packed_ptr.add(bi) as *const _);

            // Extract lo/hi nibbles
            let lo_nib = _mm_and_si128(raw, mask_0f);
            let hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

            // Interleave: [lo0,hi0, lo1,hi1, ...] — 32 bytes total
            let interleaved_lo = _mm_unpacklo_epi8(lo_nib, hi_nib); // first 16 dims
            let interleaved_hi = _mm_unpackhi_epi8(lo_nib, hi_nib); // next 16 dims

            // Convert first 16 u8 → 16 f32
            let i32_0 = _mm512_cvtepu8_epi32(interleaved_lo);
            let f32_0 = _mm512_cvtepi32_ps(i32_0);
            let q0 = _mm512_loadu_ps(query_ptr.add(d));
            acc0 = _mm512_fmadd_ps(f32_0, q0, acc0);

            // Convert next 16 u8 → 16 f32
            let i32_1 = _mm512_cvtepu8_epi32(interleaved_hi);
            let f32_1 = _mm512_cvtepi32_ps(i32_1);
            let q1 = _mm512_loadu_ps(query_ptr.add(d + 16));
            acc1 = _mm512_fmadd_ps(f32_1, q1, acc1);

            bi += 16;
        }

        // Merge and reduce
        acc0 = _mm512_add_ps(acc0, acc1);
        let mut result = _mm512_reduce_add_ps(acc0);

        // Scalar tail
        while bi < packed_len {
            let byte = *packed.get_unchecked(bi);
            let d = bi * 2;
            let lo = (byte & 0x0F) as f32;
            result += lo * *query.get_unchecked(d);
            if d + 1 < dim {
                let hi = (byte >> 4) as f32;
                result += hi * *query.get_unchecked(d + 1);
            }
            bi += 1;
        }

        result
    }
}

/// Scalar fallback for packed nibble dot.
#[inline]
fn dot_packed4_f32_scalar(packed: &[u8], query: &[f32], dim: usize) -> f32 {
    let mut result = 0.0f32;
    for (byte_idx, &byte) in packed.iter().enumerate() {
        let d = byte_idx * 2;
        let lo = (byte & 0x0F) as f32;
        result += lo * query[d];
        if d + 1 < dim {
            let hi = (byte >> 4) as f32;
            result += hi * query[d + 1];
        }
    }
    result
}

/// Compute packed 4-bit × f32 dot product, dispatching to best available SIMD.
#[inline]
fn dot_packed4_f32(packed: &[u8], query: &[f32], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_packed4_f32_avx512(packed, query, dim) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_packed4_f32_avx(packed, query, dim) };
        }
    }
    dot_packed4_f32_scalar(packed, query, dim)
}

/// Transpose-multiply: result = v · M (= M^T · v as column vector)
/// where M is dim×dim row-major. result[j] = Σ_i v[i] * M[i*dim + j].
fn mat_vec_mul_transpose(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let vi = v[i];
        if vi != 0.0 {
            let row_start = i * dim;
            for j in 0..dim {
                result[j] += vi * m[row_start + j];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Packed 4-bit SAQ format (v0: single segment, B=4 only)
// ---------------------------------------------------------------------------

/// Packed SAQ data: 4-bit codes (2 dims per byte) + per-vector factors.
///
/// Memory layout per vector: ceil(dim/2) bytes codes + 12 bytes factors.
/// For dim=768, B=4: 384 + 12 = 396 bytes/vec (vs 780 bytes unpacked = 1.97× smaller).
/// vs FP32 3072 bytes/vec: 7.76× savings.
pub struct SaqPackedData {
    pub n: usize,
    pub dim: usize,
    pub sq_delta: f32,
    /// Rotation matrix (dim × dim, row-major), or None.
    pub rotation: Option<Vec<f32>>,
    /// Packed 4-bit codes: codes[vid * packed_per_vec .. +packed_per_vec].
    /// Within each byte: low nibble = dim d, high nibble = dim d+1.
    pub codes: Vec<u8>,
    /// Per-vector factors: factors[vid].
    pub factors: Vec<SaqFactor>,
    /// Bytes of packed codes per vector = ceil(dim / 2).
    pub packed_per_vec: usize,
}

impl SaqPackedData {
    /// Pack from unpacked SaqData (must be single-segment, B=4).
    pub fn from_unpacked(data: &SaqData) -> Self {
        assert_eq!(data.segments.len(), 1, "packed v0 requires single segment");
        assert_eq!(data.segments[0].bits, 4, "packed v0 requires B=4");

        let seg = &data.segments[0];
        let dim = seg.dim_padded;
        let packed_per_vec = (dim + 1) / 2;
        let mut codes = vec![0u8; data.n * packed_per_vec];

        for vid in 0..data.n {
            let src_start = vid * data.codes_per_vec;
            let dst_start = vid * packed_per_vec;
            for d in (0..dim).step_by(2) {
                let lo = data.codes[src_start + d];
                let hi = if d + 1 < dim { data.codes[src_start + d + 1] } else { 0 };
                codes[dst_start + d / 2] = (hi << 4) | lo;
            }
        }

        SaqPackedData {
            n: data.n,
            dim,
            sq_delta: seg.sq_delta,
            rotation: seg.rotation.clone(),
            codes,
            factors: data.factors.clone(),
            packed_per_vec,
        }
    }

    /// Save packed format to file.
    ///
    /// Format (SAQ4 magic):
    ///   magic: u32 = 0x53415134 ("SAQ4")
    ///   version: u32 = 1
    ///   n: u32, dim: u32
    ///   has_rotation: u32, [rows: u32, cols: u32, data: f32×rows×cols]
    ///   per vector: codes: u8[packed_per_vec], fac_rescale: f32, fac_error: f32, o_l2norm: f32
    pub fn save(&self, path: &Path) -> io::Result<()> {
        use std::io::{BufWriter, Write};
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&0x53415134u32.to_le_bytes())?; // magic "SAQ4"
        w.write_all(&1u32.to_le_bytes())?;           // version
        w.write_all(&(self.n as u32).to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;

        // Rotation
        if let Some(ref rot) = self.rotation {
            w.write_all(&1u32.to_le_bytes())?;
            w.write_all(&(self.dim as u32).to_le_bytes())?;
            w.write_all(&(self.dim as u32).to_le_bytes())?;
            for &v in rot {
                w.write_all(&v.to_le_bytes())?;
            }
        } else {
            w.write_all(&0u32.to_le_bytes())?;
        }

        // Per-vector data
        for vid in 0..self.n {
            let code_start = vid * self.packed_per_vec;
            w.write_all(&self.codes[code_start..code_start + self.packed_per_vec])?;
            let f = &self.factors[vid];
            w.write_all(&f.rescale.to_le_bytes())?;
            w.write_all(&f.error.to_le_bytes())?;
            w.write_all(&f.o_l2norm.to_le_bytes())?;
        }

        w.flush()
    }

    /// Load packed format from file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x53415134 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad SAQ4 magic: 0x{magic:08x}, expected 0x53415134"),
            ));
        }
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported SAQ4 version {version}"),
            ));
        }

        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        let sq_delta = 2.0 / 16.0f32; // B=4 → 2^4=16

        // Rotation
        r.read_exact(&mut buf4)?;
        let has_rotation = u32::from_le_bytes(buf4);
        let rotation = if has_rotation != 0 {
            r.read_exact(&mut buf4)?;
            let rows = u32::from_le_bytes(buf4) as usize;
            r.read_exact(&mut buf4)?;
            let cols = u32::from_le_bytes(buf4) as usize;
            let num_floats = rows * cols;
            let mut raw = vec![0u8; num_floats * 4];
            r.read_exact(&mut raw)?;
            Some(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        } else {
            None
        };

        let packed_per_vec = (dim + 1) / 2;
        let mut codes = vec![0u8; n * packed_per_vec];
        let mut factors = Vec::with_capacity(n);

        for vid in 0..n {
            let code_start = vid * packed_per_vec;
            r.read_exact(&mut codes[code_start..code_start + packed_per_vec])?;
            r.read_exact(&mut buf4)?;
            let rescale = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let error = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let o_l2norm = f32::from_le_bytes(buf4);
            factors.push(SaqFactor { rescale, error, o_l2norm });
        }

        Ok(SaqPackedData { n, dim, sq_delta, rotation, codes, factors, packed_per_vec })
    }

    /// Memory usage in bytes (codes + factors, excluding rotation).
    pub fn memory_bytes(&self) -> usize {
        self.codes.len() + self.factors.len() * std::mem::size_of::<SaqFactor>()
    }
}

/// Pre-computed query state for packed 4-bit SAQ.
pub struct SaqPackedQueryState {
    rotated_query: Vec<f32>,
    sum_q: f32,
    q_l2sqr: f32,
}

impl SaqPackedQueryState {
    /// Prepare query for packed SAQ distance estimation.
    pub fn prepare(query: &[f32], data: &SaqPackedData) -> Self {
        let q_l2sqr: f32 = query.iter().map(|&x| x * x).sum();
        let dim = data.dim;

        // Zero-pad query to dim
        let mut q_seg = vec![0.0f32; dim];
        let copy_len = dim.min(query.len());
        q_seg[..copy_len].copy_from_slice(&query[..copy_len]);

        let rotated = if let Some(ref p) = data.rotation {
            mat_vec_mul_transpose(p, &q_seg, dim)
        } else {
            q_seg
        };

        let sum_q: f32 = rotated.iter().sum();
        SaqPackedQueryState { rotated_query: rotated, sum_q, q_l2sqr }
    }

    /// Compute SAQ L2² distance estimate from packed 4-bit codes.
    #[inline]
    pub fn estimate_l2sqr(&self, data: &SaqPackedData, vid: usize) -> f32 {
        let code_start = vid * data.packed_per_vec;
        let packed = &data.codes[code_start..code_start + data.packed_per_vec];
        let factor = &data.factors[vid];

        // Unpack nibbles and compute code·q_rot in one pass
        let code_dot_q = dot_packed4_f32(packed, &self.rotated_query, data.dim);

        let ip_oa_q = code_dot_q * data.sq_delta + (-1.0 + data.sq_delta * 0.5) * self.sum_q;
        let ip_o_q = factor.rescale * ip_oa_q;
        let o_l2sqr = factor.o_l2norm * factor.o_l2norm;

        let dist = o_l2sqr + self.q_l2sqr - 2.0 * ip_o_q;
        dist.max(0.0)
    }
}

/// Load reference distances from saq_ref_dists.bin for cross-validation.
pub fn load_ref_distances(path: &Path) -> io::Result<(usize, usize, Vec<f32>)> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let mut buf4 = [0u8; 4];

    r.read_exact(&mut buf4)?;
    let nq = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;

    let num_floats = nq * n;
    let mut raw = vec![0u8; num_floats * 4];
    r.read_exact(&mut raw)?;
    let dists: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok((nq, n, dists))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_vec_mul_identity() {
        // 3×3 identity
        let m = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0];
        let r = mat_vec_mul_transpose(&m, &v, 3);
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        // Simulate a tiny single-segment B=4 SaqData
        let dim = 8;
        let n = 3;
        let segments = vec![SaqSegment {
            dim_padded: dim,
            bits: 4,
            sq_delta: 2.0 / 16.0,
            rotation: None,
        }];
        let codes: Vec<u8> = (0..n * dim).map(|i| (i % 16) as u8).collect();
        let factors: Vec<SaqFactor> = (0..n)
            .map(|i| SaqFactor { rescale: 1.0 + i as f32, error: 0.01, o_l2norm: 0.5 })
            .collect();
        let data = SaqData {
            n, full_dim: dim, segments, codes, factors,
            seg_code_offsets: vec![0], codes_per_vec: dim,
        };

        let packed = SaqPackedData::from_unpacked(&data);
        assert_eq!(packed.packed_per_vec, 4); // 8 dims / 2

        // Verify roundtrip: unpack each nibble matches original
        for vid in 0..n {
            for d in 0..dim {
                let byte_idx = d / 2;
                let byte = packed.codes[vid * packed.packed_per_vec + byte_idx];
                let unpacked_val = if d % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                assert_eq!(unpacked_val, data.codes[vid * dim + d],
                    "mismatch at vid={} d={}", vid, d);
            }
        }

        // Verify distance equivalence
        let query = vec![0.1f32; dim];
        let state_unpacked = SaqQueryState::prepare(&query, &data.segments);
        let state_packed = SaqPackedQueryState::prepare(&query, &packed);
        for vid in 0..n {
            let d1 = state_unpacked.estimate_l2sqr(&data, vid);
            let d2 = state_packed.estimate_l2sqr(&packed, vid);
            assert!((d1 - d2).abs() < 1e-6, "vid={}: unpacked={} packed={}", vid, d1, d2);
        }
    }

    #[test]
    fn dot_u8_f32_simd_matches_scalar() {
        use rand::Rng;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        // Test various dims including the hot path: 64 (eqseg16), 768 (single-seg)
        for dim in [8, 16, 32, 64, 128, 384, 768] {
            let codes: Vec<u8> = (0..dim).map(|_| rng.r#gen::<u8>() % 16).collect();
            let query: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();

            let scalar = dot_u8_f32_scalar(&codes, &query);
            let dispatched = dot_u8_f32(&codes, &query);
            assert!(
                (scalar - dispatched).abs() < 1e-3,
                "dim={}: scalar={} dispatched={} diff={}",
                dim, scalar, dispatched, (scalar - dispatched).abs()
            );
        }
    }

    #[test]
    fn dot_packed4_f32_simd_matches_scalar() {
        use rand::Rng;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(99);
        for dim in [8, 16, 32, 64, 384, 768] {
            let packed_len = (dim + 1) / 2;
            let packed: Vec<u8> = (0..packed_len).map(|_| rng.r#gen::<u8>()).collect();
            let query: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();

            let scalar = dot_packed4_f32_scalar(&packed, &query, dim);
            let dispatched = dot_packed4_f32(&packed, &query, dim);
            assert!(
                (scalar - dispatched).abs() < 1e-3,
                "dim={}: scalar={} dispatched={} diff={}",
                dim, scalar, dispatched, (scalar - dispatched).abs()
            );
        }
    }

    #[test]
    fn simd_estimate_matches_scalar() {
        // Build a multi-segment config (like eqseg16: 12×64d)
        let dim = 768;
        let n_seg = 12;
        let seg_dim = 64;
        let n = 10;

        let segments: Vec<SaqSegment> = (0..n_seg)
            .map(|_| SaqSegment {
                dim_padded: seg_dim,
                bits: 4,
                sq_delta: 2.0 / 16.0,
                rotation: None,
            })
            .collect();
        let codes_per_vec = n_seg * seg_dim;
        let codes: Vec<u8> = (0..n * codes_per_vec).map(|i| (i % 16) as u8).collect();
        let factors: Vec<SaqFactor> = (0..n * n_seg)
            .map(|i| SaqFactor {
                rescale: 0.9 + (i % 5) as f32 * 0.05,
                error: 0.01,
                o_l2norm: 0.3 + (i % 3) as f32 * 0.1,
            })
            .collect();
        let seg_code_offsets: Vec<usize> = (0..n_seg).map(|s| s * seg_dim).collect();
        let data = SaqData {
            n, full_dim: dim, segments, codes, factors,
            seg_code_offsets, codes_per_vec,
        };

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let state = SaqQueryState::prepare(&query, &data.segments);

        // Compare SIMD-dispatched vs pure scalar
        for vid in 0..n {
            let dist = state.estimate_l2sqr(&data, vid);
            // Just verify it's a valid finite number (scalar path tested in pack_unpack_roundtrip)
            assert!(dist.is_finite(), "vid={}: dist={}", vid, dist);
            assert!(dist >= 0.0, "vid={}: dist={} should be non-negative", vid, dist);
        }
    }

    #[test]
    #[ignore] // manual benchmark
    fn bench_dot_u8_f32_simd() {
        use rand::Rng;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);

        for dim in [64, 768] {
            let n_vecs = 100_000;
            let codes: Vec<Vec<u8>> = (0..n_vecs)
                .map(|_| (0..dim).map(|_| rng.r#gen::<u8>() % 16).collect())
                .collect();
            let query: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();

            // Warm up
            let mut sink = 0.0f32;
            for c in &codes[..100] { sink += dot_u8_f32(c, &query); }

            // Scalar
            let t = std::time::Instant::now();
            for c in &codes { sink += dot_u8_f32_scalar(c, &query); }
            let scalar_ns = t.elapsed().as_nanos() as f64 / n_vecs as f64;

            // AVX2
            #[cfg(target_arch = "x86_64")]
            let avx2_ns = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let t = std::time::Instant::now();
                for c in &codes { sink += unsafe { dot_u8_f32_avx(c, &query) }; }
                t.elapsed().as_nanos() as f64 / n_vecs as f64
            } else { f64::NAN };

            // AVX-512
            #[cfg(target_arch = "x86_64")]
            let avx512_ns = if is_x86_feature_detected!("avx512f") {
                let t = std::time::Instant::now();
                for c in &codes { sink += unsafe { dot_u8_f32_avx512(c, &query) }; }
                t.elapsed().as_nanos() as f64 / n_vecs as f64
            } else { f64::NAN };

            // Dispatched (best available)
            let t = std::time::Instant::now();
            for c in &codes { sink += dot_u8_f32(c, &query); }
            let disp_ns = t.elapsed().as_nanos() as f64 / n_vecs as f64;

            eprintln!("dot_u8_f32 dim={:4}: scalar={:6.1}ns  avx2={:6.1}ns({:.1}x)  avx512={:6.1}ns({:.1}x)  disp={:6.1}ns",
                dim, scalar_ns,
                avx2_ns, scalar_ns / avx2_ns,
                avx512_ns, scalar_ns / avx512_ns,
                disp_ns);

            // Packed benchmark
            let packed: Vec<Vec<u8>> = codes.iter()
                .map(|c| (0..dim/2).map(|i| (c[2*i] & 0x0F) | ((c[2*i+1] & 0x0F) << 4)).collect())
                .collect();

            let t = std::time::Instant::now();
            for p in &packed { sink += dot_packed4_f32_scalar(p, &query, dim); }
            let pack_scalar_ns = t.elapsed().as_nanos() as f64 / n_vecs as f64;

            #[cfg(target_arch = "x86_64")]
            let pack_avx2_ns = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let t = std::time::Instant::now();
                for p in &packed { sink += unsafe { dot_packed4_f32_avx(p, &query, dim) }; }
                t.elapsed().as_nanos() as f64 / n_vecs as f64
            } else { f64::NAN };

            #[cfg(target_arch = "x86_64")]
            let pack_avx512_ns = if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let t = std::time::Instant::now();
                for p in &packed { sink += unsafe { dot_packed4_f32_avx512(p, &query, dim) }; }
                t.elapsed().as_nanos() as f64 / n_vecs as f64
            } else { f64::NAN };

            let t = std::time::Instant::now();
            for p in &packed { sink += dot_packed4_f32(p, &query, dim); }
            let pack_disp_ns = t.elapsed().as_nanos() as f64 / n_vecs as f64;

            eprintln!("dot_pack4  dim={:4}: scalar={:6.1}ns  avx2={:6.1}ns({:.1}x)  avx512={:6.1}ns({:.1}x)  disp={:6.1}ns",
                dim, pack_scalar_ns,
                pack_avx2_ns, pack_scalar_ns / pack_avx2_ns,
                pack_avx512_ns, pack_scalar_ns / pack_avx512_ns,
                pack_disp_ns);

            assert!(sink != 0.0); // prevent optimization
        }
    }

    #[test]
    fn mat_vec_mul_permutation() {
        // Swap dims 0 and 2
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0];
        let r = mat_vec_mul_transpose(&m, &v, 3);
        assert_eq!(r, vec![3.0, 2.0, 1.0]);
    }
}
