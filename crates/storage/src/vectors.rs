//! Exact vector storage — contiguous f32 array on disk.
//!
//! Vector N starts at byte offset N * dim * 4.
//! Written with std::fs (sync). Read back with io_uring during refinement.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

/// Write vectors.dat: contiguous f32 values in native (LE) byte order.
pub fn write_vectors_file(path: &Path, vectors: &[f32]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Safe: f32 slice → u8 slice, same memory layout (LE on x86)
    let bytes = unsafe {
        std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
    };
    writer.write_all(bytes)?;
    writer.flush()
}

/// Write vectors_int8.dat: FP32 vectors quantized to int8 (scale=127).
///
/// For L2-normalized cosine vectors, each component is in [-1, 1].
/// Encoding: round(v[i] * 127.0), clamped to [-127, 127], stored as i8.
/// Vector N starts at byte offset N * dim.
pub fn write_vectors_int8_file(path: &Path, vectors: &[f32], dim: usize) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    if dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "dim must be > 0"));
    }
    if vectors.len() % dim != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("vectors.len={} not divisible by dim={}", vectors.len(), dim),
        ));
    }

    let mut codes = vec![0i8; dim];
    for chunk in vectors.chunks_exact(dim) {
        for (j, &v) in chunk.iter().enumerate() {
            codes[j] = (v * 127.0).round().clamp(-127.0, 127.0) as i8;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(codes.as_ptr() as *const u8, dim)
        };
        writer.write_all(bytes)?;
    }

    // For O_DIRECT friendliness: pad file length to 512B.
    let bytes_written = vectors.len() as u64; // dim bytes per vector, vectors is already N*dim
    let pad = (512 - (bytes_written % 512)) % 512;
    if pad != 0 {
        let zeros = [0u8; 512];
        writer.write_all(&zeros[..pad as usize])?;
    }
    writer.flush()
}

/// Write vectors_fp16.dat: FP32 vectors quantized to IEEE 754 half-precision (f16).
///
/// For dim=768: 1536 bytes/vec vs FP32's 3072 bytes/vec = 2× smaller.
/// f16 has ~3 decimal digits of precision — much better than int8's ~2.1 digits.
/// Vector N starts at byte offset N * dim * 2.
pub fn write_vectors_fp16_file(path: &Path, vectors: &[f32], dim: usize) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    if dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "dim must be > 0"));
    }
    if vectors.len() % dim != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("vectors.len={} not divisible by dim={}", vectors.len(), dim),
        ));
    }

    let mut codes = vec![0u16; dim];
    for chunk in vectors.chunks_exact(dim) {
        for (j, &v) in chunk.iter().enumerate() {
            codes[j] = f32_to_f16(v);
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(codes.as_ptr() as *const u8, dim * 2)
        };
        writer.write_all(bytes)?;
    }

    // For O_DIRECT friendliness: pad file length to 512B.
    let bytes_written = (vectors.len() / dim) as u64 * (dim as u64 * 2);
    let pad = (512 - (bytes_written % 512)) % 512;
    if pad != 0 {
        let zeros = [0u8; 512];
        writer.write_all(&zeros[..pad as usize])?;
    }
    writer.flush()
}

/// Convert f32 to IEEE 754 half-precision (f16) stored as u16.
/// Handles normal, subnormal, infinity, and NaN.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x7F_FFFF;

    if exp > 15 {
        // Overflow → infinity (preserves sign)
        return (sign | 0x7C00) as u16;
    }
    if exp < -24 {
        // Too small → zero
        return sign as u16;
    }
    if exp < -14 {
        // Subnormal in f16
        let shift = -14 - exp;
        let m = (0x800000 | mantissa) >> (shift + 13);
        return (sign | m) as u16;
    }
    // Normal
    let e16 = ((exp + 15) as u32) << 10;
    let m16 = mantissa >> 13;
    (sign | e16 | m16) as u16
}

/// Convert IEEE 754 half-precision (f16) stored as u16 to f32.
pub fn f16_to_f32(val: u16) -> f32 {
    let sign = ((val as u32) << 16) & 0x8000_0000;
    let exp = ((val >> 10) & 0x1F) as u32;
    let mantissa = (val & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalize
        let mut m = mantissa;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_m = (m & 0x3FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_m);
    }
    if exp == 31 {
        // Infinity or NaN
        return f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13));
    }
    // Normal
    let f32_exp = ((exp as i32 + 127 - 15) as u32) << 23;
    let f32_m = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_m)
}

/// Load vectors.dat back into memory.
pub fn load_vectors(path: &Path, num_vectors: usize, dim: usize) -> io::Result<Vec<f32>> {
    let expected_bytes = num_vectors * dim * 4;
    let mut file = File::open(path)?;
    let mut bytes = vec![0u8; expected_bytes];
    file.read_exact(&mut bytes)?;

    // Safe: u8 vec → f32 vec, properly aligned (Vec guarantees alignment ≥ 4)
    let mut floats = Vec::with_capacity(num_vectors * dim);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_vectors() {
        let dim = 8;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.1).collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.dat");

        write_vectors_file(&path, &vectors).unwrap();
        let loaded = load_vectors(&path, n, dim).unwrap();
        assert_eq!(loaded, vectors);
    }

    #[test]
    fn f16_roundtrip_accuracy() {
        // Test that f16 conversion preserves values in [-1, 1] with ~3 decimal digits
        let test_vals = [0.0f32, 1.0, -1.0, 0.5, -0.5, 0.123, -0.789, 0.001];
        for &v in &test_vals {
            let half = f32_to_f16(v);
            let back = f16_to_f32(half);
            let err = (v - back).abs();
            assert!(
                err < 0.001,
                "f16 roundtrip error too large: {} -> {} (err={})",
                v, back, err
            );
        }
    }

    #[test]
    fn fp16_file_roundtrip() {
        let dim = 8;
        let n = 100;
        // L2-normalized vectors (components in [-1, 1])
        let vectors: Vec<f32> = (0..n * dim)
            .map(|i| ((i as f32 * 0.1).sin() * 0.5))
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_fp16.dat");

        write_vectors_fp16_file(&path, &vectors, dim).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        // File size should be at least n * dim * 2, padded to 512
        let expected = ((n * dim * 2 + 511) / 512) * 512;
        assert_eq!(meta.len(), expected as u64);
    }

    #[test]
    fn file_size_correct() {
        let dim = 32;
        let n = 50;
        let vectors: Vec<f32> = vec![1.0; n * dim];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.dat");

        write_vectors_file(&path, &vectors).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), (n * dim * 4) as u64);
    }
}
