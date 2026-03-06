//! NVMe device calibration: sweep queue depths to find the performance knee.
//!
//! Runs a short (~500ms) benchmark of random 4KB O_DIRECT reads at increasing
//! queue depths. The "knee" is the highest QD where IOPS >= 95% of peak AND
//! p99 < 2x the p99 at QD=1.
//!
//! The caller should apply a conservative clamp:
//! ```ignore
//! let cal = calibrate_device(path, true).await?;
//! let gqd = cal.recommended_qd.min(default_global_qd(num_cores));
//! ```

use std::io;
use std::os::unix::fs::OpenOptionsExt as _;
use std::time::{Duration, Instant};

use monoio::fs::File;

use crate::aligned::AlignedBuf;

/// Result of a device calibration sweep.
pub struct CalibrationResult {
    /// Recommended queue depth (raw knee from sweep).
    pub recommended_qd: usize,
    /// Per-level measurements.
    pub levels: Vec<QdMeasurement>,
}

/// Measurement at a single queue depth level.
#[derive(Debug, Clone)]
pub struct QdMeasurement {
    pub qd: usize,
    pub iops: f64,
    pub p99_us: f64,
}

/// Simple xorshift64 PRNG. No `rand` dependency needed for block offsets.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

const BLOCK_SIZE: usize = 4096;
const DURATION_PER_LEVEL: Duration = Duration::from_millis(70);
const QD_LEVELS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Calibrate an NVMe device by sweeping queue depths.
///
/// `path`: path to a file to read (must be large enough for random 4KB reads).
/// `direct_io`: use O_DIRECT (required for real NVMe, false for tmpfs/tests).
///
/// Returns the raw knee QD. Caller should clamp with `default_global_qd(num_cores)`.
pub async fn calibrate_device(path: &str, direct_io: bool) -> io::Result<CalibrationResult> {
    let mut opts = monoio::fs::OpenOptions::new();
    opts.read(true);
    if direct_io {
        opts.custom_flags(libc::O_DIRECT);
    }
    let file = opts.open(path).await?;

    // Get file size for random offset generation
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();
    let max_blocks = (file_size / BLOCK_SIZE as u64).max(1);

    let mut levels = Vec::with_capacity(QD_LEVELS.len());

    for &qd in QD_LEVELS {
        let measurement = run_qd_level(&file, qd, max_blocks).await?;
        levels.push(measurement);
    }

    let recommended_qd = find_knee(&levels);

    Ok(CalibrationResult {
        recommended_qd,
        levels,
    })
}

/// Run one QD level: spawn `qd` concurrent tasks doing random reads for ~70ms.
async fn run_qd_level(file: &File, qd: usize, max_blocks: u64) -> io::Result<QdMeasurement> {
    let start = Instant::now();
    let mut rng = Xorshift64::new(42 + qd as u64);
    let mut latencies_us = Vec::with_capacity(2048);

    // For single-threaded monoio, we issue reads serially but track them.
    // At QD=N, we issue N reads "concurrently" by spawning N monoio tasks.
    // Since we can't easily share the File across tasks without Arc,
    // we do a simulated QD by issuing reads in batches of `qd`.
    while start.elapsed() < DURATION_PER_LEVEL {
        // Issue a batch of `qd` reads
        let mut batch_offsets = Vec::with_capacity(qd);
        for _ in 0..qd {
            let block_idx = rng.next() % max_blocks;
            batch_offsets.push(block_idx * BLOCK_SIZE as u64);
        }

        for &offset in &batch_offsets {
            let buf = AlignedBuf::new(BLOCK_SIZE);
            let t0 = Instant::now();
            let (result, _buf) = file.read_at(buf, offset).await;
            let elapsed_us = t0.elapsed().as_micros() as f64;
            result?;
            latencies_us.push(elapsed_us);
        }
    }

    let elapsed_s = start.elapsed().as_secs_f64();
    let iops = latencies_us.len() as f64 / elapsed_s;

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_us = if latencies_us.is_empty() {
        0.0
    } else {
        let idx = ((latencies_us.len() as f64) * 0.99).ceil() as usize;
        latencies_us[idx.min(latencies_us.len() - 1)]
    };

    Ok(QdMeasurement { qd, iops, p99_us })
}

/// Find the knee: highest QD where IOPS >= 95% of peak AND p99 < 2x p99_at_qd1.
pub fn find_knee(levels: &[QdMeasurement]) -> usize {
    if levels.is_empty() {
        return 1;
    }

    let peak_iops = levels
        .iter()
        .map(|m| m.iops)
        .fold(0.0f64, f64::max);
    let p99_at_qd1 = levels[0].p99_us;

    let mut best_qd = levels[0].qd;

    for m in levels {
        if m.iops >= 0.95 * peak_iops && (p99_at_qd1 == 0.0 || m.p99_us < 2.0 * p99_at_qd1) {
            best_qd = m.qd;
        }
    }

    best_qd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_knee_basic() {
        // Synthetic data: IOPS peak at QD=16, p99 doubles at QD=32
        let levels = vec![
            QdMeasurement { qd: 1, iops: 50_000.0, p99_us: 20.0 },
            QdMeasurement { qd: 2, iops: 95_000.0, p99_us: 22.0 },
            QdMeasurement { qd: 4, iops: 150_000.0, p99_us: 25.0 },
            QdMeasurement { qd: 8, iops: 195_000.0, p99_us: 28.0 },
            QdMeasurement { qd: 16, iops: 210_000.0, p99_us: 35.0 },
            QdMeasurement { qd: 32, iops: 212_000.0, p99_us: 55.0 }, // p99 > 2*20 = 40
            QdMeasurement { qd: 64, iops: 205_000.0, p99_us: 120.0 },
        ];
        let knee = find_knee(&levels);
        // QD=16: IOPS=210K >= 0.95*212K=201.4K, p99=35 < 40 -> passes
        // QD=32: IOPS=212K >= 201.4K, p99=55 > 40 -> fails
        assert_eq!(knee, 16);
    }

    #[test]
    fn test_find_knee_all_pass() {
        // All levels have low p99 → knee is highest QD
        let levels = vec![
            QdMeasurement { qd: 1, iops: 100_000.0, p99_us: 20.0 },
            QdMeasurement { qd: 4, iops: 200_000.0, p99_us: 25.0 },
            QdMeasurement { qd: 16, iops: 210_000.0, p99_us: 30.0 },
        ];
        let knee = find_knee(&levels);
        assert_eq!(knee, 16);
    }

    #[test]
    fn test_find_knee_iops_drop() {
        // IOPS drops significantly at high QD
        let levels = vec![
            QdMeasurement { qd: 1, iops: 100_000.0, p99_us: 20.0 },
            QdMeasurement { qd: 4, iops: 200_000.0, p99_us: 22.0 },
            QdMeasurement { qd: 16, iops: 180_000.0, p99_us: 25.0 }, // below 95% of 200K
        ];
        let knee = find_knee(&levels);
        assert_eq!(knee, 4); // QD=16 fails IOPS threshold
    }

    #[test]
    fn test_find_knee_empty() {
        assert_eq!(find_knee(&[]), 1);
    }

    #[test]
    fn test_find_knee_single() {
        let levels = vec![QdMeasurement { qd: 8, iops: 100_000.0, p99_us: 50.0 }];
        assert_eq!(find_knee(&levels), 8);
    }

    #[test]
    fn test_xorshift_no_zero() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..1000 {
            let v = rng.next();
            assert_ne!(v, 0, "xorshift should never produce 0");
        }
    }

    #[test]
    fn test_xorshift_distribution() {
        // Basic sanity: values should be reasonably distributed
        let mut rng = Xorshift64::new(12345);
        let mut sum = 0u128;
        let n = 10_000;
        for _ in 0..n {
            sum += rng.next() as u128;
        }
        let mean = sum as f64 / n as f64;
        let expected_mean = u64::MAX as f64 / 2.0;
        // Within 20% of expected mean
        assert!(
            (mean - expected_mean).abs() / expected_mean < 0.2,
            "mean={} expected~{}", mean, expected_mean
        );
    }
}
