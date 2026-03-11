//! Per-query profiling: SearchPerfContext, SearchGuard, Histogram, QueryRecorder.
//!
//! Design follows RocksDB's PerfContext pattern: per-core counters with
//! reset→operate→snapshot lifecycle. SearchGuard provides RAII recording
//! so future PRs cannot skip histogram updates.
//!
//! Contracts:
//! - Cell<u64> / plain u64 only — no atomics (monoio is single-threaded per core)
//! - PerfLevel is per-core static config, not global atomic
//! - Counters always on (near-zero overhead); timing sampled via PerfLevel

use std::cell::{Cell, RefCell};
use std::fmt;

// ---------------------------------------------------------------------------
// PerfLevel
// ---------------------------------------------------------------------------

/// Profiling level. Set per-core at startup, not changed per-query.
/// Not a global atomic — each core's config is independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerfLevel {
    /// Counters only: blocks_read, distance_computes, etc.
    /// Near-zero overhead (~1ns per increment).
    CountOnly = 0,
    /// Counters + wall-clock timing of IO wait and compute phases.
    /// ~20ns per Instant::now() call, ~2-3 calls per expansion.
    EnableTime = 1,
}

// ---------------------------------------------------------------------------
// SearchPerfContext — per-query counters and timers
// ---------------------------------------------------------------------------

/// Per-query performance counters. Reset before each query, accumulated
/// during search, snapshotted after.
///
/// 11 fields: 7 counters (always on) + 4 timers (PerfLevel::EnableTime).
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchPerfContext {
    // --- Counters (always on) ---
    /// Total adjacency block accesses (get_or_load calls).
    pub blocks_read: u64,
    /// Cache hits (block found READY in pool).
    pub blocks_hit: u64,
    /// Cache misses (required NVMe IO or bypass).
    pub blocks_miss: u64,
    /// Physical NVMe reads (miss loads + prefetch loads + bypass reads).
    /// Unlike blocks_miss, this counts prefetch IO that later becomes a hit.
    pub phys_reads: u64,
    /// Single-flight waits (found LOADING, awaited existing IO).
    pub singleflight_waits: u64,
    /// Candidates expanded (while loop iterations).
    pub expansions: u64,
    /// Distance computations (distance() calls).
    pub distance_computes: u64,
    /// Exact-vector refinements (future: top-R re-ranking).
    pub refine_count: u64,

    // --- Diagnostics (always on, near-zero overhead) ---
    /// Expansions that added ≥1 neighbor to the beam.
    pub useful_expansions: u64,
    /// Expansions that added 0 neighbors (all visited or dominated).
    pub wasted_expansions: u64,
    /// Expansion number (1-indexed) at which the rank-1 result first entered the beam.
    /// 0 means rank-1 result was a seed entry (came from entry_set, no expansion needed).
    pub best_result_at_expansion: u64,
    /// Expansion number at which the FIRST of the final top-k results entered the beam.
    /// 0 means a seed entry was in the final top-k.
    pub first_topk_at_expansion: u64,

    // --- Prefetch diagnostics (always on) ---
    /// Prefetch hints issued by the search loop.
    pub prefetch_issued: u64,
    /// Prefetch hints that resulted in cache hits (block was prefetched and used).
    pub prefetch_consumed: u64,
    /// Sum of inflight IO depth samples (sampled before each get_or_load).
    pub inflight_sum: u64,
    /// Number of inflight depth samples taken.
    pub inflight_samples: u64,
    /// Maximum inflight IO depth observed during this query.
    pub inflight_max: u64,

    // --- Global inflight diagnostics (always on, only when GlobalIoBudget configured) ---
    /// Sum of global inflight depth samples (sampled before each get_or_load).
    pub global_inflight_sum: u64,
    /// Number of global inflight depth samples taken.
    pub global_inflight_samples: u64,
    /// Maximum global inflight depth observed during this query.
    pub global_inflight_max: u64,

    // --- Adaptive stopping diagnostics (always on) ---
    /// Whether this query was stopped early by the stall detector.
    pub stopped_early: bool,
    /// Number of consecutive stalls when search ended.
    pub consecutive_stalls_at_end: u64,
    /// Expansion count at which search stopped (0 if not stopped early).
    pub expansions_at_stop: u64,

    // --- Page-aware scheduling diagnostics (always on) ---
    /// Times a non-top-1 candidate was chosen because its page was resident.
    pub page_sched_hits: u64,

    // --- PQ gating diagnostics (always on) ---
    /// Total neighbors scored with PQ approximate distance.
    pub pq_candidates_scored: u64,
    /// Neighbors that passed the PQ gate (went to exact scoring).
    pub pq_candidates_passed: u64,
    /// Neighbors filtered out by PQ gate (skipped exact scoring).
    pub pq_candidates_filtered: u64,

    // --- Timers (PerfLevel::EnableTime) ---
    /// Wall-clock nanoseconds spent awaiting adjacency block loads.
    pub io_wait_ns: u64,
    /// Wall-clock nanoseconds on distance + heap + decode (full compute phase).
    pub compute_ns: u64,
    /// Wall-clock nanoseconds in distance kernel only (subset of compute_ns).
    /// overhead_ns = compute_ns - dist_ns (derived: decode + heap + visited).
    pub dist_ns: u64,
    /// Wall-clock nanoseconds for exact-distance refinement phase.
    pub refine_ns: u64,
    /// Total bytes actually submitted to kernel for refine reads (honest IO accounting).
    pub refine_bytes: u64,
    /// Wall-clock nanoseconds for entire search (set by SearchGuard on drop).
    pub total_ns: u64,
}

impl fmt::Display for SearchPerfContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let io_pct = if self.total_ns > 0 {
            (self.io_wait_ns as f64 / self.total_ns as f64) * 100.0
        } else {
            0.0
        };
        let dist_pct = if self.total_ns > 0 {
            (self.dist_ns as f64 / self.total_ns as f64) * 100.0
        } else {
            0.0
        };
        let overhead_ns = self.compute_ns.saturating_sub(self.dist_ns);
        let overhead_pct = if self.total_ns > 0 {
            (overhead_ns as f64 / self.total_ns as f64) * 100.0
        } else {
            0.0
        };
        let hit_rate = if self.blocks_read > 0 {
            (self.blocks_hit as f64 / self.blocks_read as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "blocks={} (hit={} miss={} dedup={}) dist_calls={} expand={} | \
             io={:.0}us ({:.0}%) dist={:.0}us ({:.0}%) overhead={:.0}us ({:.0}%) total={:.0}us | \
             hit_rate={:.1}% pf_iss={} pf_con={}",
            self.blocks_read,
            self.blocks_hit,
            self.blocks_miss,
            self.singleflight_waits,
            self.distance_computes,
            self.expansions,
            self.io_wait_ns as f64 / 1000.0,
            io_pct,
            self.dist_ns as f64 / 1000.0,
            dist_pct,
            overhead_ns as f64 / 1000.0,
            overhead_pct,
            self.total_ns as f64 / 1000.0,
            hit_rate,
            self.prefetch_issued,
            self.prefetch_consumed,
        )?;
        if self.stopped_early {
            write!(f, " | STOPPED@{} stalls={}", self.expansions_at_stop, self.consecutive_stalls_at_end)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Histogram — fixed-bucket, log2, no allocation
// ---------------------------------------------------------------------------

/// Fixed-size log2 histogram. 64 buckets, bucket i covers [2^i, 2^(i+1)).
///
/// Properties: no allocation after construction, O(1) record, O(64) percentile,
/// mergeable across cores.
///
/// Resolution: 2x per bucket. For query latencies (10μs–10ms), covers
/// buckets 13–23 with ~10 distinct bins.
pub struct Histogram {
    buckets: [u64; 64],
    count: u64,
    sum: u64,
    min: u64,
    max: u64,
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            buckets: [0; 64],
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    /// Record a value. O(1).
    pub fn record(&mut self, value: u64) {
        let bucket = if value == 0 {
            0
        } else {
            (63 - value.leading_zeros()) as usize
        };
        self.buckets[bucket.min(63)] += 1;
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Compute the p-th percentile (0–100). Returns lower bound of bucket.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((self.count as f64) * p / 100.0).ceil().max(1.0) as u64;
        let mut running = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            running += count;
            if running >= target {
                return if i == 0 { 0 } else { 1u64 << i };
            }
        }
        self.max
    }

    pub fn p50(&self) -> u64 {
        self.percentile(50.0)
    }
    pub fn p99(&self) -> u64 {
        self.percentile(99.0)
    }
    pub fn p999(&self) -> u64 {
        self.percentile(99.9)
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f64 / self.count as f64
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn min_val(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.min
        }
    }

    pub fn max_val(&self) -> u64 {
        self.max
    }

    /// Merge another histogram into this one. Element-wise addition.
    pub fn merge(&mut self, other: &Histogram) {
        for i in 0..64 {
            self.buckets[i] += other.buckets[i];
        }
        self.count += other.count;
        self.sum += other.sum;
        if other.count > 0 {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.buckets = [0; 64];
        self.count = 0;
        self.sum = 0;
        self.min = u64::MAX;
        self.max = 0;
    }
}

// ---------------------------------------------------------------------------
// QueryRecorder — per-core histogram aggregator
// ---------------------------------------------------------------------------

/// Per-core query recorder. Holds histograms for key metrics.
/// Single-threaded (RefCell), lives on one monoio core.
pub struct QueryRecorder {
    total_ns: RefCell<Histogram>,
    io_wait_ns: RefCell<Histogram>,
    dist_ns: RefCell<Histogram>,
    overhead_ns: RefCell<Histogram>,
    blocks_read: RefCell<Histogram>,
    distance_computes: RefCell<Histogram>,
    /// Per-query average local inflight depth histogram.
    inflight_local: RefCell<Histogram>,
    /// Per-query average global inflight depth histogram.
    inflight_global: RefCell<Histogram>,
    query_count: Cell<u64>,
}

impl QueryRecorder {
    pub fn new() -> Self {
        Self {
            total_ns: RefCell::new(Histogram::new()),
            io_wait_ns: RefCell::new(Histogram::new()),
            dist_ns: RefCell::new(Histogram::new()),
            overhead_ns: RefCell::new(Histogram::new()),
            blocks_read: RefCell::new(Histogram::new()),
            distance_computes: RefCell::new(Histogram::new()),
            inflight_local: RefCell::new(Histogram::new()),
            inflight_global: RefCell::new(Histogram::new()),
            query_count: Cell::new(0),
        }
    }

    /// Record a completed query's perf context into histograms.
    pub fn record(&self, ctx: &SearchPerfContext) {
        self.total_ns.borrow_mut().record(ctx.total_ns);
        self.io_wait_ns.borrow_mut().record(ctx.io_wait_ns);
        self.dist_ns.borrow_mut().record(ctx.dist_ns);
        self.overhead_ns
            .borrow_mut()
            .record(ctx.compute_ns.saturating_sub(ctx.dist_ns));
        self.blocks_read.borrow_mut().record(ctx.blocks_read);
        self.distance_computes
            .borrow_mut()
            .record(ctx.distance_computes);
        // Record per-query average inflight depths
        if ctx.inflight_samples > 0 {
            let avg_local = ctx.inflight_sum / ctx.inflight_samples;
            self.inflight_local.borrow_mut().record(avg_local);
        }
        if ctx.global_inflight_samples > 0 {
            let avg_global = ctx.global_inflight_sum / ctx.global_inflight_samples;
            self.inflight_global.borrow_mut().record(avg_global);
        }
        self.query_count.set(self.query_count.get() + 1);
    }

    pub fn query_count(&self) -> u64 {
        self.query_count.get()
    }

    /// Print a summary report.
    pub fn report(&self) -> String {
        let total = self.total_ns.borrow();
        let io = self.io_wait_ns.borrow();
        let dist = self.dist_ns.borrow();
        let overhead = self.overhead_ns.borrow();
        let blocks = self.blocks_read.borrow();
        let dists = self.distance_computes.borrow();
        let ifl_local = self.inflight_local.borrow();
        let ifl_global = self.inflight_global.borrow();

        let io_pct = if total.mean() > 0.0 {
            (io.mean() / total.mean()) * 100.0
        } else {
            0.0
        };
        let dist_pct = if total.mean() > 0.0 {
            (dist.mean() / total.mean()) * 100.0
        } else {
            0.0
        };
        let overhead_pct = if total.mean() > 0.0 {
            (overhead.mean() / total.mean()) * 100.0
        } else {
            0.0
        };

        let mut s = format!(
            "Queries: {}\n\
             Latency (us):  p50={:.0}  p99={:.0}  p999={:.0}  mean={:.0}  max={:.0}\n\
             IO wait (us):  p50={:.0}  p99={:.0}  mean={:.0}  ({:.0}% of total)\n\
             Distance (us): p50={:.0}  p99={:.0}  mean={:.0}  ({:.0}% of total)\n\
             Overhead (us): p50={:.0}  p99={:.0}  mean={:.0}  ({:.0}% of total)\n\
             Blocks/query:  p50={}  p99={}  mean={:.1}\n\
             Dists/query:   p50={}  p99={}  mean={:.1}",
            self.query_count.get(),
            total.p50() as f64 / 1000.0,
            total.p99() as f64 / 1000.0,
            total.p999() as f64 / 1000.0,
            total.mean() / 1000.0,
            total.max_val() as f64 / 1000.0,
            io.p50() as f64 / 1000.0,
            io.p99() as f64 / 1000.0,
            io.mean() / 1000.0,
            io_pct,
            dist.p50() as f64 / 1000.0,
            dist.p99() as f64 / 1000.0,
            dist.mean() / 1000.0,
            dist_pct,
            overhead.p50() as f64 / 1000.0,
            overhead.p99() as f64 / 1000.0,
            overhead.mean() / 1000.0,
            overhead_pct,
            blocks.p50(),
            blocks.p99(),
            blocks.mean(),
            dists.p50(),
            dists.p99(),
            dists.mean(),
        );

        if ifl_local.count() > 0 {
            s.push_str(&format!(
                "\nInflight local: p50={}  p99={}  mean={:.1}  max={}",
                ifl_local.p50(), ifl_local.p99(), ifl_local.mean(), ifl_local.max_val(),
            ));
        }
        if ifl_global.count() > 0 {
            s.push_str(&format!(
                "\nInflight global: p50={}  p99={}  mean={:.1}  max={}",
                ifl_global.p50(), ifl_global.p99(), ifl_global.mean(), ifl_global.max_val(),
            ));
        }

        s
    }

    /// Total latency p50 in microseconds.
    pub fn p50_total_us(&self) -> f64 {
        self.total_ns.borrow().p50() as f64 / 1000.0
    }

    /// Total latency p99 in microseconds.
    pub fn p99_total_us(&self) -> f64 {
        self.total_ns.borrow().p99() as f64 / 1000.0
    }

    /// Total latency p999 in microseconds.
    pub fn p999_total_us(&self) -> f64 {
        self.total_ns.borrow().p999() as f64 / 1000.0
    }

    /// Mean per-query average global inflight depth.
    pub fn global_inflight_mean(&self) -> f64 {
        self.inflight_global.borrow().mean()
    }

    /// Max per-query average global inflight depth.
    pub fn global_inflight_max(&self) -> u64 {
        self.inflight_global.borrow().max_val()
    }

    /// Mean IO wait percentage (io_wait_ns / total_ns).
    pub fn io_wait_pct(&self) -> f64 {
        let total_mean = self.total_ns.borrow().mean();
        if total_mean > 0.0 {
            (self.io_wait_ns.borrow().mean() / total_mean) * 100.0
        } else {
            0.0
        }
    }

    /// Take a lightweight snapshot of current stats and reset for fresh window.
    ///
    /// `io_timing` is from `IoDriver::take_io_timing()`: (sem_wait_ns, device_ns, io_count).
    pub fn take_snapshot(&self, io_timing: (u64, u64, u64)) -> CoreSnapshot {
        let snap = CoreSnapshot {
            queries: self.query_count.get(),
            lat_p50_us: self.p50_total_us(),
            lat_p99_us: self.p99_total_us(),
            lat_p999_us: self.p999_total_us(),
            io_wait_pct: self.io_wait_pct(),
            global_inflight_avg: self.global_inflight_mean(),
            global_inflight_max: self.global_inflight_max(),
            sem_wait_ns: io_timing.0,
            device_ns: io_timing.1,
            io_count: io_timing.2,
            admit_wait_ns: 0,
            admit_count: 0,
            admit_total: 0,
            health_at_snapshot: 0,
        };
        self.reset();
        snap
    }

    /// Reset all histograms (for window boundary).
    pub fn reset(&self) {
        self.total_ns.borrow_mut().reset();
        self.io_wait_ns.borrow_mut().reset();
        self.dist_ns.borrow_mut().reset();
        self.overhead_ns.borrow_mut().reset();
        self.blocks_read.borrow_mut().reset();
        self.distance_computes.borrow_mut().reset();
        self.inflight_local.borrow_mut().reset();
        self.inflight_global.borrow_mut().reset();
        self.query_count.set(0);
    }
}

// ---------------------------------------------------------------------------
// CoreSnapshot — lightweight per-core stats snapshot
// ---------------------------------------------------------------------------

/// Lightweight per-core stats snapshot. Send-safe, no histograms.
/// Produced by `QueryRecorder::take_snapshot()`, consumed by Engine for aggregation.
#[derive(Debug, Clone, Copy)]
pub struct CoreSnapshot {
    pub queries: u64,
    pub lat_p50_us: f64,
    pub lat_p99_us: f64,
    pub lat_p999_us: f64,
    /// io_wait_ns / total_ns as percentage (0–100).
    pub io_wait_pct: f64,
    pub global_inflight_avg: f64,
    pub global_inflight_max: u64,
    pub sem_wait_ns: u64,
    pub device_ns: u64,
    pub io_count: u64,
    /// Cumulative admission wait time in nanoseconds.
    pub admit_wait_ns: u64,
    /// Queries that had non-zero admission wait.
    pub admit_count: u64,
    /// Total queries attempted (including those with zero wait).
    pub admit_total: u64,
    /// EngineHealth value when this snapshot was taken.
    pub health_at_snapshot: u8,
}

// Safety: CoreSnapshot is all Copy scalars, inherently Send+Sync.
unsafe impl Send for CoreSnapshot {}
unsafe impl Sync for CoreSnapshot {}

impl Default for CoreSnapshot {
    fn default() -> Self {
        Self {
            queries: 0,
            lat_p50_us: 0.0,
            lat_p99_us: 0.0,
            lat_p999_us: 0.0,
            io_wait_pct: 0.0,
            global_inflight_avg: 0.0,
            global_inflight_max: 0,
            sem_wait_ns: 0,
            device_ns: 0,
            io_count: 0,
            admit_wait_ns: 0,
            admit_count: 0,
            admit_total: 0,
            health_at_snapshot: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SearchGuard — RAII: reset on new, record on drop
// ---------------------------------------------------------------------------

/// RAII guard for per-query profiling. Created before search, dropped after.
///
/// - `new()`: resets context, optionally starts wall-clock timer
/// - Search function writes to `guard.ctx`
/// - `drop()`: finalizes total_ns, records to QueryRecorder's histograms
///
/// This ensures every query is recorded — future PRs cannot skip it.
pub struct SearchGuard<'a> {
    recorder: &'a QueryRecorder,
    /// Mutable perf context — search function writes counters here.
    pub ctx: SearchPerfContext,
    level: PerfLevel,
    start: Option<std::time::Instant>,
}

impl<'a> SearchGuard<'a> {
    /// Create a new guard. Resets context and starts timer if level >= EnableTime.
    pub fn new(recorder: &'a QueryRecorder, level: PerfLevel) -> Self {
        let start = if level >= PerfLevel::EnableTime {
            Some(std::time::Instant::now())
        } else {
            None
        };
        Self {
            recorder,
            ctx: SearchPerfContext::default(),
            level,
            start,
        }
    }

    /// Get the perf level for conditional timing in search.
    pub fn level(&self) -> PerfLevel {
        self.level
    }

    /// Get an immutable snapshot (before drop). Does NOT record.
    pub fn snapshot(&self) -> SearchPerfContext {
        let mut ctx = self.ctx;
        if let Some(start) = self.start {
            ctx.total_ns = start.elapsed().as_nanos() as u64;
        }
        ctx
    }
}

impl Drop for SearchGuard<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            self.ctx.total_ns = start.elapsed().as_nanos() as u64;
        }
        self.recorder.record(&self.ctx);
    }
}

// ---------------------------------------------------------------------------
// SchedLagTracker — event loop stall detection
// ---------------------------------------------------------------------------

/// Detects monoio event loop stalls by measuring deviation from a 1ms tick.
///
/// Spawn as a background task on the monoio runtime. When p99 query latency
/// spikes, check sched_lag p99:
/// - If sched_lag spikes too → event loop blocked by compute (HOL)
/// - If sched_lag is stable → IO is genuinely slow
pub struct SchedLagTracker {
    histogram: RefCell<Histogram>,
}

impl SchedLagTracker {
    pub fn new() -> Self {
        Self {
            histogram: RefCell::new(Histogram::new()),
        }
    }

    /// Run the 1ms tick loop. Measures scheduling jitter.
    /// Call via `monoio::spawn(tracker.run())` or similar.
    pub async fn run(&self) {
        loop {
            let before = std::time::Instant::now();
            monoio::time::sleep(std::time::Duration::from_millis(1)).await;
            let elapsed_ns = before.elapsed().as_nanos() as u64;
            let lag_ns = elapsed_ns.saturating_sub(1_000_000);
            self.histogram.borrow_mut().record(lag_ns);
        }
    }

    /// Get scheduling lag report.
    pub fn report(&self) -> String {
        let h = self.histogram.borrow();
        format!(
            "Sched lag (us): p50={:.1} p99={:.1} p999={:.1} max={:.1} samples={}",
            h.p50() as f64 / 1000.0,
            h.p99() as f64 / 1000.0,
            h.p999() as f64 / 1000.0,
            h.max_val() as f64 / 1000.0,
            h.count(),
        )
    }

    /// Reset histogram (for window boundary).
    pub fn reset(&self) {
        self.histogram.borrow_mut().reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_basic() {
        let mut h = Histogram::new();
        h.record(100);
        h.record(200);
        h.record(300);
        assert_eq!(h.count(), 3);
        assert_eq!(h.min_val(), 100);
        assert_eq!(h.max_val(), 300);
    }

    #[test]
    fn histogram_zero_value() {
        let mut h = Histogram::new();
        h.record(0);
        assert_eq!(h.count(), 1);
        assert_eq!(h.p50(), 0);
        assert_eq!(h.min_val(), 0);
    }

    #[test]
    fn histogram_percentiles() {
        let mut h = Histogram::new();
        for i in 1..=1000u64 {
            h.record(i);
        }
        // p50 ~500, bucket covers [256, 512) or [512, 1024)
        let p50 = h.p50();
        assert!(p50 >= 256 && p50 <= 512, "p50={}", p50);
        let p99 = h.p99();
        assert!(p99 >= 512 && p99 <= 1024, "p99={}", p99);
    }

    #[test]
    fn histogram_single_value() {
        let mut h = Histogram::new();
        h.record(42);
        assert_eq!(h.p50(), 32); // bucket lower bound for 42 is 2^5 = 32
        assert_eq!(h.p99(), 32);
        assert_eq!(h.mean(), 42.0);
    }

    #[test]
    fn histogram_merge() {
        let mut a = Histogram::new();
        let mut b = Histogram::new();
        a.record(100);
        b.record(200);
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert_eq!(a.min_val(), 100);
        assert_eq!(a.max_val(), 200);
    }

    #[test]
    fn histogram_reset() {
        let mut h = Histogram::new();
        h.record(100);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.p50(), 0);
        assert_eq!(h.min_val(), 0);
    }

    #[test]
    fn histogram_empty() {
        let h = Histogram::new();
        assert_eq!(h.p50(), 0);
        assert_eq!(h.p99(), 0);
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.count(), 0);
        assert_eq!(h.min_val(), 0);
    }

    #[test]
    fn perf_context_default() {
        let ctx = SearchPerfContext::default();
        assert_eq!(ctx.blocks_read, 0);
        assert_eq!(ctx.total_ns, 0);
    }

    #[test]
    fn perf_context_display() {
        let ctx = SearchPerfContext {
            blocks_read: 50,
            blocks_hit: 30,
            blocks_miss: 18,
            phys_reads: 20,
            singleflight_waits: 2,
            expansions: 50,
            distance_computes: 500,
            refine_count: 0,
            useful_expansions: 40,
            wasted_expansions: 10,
            best_result_at_expansion: 5,
            first_topk_at_expansion: 1,
            prefetch_issued: 0,
            prefetch_consumed: 0,
            inflight_sum: 0,
            inflight_samples: 0,
            inflight_max: 0,
            global_inflight_sum: 0,
            global_inflight_samples: 0,
            global_inflight_max: 0,
            stopped_early: false,
            consecutive_stalls_at_end: 0,
            expansions_at_stop: 0,
            pq_candidates_scored: 0,
            pq_candidates_passed: 0,
            pq_candidates_filtered: 0,
            io_wait_ns: 200_000,
            compute_ns: 300_000,
            dist_ns: 250_000,
            refine_ns: 0,
            refine_bytes: 0,
            total_ns: 550_000,
            page_sched_hits: 0,
        };
        let s = format!("{}", ctx);
        assert!(s.contains("blocks=50"));
        assert!(s.contains("hit_rate=60.0%"));
        assert!(s.contains("dist="));
        assert!(s.contains("overhead="));
    }

    #[test]
    fn search_guard_records_on_drop() {
        let recorder = QueryRecorder::new();
        {
            let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
            guard.ctx.blocks_read = 42;
            guard.ctx.distance_computes = 100;
        } // drop records
        assert_eq!(recorder.query_count(), 1);
    }

    #[test]
    fn search_guard_timing() {
        let recorder = QueryRecorder::new();
        {
            let _guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(recorder.query_count(), 1);
        // report should show non-zero latency
        let report = recorder.report();
        assert!(report.contains("Queries: 1"));
    }

    #[test]
    fn search_guard_snapshot_before_drop() {
        let recorder = QueryRecorder::new();
        let guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let snap = guard.snapshot();
        assert!(snap.total_ns >= 1_000_000, "total_ns={}", snap.total_ns);
        drop(guard);
    }

    #[test]
    fn query_recorder_reset() {
        let recorder = QueryRecorder::new();
        {
            let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
            guard.ctx.blocks_read = 42;
        }
        assert_eq!(recorder.query_count(), 1);
        recorder.reset();
        assert_eq!(recorder.query_count(), 0);
    }

    #[test]
    fn query_recorder_report() {
        let recorder = QueryRecorder::new();
        for i in 0..10 {
            let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
            guard.ctx.blocks_read = 50 + i;
            guard.ctx.distance_computes = 500 + i * 10;
        }
        let report = recorder.report();
        assert!(report.contains("Queries: 10"));
        assert!(report.contains("Blocks/query:"));
        assert!(report.contains("Dists/query:"));
    }

    #[test]
    fn query_recorder_inflight_histograms() {
        let recorder = QueryRecorder::new();
        {
            let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
            // Simulate local inflight: avg = 120/40 = 3
            guard.ctx.inflight_sum = 120;
            guard.ctx.inflight_samples = 40;
            guard.ctx.inflight_max = 5;
            // Simulate global inflight: avg = 400/40 = 10
            guard.ctx.global_inflight_sum = 400;
            guard.ctx.global_inflight_samples = 40;
            guard.ctx.global_inflight_max = 16;
        }
        assert_eq!(recorder.query_count(), 1);
        assert!(recorder.global_inflight_mean() > 0.0);
        let report = recorder.report();
        assert!(report.contains("Inflight local:"), "report missing local inflight: {}", report);
        assert!(report.contains("Inflight global:"), "report missing global inflight: {}", report);
    }

    #[test]
    fn query_recorder_inflight_no_samples() {
        let recorder = QueryRecorder::new();
        {
            let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
            guard.ctx.blocks_read = 10;
            // No inflight samples → histograms should stay empty
        }
        assert_eq!(recorder.query_count(), 1);
        assert_eq!(recorder.global_inflight_mean(), 0.0);
        let report = recorder.report();
        assert!(!report.contains("Inflight"), "should not print inflight with no samples");
    }

    #[test]
    fn perf_level_ordering() {
        assert!(PerfLevel::CountOnly < PerfLevel::EnableTime);
    }
}
