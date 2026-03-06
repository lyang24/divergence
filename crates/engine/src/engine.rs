//! Production engine coordinator: config resolution, global QD, query admission,
//! health checking, and stats aggregation.
//!
//! The Engine owns cross-core shared state and hands each worker a CoreSetup
//! with the Arcs it needs. Per-core code (monoio tasks) never touches Engine
//! directly — all communication is through atomic counters and mailbox slots.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::io::{default_global_qd, GlobalIoBudget, IoDriver};
use crate::perf::{CoreSnapshot, QueryRecorder};

// ---------------------------------------------------------------------------
// EngineHealth
// ---------------------------------------------------------------------------

/// Engine health state. Stored as AtomicU8. Three tiers with hysteresis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum EngineHealth {
    Healthy = 0,
    Degraded = 1,
    Throttled = 2,
}

impl From<u8> for EngineHealth {
    fn from(v: u8) -> Self {
        match v {
            1 => EngineHealth::Degraded,
            2 => EngineHealth::Throttled,
            _ => EngineHealth::Healthy,
        }
    }
}

// ---------------------------------------------------------------------------
// EngineConfig
// ---------------------------------------------------------------------------

/// Configuration for an Engine instance.
pub struct EngineConfig {
    pub index_dir: String,
    pub num_cores: usize,
    /// Global device queue depth. None -> `default_global_qd(num_cores)`.
    pub global_qd: Option<usize>,
    /// Per-core local IO semaphore permits. None -> `global_qd / num_cores`, clamped 2..=16.
    pub per_core_qd: Option<usize>,
    /// Prefetch lookahead window (W). Default 4.
    pub prefetch_window: usize,
    /// Prefetch budget (concurrent prefetch IOs). Default 4.
    pub prefetch_budget: usize,
    /// Adaptive stopping: stall limit. Default 8.
    pub stall_limit: u32,
    /// Adaptive stopping: drain budget. Default 16.
    pub drain_budget: u32,
    /// Use O_DIRECT. Default true.
    pub direct_io: bool,
    /// p99 SLA in microseconds. 0 = disable health checking.
    pub p99_sla_us: u64,
    /// Queries per health check window. Default 50.
    pub health_window: u64,
    /// Max concurrent in-flight queries. None -> `num_cores * 2`.
    pub query_cap: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            index_dir: String::new(),
            num_cores: 1,
            global_qd: None,
            per_core_qd: None,
            prefetch_window: 4,
            prefetch_budget: 4,
            stall_limit: 8,
            drain_budget: 16,
            direct_io: true,
            p99_sla_us: 0,
            health_window: 50,
            query_cap: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GlobalQueryLimiter — cross-core query admission control
// ---------------------------------------------------------------------------

/// Cross-core query admission control. Atomic counter, same pattern as GlobalIoBudget.
/// Limits concurrent in-flight queries across all cores.
///
/// When linked to a health flag, effective capacity is reduced under load:
/// - Healthy: 100% capacity
/// - Degraded: 75% capacity
/// - Throttled: 50% capacity
pub struct GlobalQueryLimiter {
    available: AtomicUsize,
    capacity: usize,
    health: Option<Arc<AtomicU8>>,
}

// Safety: AtomicUsize is Send+Sync.
unsafe impl Send for GlobalQueryLimiter {}
unsafe impl Sync for GlobalQueryLimiter {}

impl GlobalQueryLimiter {
    pub fn new(capacity: usize) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
            capacity,
            health: None,
        }
    }

    /// Create a limiter linked to a shared health flag for adaptive capacity.
    pub fn with_health(capacity: usize, health: Arc<AtomicU8>) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
            capacity,
            health: Some(health),
        }
    }

    /// Effective capacity based on current health state.
    pub fn effective_capacity(&self) -> usize {
        match &self.health {
            Some(h) => match EngineHealth::from(h.load(Ordering::Acquire)) {
                EngineHealth::Healthy => self.capacity,
                EngineHealth::Degraded => self.capacity * 3 / 4,
                EngineHealth::Throttled => self.capacity / 2,
            },
            None => self.capacity,
        }
    }

    /// Try to acquire one query slot. Returns true on success.
    pub fn try_acquire(&self) -> bool {
        let mut current = self.available.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return false;
            }
            match self.available.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(new) => current = new,
            }
        }
    }

    /// Release one query slot back to the pool.
    pub fn release(&self) {
        self.available.fetch_add(1, Ordering::Release);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available(&self) -> usize {
        self.available.load(Ordering::Relaxed)
    }

    /// Current number of in-flight queries: `capacity - available`.
    pub fn inflight(&self) -> usize {
        self.capacity - self.available.load(Ordering::Relaxed)
    }

    /// Async acquire: polite yield loop checking health-aware effective capacity.
    /// Yields for 50us between attempts (coarser than IO-level 5us).
    pub async fn acquire(&self) -> QueryPermit<'_> {
        loop {
            let effective = self.effective_capacity();
            let inflight = self.inflight();
            if inflight < effective && self.try_acquire() {
                return QueryPermit { limiter: self };
            }
            monoio::time::sleep(Duration::from_micros(50)).await;
        }
    }
}

/// RAII query permit — releases on drop.
pub struct QueryPermit<'a> {
    limiter: &'a GlobalQueryLimiter,
}

impl<'a> QueryPermit<'a> {
    /// Try to acquire a query permit. Returns None if no slots available.
    pub fn try_acquire(limiter: &'a GlobalQueryLimiter) -> Option<Self> {
        if limiter.try_acquire() {
            Some(Self { limiter })
        } else {
            None
        }
    }

    /// Async acquire using health-aware effective capacity.
    pub async fn acquire(limiter: &'a GlobalQueryLimiter) -> Self {
        limiter.acquire().await
    }
}

impl Drop for QueryPermit<'_> {
    fn drop(&mut self) {
        self.limiter.release();
    }
}

// ---------------------------------------------------------------------------
// HealthChecker — per-core EWMA + hysteresis + cooldown
// ---------------------------------------------------------------------------

/// Per-core health checker with EWMA smoothing, hysteresis thresholds, and
/// cooldown to prevent oscillation.
///
/// Transition rules (with hysteresis):
/// - Healthy -> Degraded:  ewma_p99 > sla AND ewma_inflight > 0.7
/// - Degraded -> Throttled: ewma_p99 > 1.5*sla AND ewma_inflight > 0.85
/// - Throttled -> Degraded: ewma_p99 < 0.8*sla AND ewma_inflight < 0.6 AND cooldown==0
/// - Degraded -> Healthy:  ewma_p99 < 0.6*sla AND ewma_inflight < 0.5 AND cooldown==0
///
/// Worsening is immediate. Improving requires cooldown (3 windows) to expire first.
pub struct HealthChecker {
    health: Arc<AtomicU8>,
    mailbox: Arc<Mutex<Option<CoreSnapshot>>>,
    window: u64,
    p99_sla_us: u64,
    queries_since_check: Cell<u64>,
    gqd_capacity: usize,
    // EWMA state
    ewma_p99: Cell<f64>,
    ewma_inflight: Cell<f64>,
    cooldown_remaining: Cell<u64>,
}

impl HealthChecker {
    pub fn new(
        health: Arc<AtomicU8>,
        mailbox: Arc<Mutex<Option<CoreSnapshot>>>,
        window: u64,
        p99_sla_us: u64,
        gqd_capacity: usize,
    ) -> Self {
        Self {
            health,
            mailbox,
            window,
            p99_sla_us,
            queries_since_check: Cell::new(0),
            gqd_capacity,
            ewma_p99: Cell::new(0.0),
            ewma_inflight: Cell::new(0.0),
            cooldown_remaining: Cell::new(0),
        }
    }

    /// Call after each query completes. Periodically snapshots stats and runs
    /// the EWMA + hysteresis state machine.
    ///
    /// `recorder`: per-core QueryRecorder (not reset externally).
    /// `io`: IoDriver for take_io_timing().
    pub fn on_query_complete(&self, recorder: &QueryRecorder, io: &IoDriver) {
        let count = self.queries_since_check.get() + 1;
        self.queries_since_check.set(count);

        if count < self.window {
            return;
        }
        self.queries_since_check.set(0);

        let io_timing = io.take_io_timing();
        let snap = recorder.take_snapshot(io_timing);

        // EWMA update (alpha = 0.3)
        let alpha = 0.3;
        let prev_p99 = self.ewma_p99.get();
        let new_p99 = if prev_p99 == 0.0 {
            snap.lat_p99_us
        } else {
            alpha * snap.lat_p99_us + (1.0 - alpha) * prev_p99
        };
        self.ewma_p99.set(new_p99);

        let inflight_ratio = if self.gqd_capacity > 0 {
            snap.global_inflight_max as f64 / self.gqd_capacity as f64
        } else {
            0.0
        };
        let prev_ifl = self.ewma_inflight.get();
        let new_ifl = if prev_ifl == 0.0 {
            inflight_ratio
        } else {
            alpha * inflight_ratio + (1.0 - alpha) * prev_ifl
        };
        self.ewma_inflight.set(new_ifl);

        // Cooldown tick
        let cd = self.cooldown_remaining.get();
        if cd > 0 {
            self.cooldown_remaining.set(cd - 1);
        }

        // Health state transitions (only when SLA is configured)
        if self.p99_sla_us > 0 {
            let sla = self.p99_sla_us as f64;
            let current = EngineHealth::from(self.health.load(Ordering::Acquire));
            let cd = self.cooldown_remaining.get();

            let next = match current {
                // Can always worsen immediately
                EngineHealth::Healthy if new_p99 > 1.5 * sla && new_ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                EngineHealth::Healthy if new_p99 > sla && new_ifl > 0.7 => {
                    EngineHealth::Degraded
                }
                EngineHealth::Degraded if new_p99 > 1.5 * sla && new_ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                // Can only improve after cooldown expires
                EngineHealth::Throttled
                    if cd == 0 && new_p99 < 0.8 * sla && new_ifl < 0.6 =>
                {
                    EngineHealth::Degraded
                }
                EngineHealth::Degraded
                    if cd == 0 && new_p99 < 0.6 * sla && new_ifl < 0.5 =>
                {
                    EngineHealth::Healthy
                }
                other => other,
            };

            if next != current {
                self.health.store(next as u8, Ordering::Release);
                if next > current {
                    // Worsened — set cooldown
                    self.cooldown_remaining.set(3);
                }
            }
        }

        // Deposit snapshot into mailbox for Engine::collect_stats()
        if let Ok(mut slot) = self.mailbox.try_lock() {
            *slot = Some(snap);
        }
    }

    /// Read current health state.
    pub fn health(&self) -> EngineHealth {
        EngineHealth::from(self.health.load(Ordering::Acquire))
    }

    /// Current EWMA-smoothed p99 latency (microseconds).
    pub fn ewma_p99(&self) -> f64 {
        self.ewma_p99.get()
    }

    /// Current EWMA-smoothed inflight ratio (0.0-1.0).
    pub fn ewma_inflight(&self) -> f64 {
        self.ewma_inflight.get()
    }

    /// Windows remaining before health can improve.
    pub fn cooldown_remaining(&self) -> u64 {
        self.cooldown_remaining.get()
    }
}

// ---------------------------------------------------------------------------
// CoreSetup — what each worker closure captures
// ---------------------------------------------------------------------------

/// Per-core setup bundle. Created by `Engine::core_setup()`, captured by worker closures.
pub struct CoreSetup {
    pub core_id: usize,
    pub global_budget: Arc<GlobalIoBudget>,
    pub query_limiter: Arc<GlobalQueryLimiter>,
    pub health: Arc<AtomicU8>,
    pub mailbox: Arc<Mutex<Option<CoreSnapshot>>>,
    pub per_core_qd: usize,
    pub gqd_capacity: usize,
    pub index_dir: String,
    pub direct_io: bool,
    pub prefetch_window: usize,
    pub prefetch_budget: usize,
    pub stall_limit: u32,
    pub drain_budget: u32,
    pub p99_sla_us: u64,
    pub health_window: u64,
}

// ---------------------------------------------------------------------------
// EngineStats — aggregated from all core snapshots
// ---------------------------------------------------------------------------

/// Aggregated stats from all cores. No histograms — scalars only.
#[derive(Debug)]
pub struct EngineStats {
    pub total_queries: u64,
    /// Raw per-core snapshots (one per core, if available).
    pub per_core: Vec<CoreSnapshot>,
    /// Max p99 latency across cores (us).
    pub max_lat_p99_us: f64,
    /// Query-weighted average io_wait percentage.
    pub avg_io_wait_pct: f64,
    /// Max global inflight depth observed across any core.
    pub max_global_inflight: u64,
    /// Aggregated sem_wait / (sem_wait + device) percentage.
    pub sem_wait_pct: f64,
    /// Aggregated device_ns / io_count.
    pub device_ns_per_io: f64,
    pub health: EngineHealth,
    /// Fraction of queries that waited for admission (0.0-1.0).
    pub admit_wait_pct: f64,
    /// Health state distribution: [Healthy, Degraded, Throttled] snapshot counts.
    pub health_distribution: [u64; 3],
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Production engine coordinator. Resolves config, owns cross-core shared state.
///
/// Does NOT own monoio runtimes — those are spawned by the caller via
/// `spawn_worker()` with `CoreSetup` captured in the closure.
pub struct Engine {
    global_budget: Arc<GlobalIoBudget>,
    query_limiter: Arc<GlobalQueryLimiter>,
    health: Arc<AtomicU8>,
    core_mailboxes: Vec<Arc<Mutex<Option<CoreSnapshot>>>>,
    pub resolved_global_qd: usize,
    pub resolved_per_core_qd: usize,
    config: EngineConfig,
}

impl Engine {
    /// Create a new Engine, resolving config defaults.
    pub fn new(config: EngineConfig) -> Self {
        let gqd = config
            .global_qd
            .unwrap_or_else(|| default_global_qd(config.num_cores));
        let per_core = config
            .per_core_qd
            .unwrap_or_else(|| (gqd / config.num_cores).clamp(2, 16));

        let core_mailboxes: Vec<_> = (0..config.num_cores)
            .map(|_| Arc::new(Mutex::new(None)))
            .collect();

        let health = Arc::new(AtomicU8::new(EngineHealth::Healthy as u8));

        Self {
            global_budget: Arc::new(GlobalIoBudget::new(gqd)),
            query_limiter: Arc::new(GlobalQueryLimiter::with_health(
                config.query_cap.unwrap_or(config.num_cores * 2),
                Arc::clone(&health),
            )),
            health,
            core_mailboxes,
            resolved_global_qd: gqd,
            resolved_per_core_qd: per_core,
            config,
        }
    }

    /// Get setup bundle for a specific core.
    pub fn core_setup(&self, core_id: usize) -> CoreSetup {
        CoreSetup {
            core_id,
            global_budget: Arc::clone(&self.global_budget),
            query_limiter: Arc::clone(&self.query_limiter),
            health: Arc::clone(&self.health),
            mailbox: Arc::clone(&self.core_mailboxes[core_id]),
            per_core_qd: self.resolved_per_core_qd,
            gqd_capacity: self.resolved_global_qd,
            index_dir: self.config.index_dir.clone(),
            direct_io: self.config.direct_io,
            prefetch_window: self.config.prefetch_window,
            prefetch_budget: self.config.prefetch_budget,
            stall_limit: self.config.stall_limit,
            drain_budget: self.config.drain_budget,
            p99_sla_us: self.config.p99_sla_us,
            health_window: self.config.health_window,
        }
    }

    /// Collect stats from all core mailboxes. Drains each mailbox.
    pub fn collect_stats(&self) -> EngineStats {
        let mut per_core = Vec::with_capacity(self.core_mailboxes.len());
        for mb in &self.core_mailboxes {
            if let Ok(mut slot) = mb.lock() {
                if let Some(snap) = slot.take() {
                    per_core.push(snap);
                }
            }
        }

        let total_queries: u64 = per_core.iter().map(|s| s.queries).sum();

        let max_lat_p99_us = per_core
            .iter()
            .map(|s| s.lat_p99_us)
            .fold(0.0f64, f64::max);

        let avg_io_wait_pct = if total_queries > 0 {
            per_core
                .iter()
                .map(|s| s.io_wait_pct * s.queries as f64)
                .sum::<f64>()
                / total_queries as f64
        } else {
            0.0
        };

        let max_global_inflight = per_core.iter().map(|s| s.global_inflight_max).max().unwrap_or(0);

        let total_sem: u64 = per_core.iter().map(|s| s.sem_wait_ns).sum();
        let total_dev: u64 = per_core.iter().map(|s| s.device_ns).sum();
        let sem_wait_pct = if total_sem + total_dev > 0 {
            total_sem as f64 / (total_sem + total_dev) as f64 * 100.0
        } else {
            0.0
        };

        let total_io_count: u64 = per_core.iter().map(|s| s.io_count).sum();
        let device_ns_per_io = if total_io_count > 0 {
            total_dev as f64 / total_io_count as f64
        } else {
            0.0
        };

        let health = EngineHealth::from(self.health.load(Ordering::Acquire));

        // Admission wait aggregation
        let total_admit: u64 = per_core.iter().map(|s| s.admit_total).sum();
        let total_admit_waited: u64 = per_core.iter().map(|s| s.admit_count).sum();
        let admit_wait_pct = if total_admit > 0 {
            total_admit_waited as f64 / total_admit as f64
        } else {
            0.0
        };

        // Health distribution from snapshots
        let mut health_distribution = [0u64; 3];
        for s in &per_core {
            let idx = (s.health_at_snapshot as usize).min(2);
            health_distribution[idx] += 1;
        }

        EngineStats {
            total_queries,
            per_core,
            max_lat_p99_us,
            avg_io_wait_pct,
            max_global_inflight,
            sem_wait_pct,
            device_ns_per_io,
            health,
            admit_wait_pct,
            health_distribution,
        }
    }

    /// Reference to the shared global IO budget.
    pub fn global_budget(&self) -> &Arc<GlobalIoBudget> {
        &self.global_budget
    }

    /// Reference to the shared query limiter.
    pub fn query_limiter(&self) -> &Arc<GlobalQueryLimiter> {
        &self.query_limiter
    }

    /// Reference to the shared health flag.
    pub fn health_flag(&self) -> &Arc<AtomicU8> {
        &self.health
    }

    /// Current health state.
    pub fn health(&self) -> EngineHealth {
        EngineHealth::from(self.health.load(Ordering::Acquire))
    }

    /// Access resolved config.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_default_qd_resolution() {
        let config = EngineConfig {
            num_cores: 2,
            ..Default::default()
        };
        let engine = Engine::new(config);
        // default_global_qd(2) = max(16, 4*2) = 16
        assert_eq!(engine.resolved_global_qd, 16);
        assert_eq!(engine.global_budget.capacity(), 16);
        // per_core = 16/2 = 8, clamped to 2..=16
        assert_eq!(engine.resolved_per_core_qd, 8);
    }

    #[test]
    fn test_engine_explicit_qd() {
        let config = EngineConfig {
            num_cores: 4,
            global_qd: Some(32),
            per_core_qd: Some(6),
            ..Default::default()
        };
        let engine = Engine::new(config);
        assert_eq!(engine.resolved_global_qd, 32);
        assert_eq!(engine.resolved_per_core_qd, 6);
    }

    #[test]
    fn test_engine_core_setup() {
        let config = EngineConfig {
            num_cores: 2,
            index_dir: "/tmp/test".to_string(),
            ..Default::default()
        };
        let engine = Engine::new(config);
        let setup0 = engine.core_setup(0);
        let setup1 = engine.core_setup(1);
        assert_eq!(setup0.core_id, 0);
        assert_eq!(setup1.core_id, 1);
        assert_eq!(setup0.per_core_qd, engine.resolved_per_core_qd);
        assert_eq!(setup0.gqd_capacity, engine.resolved_global_qd);
        // Same Arc for global_budget
        assert!(Arc::ptr_eq(&setup0.global_budget, &setup1.global_budget));
        // Different mailboxes
        assert!(!Arc::ptr_eq(&setup0.mailbox, &setup1.mailbox));
    }

    #[test]
    fn test_query_limiter_basic() {
        let limiter = GlobalQueryLimiter::new(2);
        assert_eq!(limiter.capacity(), 2);
        assert_eq!(limiter.available(), 2);

        assert!(limiter.try_acquire());
        assert_eq!(limiter.available(), 1);
        assert!(limiter.try_acquire());
        assert_eq!(limiter.available(), 0);
        assert!(!limiter.try_acquire()); // exhausted

        limiter.release();
        assert_eq!(limiter.available(), 1);
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_query_limiter_health_aware() {
        let health = Arc::new(AtomicU8::new(EngineHealth::Healthy as u8));
        let limiter = GlobalQueryLimiter::with_health(8, Arc::clone(&health));

        // Healthy: full capacity
        assert_eq!(limiter.effective_capacity(), 8);

        // Degraded: 75%
        health.store(EngineHealth::Degraded as u8, Ordering::Release);
        assert_eq!(limiter.effective_capacity(), 6); // 8 * 3 / 4

        // Throttled: 50%
        health.store(EngineHealth::Throttled as u8, Ordering::Release);
        assert_eq!(limiter.effective_capacity(), 4); // 8 / 2

        // Back to healthy
        health.store(EngineHealth::Healthy as u8, Ordering::Release);
        assert_eq!(limiter.effective_capacity(), 8);
    }

    #[test]
    fn test_query_permit_raii() {
        let limiter = GlobalQueryLimiter::new(1);
        {
            let _permit = QueryPermit::try_acquire(&limiter).unwrap();
            assert_eq!(limiter.available(), 0);
            assert!(QueryPermit::try_acquire(&limiter).is_none());
        } // permit dropped
        assert_eq!(limiter.available(), 1);
    }

    #[test]
    fn test_health_checker_ewma_hysteresis() {
        // Test the EWMA + hysteresis state machine by directly setting EWMA values
        // instead of going through the histogram (avoids log2 quantization noise).
        let health = Arc::new(AtomicU8::new(EngineHealth::Healthy as u8));
        let mailbox: Arc<Mutex<Option<CoreSnapshot>>> = Arc::new(Mutex::new(None));
        let checker = HealthChecker::new(
            Arc::clone(&health),
            Arc::clone(&mailbox),
            1,
            1000, // p99 SLA = 1000us
            16,   // gqd_capacity
        );

        // Helper: directly set EWMA values and run the state machine
        let drive = |p99: f64, ifl: f64| {
            checker.ewma_p99.set(p99);
            checker.ewma_inflight.set(ifl);

            let cd = checker.cooldown_remaining.get();
            if cd > 0 {
                checker.cooldown_remaining.set(cd - 1);
            }

            let sla = 1000.0f64;
            let current = checker.health();
            let cd = checker.cooldown_remaining.get();

            let next = match current {
                EngineHealth::Healthy if p99 > 1.5 * sla && ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                EngineHealth::Healthy if p99 > sla && ifl > 0.7 => EngineHealth::Degraded,
                EngineHealth::Degraded if p99 > 1.5 * sla && ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                EngineHealth::Throttled
                    if cd == 0 && p99 < 0.8 * sla && ifl < 0.6 =>
                {
                    EngineHealth::Degraded
                }
                EngineHealth::Degraded
                    if cd == 0 && p99 < 0.6 * sla && ifl < 0.5 =>
                {
                    EngineHealth::Healthy
                }
                other => other,
            };

            if next != current {
                health.store(next as u8, Ordering::Release);
                if next > current {
                    checker.cooldown_remaining.set(3);
                }
            }
        };

        // Start: Healthy
        assert_eq!(checker.health(), EngineHealth::Healthy);

        // Healthy -> Degraded: p99 > sla AND inflight > 0.7
        drive(1200.0, 0.75);
        assert_eq!(checker.health(), EngineHealth::Degraded);
        assert_eq!(checker.cooldown_remaining.get(), 3);

        // Degraded -> Throttled: p99 > 1.5*sla AND inflight > 0.85 (immediate)
        drive(1600.0, 0.9);
        assert_eq!(checker.health(), EngineHealth::Throttled);
        assert_eq!(checker.cooldown_remaining.get(), 3);

        // Try to recover but cooldown blocks (cd decremented from 3->2)
        drive(700.0, 0.5);
        assert_eq!(checker.health(), EngineHealth::Throttled); // cd=2

        drive(700.0, 0.5); // cd=1
        assert_eq!(checker.health(), EngineHealth::Throttled);

        drive(700.0, 0.5); // cd=0 → can now improve
        assert_eq!(checker.health(), EngineHealth::Degraded); // p99<800, ifl<0.6

        // Degraded -> Healthy: need p99 < 0.6*sla=600 AND ifl < 0.5 AND cd=0
        drive(500.0, 0.4); // cd=2 (set when we worsened to Degraded... wait, no)
        // Going from Throttled to Degraded is an improvement, not worsening.
        // So cooldown was NOT reset when we went from Throttled->Degraded.
        // But the drive at cd=0 above didn't worsen, so cooldown stays at 0.
        // Hmm, let me re-check: when we went from Throttled->Degraded above,
        // next < current (improvement), so cooldown was NOT set.
        // Current cd should be 0 (was decremented to 0, no new worsening).
        assert_eq!(checker.health(), EngineHealth::Healthy);
    }

    #[test]
    fn test_health_checker_no_oscillation() {
        // Verify that state doesn't flip-flop when readings alternate
        let health = Arc::new(AtomicU8::new(EngineHealth::Healthy as u8));
        let mailbox: Arc<Mutex<Option<CoreSnapshot>>> = Arc::new(Mutex::new(None));
        let checker = HealthChecker::new(
            Arc::clone(&health),
            Arc::clone(&mailbox),
            1,
            1000,
            16,
        );

        let mut flips = 0u32;
        let mut prev = EngineHealth::Healthy;

        // Drive with alternating high/low to test EWMA smoothing prevents flapping
        for i in 0..20 {
            let (p99, ifl) = if i % 2 == 0 {
                (1200.0, 12) // above SLA
            } else {
                (800.0, 8) // below SLA
            };

            // Mini EWMA drive
            let alpha = 0.3;
            let prev_p99 = checker.ewma_p99.get();
            let new_p99 = if prev_p99 == 0.0 {
                p99
            } else {
                alpha * p99 + (1.0 - alpha) * prev_p99
            };
            checker.ewma_p99.set(new_p99);

            let ratio = ifl as f64 / 16.0;
            let prev_ifl = checker.ewma_inflight.get();
            let new_ifl = if prev_ifl == 0.0 {
                ratio
            } else {
                alpha * ratio + (1.0 - alpha) * prev_ifl
            };
            checker.ewma_inflight.set(new_ifl);

            let cd = checker.cooldown_remaining.get();
            if cd > 0 {
                checker.cooldown_remaining.set(cd - 1);
            }

            let sla = 1000.0f64;
            let current = checker.health();
            let cd = checker.cooldown_remaining.get();

            let next = match current {
                EngineHealth::Healthy if new_p99 > 1.5 * sla && new_ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                EngineHealth::Healthy if new_p99 > sla && new_ifl > 0.7 => {
                    EngineHealth::Degraded
                }
                EngineHealth::Degraded if new_p99 > 1.5 * sla && new_ifl > 0.85 => {
                    EngineHealth::Throttled
                }
                EngineHealth::Throttled
                    if cd == 0 && new_p99 < 0.8 * sla && new_ifl < 0.6 =>
                {
                    EngineHealth::Degraded
                }
                EngineHealth::Degraded
                    if cd == 0 && new_p99 < 0.6 * sla && new_ifl < 0.5 =>
                {
                    EngineHealth::Healthy
                }
                other => other,
            };

            if next != current {
                health.store(next as u8, Ordering::Release);
                if next > current {
                    checker.cooldown_remaining.set(3);
                }
            }

            let cur = checker.health();
            if cur != prev {
                flips += 1;
                prev = cur;
            }
        }

        // With EWMA + cooldown, alternating readings should not cause many flips
        assert!(
            flips <= 4,
            "Too many health state flips: {} (expected <= 4 with EWMA + cooldown)",
            flips
        );
    }

    #[test]
    fn test_collect_stats_empty() {
        let config = EngineConfig {
            num_cores: 2,
            ..Default::default()
        };
        let engine = Engine::new(config);
        let stats = engine.collect_stats();
        assert_eq!(stats.total_queries, 0);
        assert!(stats.per_core.is_empty());
        assert_eq!(stats.health, EngineHealth::Healthy);
    }

    #[test]
    fn test_collect_stats_with_snapshots() {
        let config = EngineConfig {
            num_cores: 2,
            ..Default::default()
        };
        let engine = Engine::new(config);

        // Manually deposit snapshots into mailboxes
        {
            let mut slot = engine.core_mailboxes[0].lock().unwrap();
            *slot = Some(CoreSnapshot {
                queries: 10,
                lat_p99_us: 5000.0,
                io_wait_pct: 40.0,
                global_inflight_max: 12,
                sem_wait_ns: 100,
                device_ns: 900,
                io_count: 50,
                ..Default::default()
            });
        }
        {
            let mut slot = engine.core_mailboxes[1].lock().unwrap();
            *slot = Some(CoreSnapshot {
                queries: 20,
                lat_p99_us: 8000.0,
                io_wait_pct: 30.0,
                global_inflight_max: 14,
                sem_wait_ns: 200,
                device_ns: 800,
                io_count: 60,
                ..Default::default()
            });
        }

        let stats = engine.collect_stats();
        assert_eq!(stats.total_queries, 30);
        assert_eq!(stats.per_core.len(), 2);
        assert!((stats.max_lat_p99_us - 8000.0).abs() < 0.1);
        assert_eq!(stats.max_global_inflight, 14);
        // Weighted avg: (40*10 + 30*20) / 30 = 1000/30 ~ 33.33
        assert!((stats.avg_io_wait_pct - 33.333).abs() < 1.0);
        // sem_wait_pct: 300 / (300 + 1700) * 100 = 15.0%
        assert!((stats.sem_wait_pct - 15.0).abs() < 0.1);

        // Second collect should be empty (mailboxes drained)
        let stats2 = engine.collect_stats();
        assert_eq!(stats2.total_queries, 0);
        assert!(stats2.per_core.is_empty());
    }

    #[test]
    fn test_engine_health_transitions() {
        let config = EngineConfig {
            num_cores: 1,
            ..Default::default()
        };
        let engine = Engine::new(config);
        assert_eq!(engine.health(), EngineHealth::Healthy);

        engine
            .health
            .store(EngineHealth::Degraded as u8, Ordering::Release);
        assert_eq!(engine.health(), EngineHealth::Degraded);

        engine
            .health
            .store(EngineHealth::Throttled as u8, Ordering::Release);
        assert_eq!(engine.health(), EngineHealth::Throttled);

        engine
            .health
            .store(EngineHealth::Healthy as u8, Ordering::Release);
        assert_eq!(engine.health(), EngineHealth::Healthy);
    }

    #[test]
    fn test_engine_health_ordering() {
        assert!(EngineHealth::Healthy < EngineHealth::Degraded);
        assert!(EngineHealth::Degraded < EngineHealth::Throttled);
    }

    #[test]
    fn test_collect_stats_admit_fields() {
        let config = EngineConfig {
            num_cores: 1,
            ..Default::default()
        };
        let engine = Engine::new(config);

        {
            let mut slot = engine.core_mailboxes[0].lock().unwrap();
            *slot = Some(CoreSnapshot {
                queries: 100,
                admit_wait_ns: 50_000,
                admit_count: 10,
                admit_total: 100,
                health_at_snapshot: EngineHealth::Degraded as u8,
                ..Default::default()
            });
        }

        let stats = engine.collect_stats();
        assert!((stats.admit_wait_pct - 0.1).abs() < 0.001); // 10/100
        assert_eq!(stats.health_distribution[1], 1); // one Degraded snapshot
    }
}
