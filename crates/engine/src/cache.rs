//! AdjacencyPool: per-core set-associative block cache for adjacency reads.
//!
//! Design contracts:
//! 1. No hot-path allocation — all slots pre-allocated, waiters use fixed-capacity inline array
//! 2. Per-core — RefCell/Cell only, no atomics, no Mutex
//! 3. State machine — EMPTY → LOADING → READY with load_gen epoch
//! 4. In-flight dedup — LOADING entry: waiters use WaitForReady(slot_idx, vid, gen)
//! 5. Cancel safety — LoadGuard rolls back LOADING → EMPTY on drop
//! 6. CacheGuard must not escape expansion scope — decode neighbors then drop

use std::alloc::{self, Layout};
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::ptr::NonNull;
use std::task::{Context, Poll, Waker};

use divergence_storage::BLOCK_SIZE;

use crate::io::IoDriver;

const SET_WAYS: u32 = 8;
const MAX_PINNED_PER_SET: u32 = 4; // hub-pinned entries per set (leaves ≥4 ways for eviction)
const MAX_PREFETCH_QUEUE: usize = 16;

/// Max concurrent waiters per LOADING entry.
/// Bounded by coroutines-per-core: beam search (1) + prefetch window (~4-8).
/// Set to 8 (matching SET_WAYS) to cover worst case. Each Option<Waker> is
/// 16 bytes, so 128 bytes per entry × 128 entries = 16KB overhead.
const MAX_WAITERS_PER_ENTRY: usize = 8;

// ---------------------------------------------------------------------------
// WaiterArray — fixed-capacity, zero-allocation waker storage
// ---------------------------------------------------------------------------

/// Fixed-capacity array of wakers. No heap allocation ever.
///
/// Invariant: `wakers[0..len]` are all `Some`, `wakers[len..]` are all `None`.
struct WaiterArray {
    wakers: [Option<Waker>; MAX_WAITERS_PER_ENTRY],
    len: u8,
}

impl WaiterArray {
    fn new() -> Self {
        Self {
            wakers: Default::default(),
            len: 0,
        }
    }

    /// Register a waker. Returns true if registered, false if full.
    /// Caller should handle the full case (fall through to re-probe).
    fn push(&mut self, waker: Waker) -> bool {
        if (self.len as usize) < MAX_WAITERS_PER_ENTRY {
            self.wakers[self.len as usize] = Some(waker);
            self.len += 1;
            true
        } else {
            false
        }
    }

    /// Wake all registered wakers and clear the array.
    fn wake_all(&mut self) {
        for slot in self.wakers[..self.len as usize].iter_mut() {
            if let Some(w) = slot.take() {
                w.wake();
            }
        }
        self.len = 0;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len as usize
    }
}

// ---------------------------------------------------------------------------
// PrefetchChannel — SPSC ring buffer for prefetch hints
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct PrefetchEntry {
    vid: u32,
    slot_idx: u32,
    load_gen: u32,
}

struct PrefetchChannel {
    buf: [Option<PrefetchEntry>; MAX_PREFETCH_QUEUE],
    head: usize,
    tail: usize,
    len: usize,
    waker: Option<Waker>,
    stopped: bool,
    paused: bool, // when true, prefetch_hint() silently drops new hints
}

impl PrefetchChannel {
    fn new() -> Self {
        Self {
            buf: [None; MAX_PREFETCH_QUEUE],
            head: 0,
            tail: 0,
            len: 0,
            waker: None,
            stopped: false,
            paused: false,
        }
    }

    fn is_full(&self) -> bool {
        self.len >= MAX_PREFETCH_QUEUE
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a prefetch entry. Auto-wakes the worker when queue transitions empty→non-empty.
    fn push(&mut self, entry: PrefetchEntry) -> bool {
        if self.len >= MAX_PREFETCH_QUEUE {
            return false;
        }
        let was_empty = self.len == 0;
        self.buf[self.tail] = Some(entry);
        self.tail = (self.tail + 1) % MAX_PREFETCH_QUEUE;
        self.len += 1;
        if was_empty {
            if let Some(w) = self.waker.as_ref() {
                w.wake_by_ref();
            }
        }
        true
    }

    fn pop(&mut self) -> Option<PrefetchEntry> {
        if self.len == 0 {
            return None;
        }
        let entry = self.buf[self.head].take();
        self.head = (self.head + 1) % MAX_PREFETCH_QUEUE;
        self.len -= 1;
        entry
    }

    /// Discard all pending entries.
    fn drain(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Fibonacci hashing multiplier for u32 keys.
const FIB_HASH: u64 = 11400714819323198485; // 2^64 / φ

// ---------------------------------------------------------------------------
// SlotState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Empty,
    Loading,
    Ready,
}

// ---------------------------------------------------------------------------
// SlotPtr — typed wrapper for slot pointers (only SlotStore can create)
// ---------------------------------------------------------------------------

/// A 4KB-aligned pointer to a cache slot buffer. Only created by SlotStore.
/// Prevents raw pointer misuse in IoDriver API.
pub(crate) struct SlotPtr(NonNull<u8>);

impl SlotPtr {
    pub(crate) fn as_mut_ptr(&self) -> *mut u8 {
        self.0.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// Entry — metadata per cache slot
// ---------------------------------------------------------------------------

struct Entry {
    vid: u32,          // u32::MAX = empty
    state: SlotState,
    load_gen: u32,     // incremented each LOADING transition
    referenced: bool,  // clock second-chance bit
    prefetched: bool,  // true if loaded via prefetch_hint (for consumed tracking)
    pin_count: u32,    // live CacheGuards
    pinned: bool,      // hub-pinned: eviction skips this entry
    waiters: WaiterArray,  // fixed-capacity, zero-alloc
}

impl Entry {
    fn empty() -> Self {
        Self {
            vid: u32::MAX,
            state: SlotState::Empty,
            load_gen: 0,
            referenced: false,
            prefetched: false,
            pin_count: 0,
            pinned: false,
            waiters: WaiterArray::new(),
        }
    }

    fn reset(&mut self) {
        self.vid = u32::MAX;
        self.state = SlotState::Empty;
        self.referenced = false;
        self.prefetched = false;
        self.pin_count = 0;
        self.pinned = false;
        // load_gen is NOT reset — monotonically increasing
        // waiters are NOT cleared here — find_or_evict caller handles it
    }
}

// ---------------------------------------------------------------------------
// PoolState — mutable interior behind RefCell
// ---------------------------------------------------------------------------

struct PoolState {
    entries: Vec<Entry>,
    prefetch: PrefetchChannel,
}

// ---------------------------------------------------------------------------
// SlotStore — one contiguous 4KB-aligned allocation
// ---------------------------------------------------------------------------

struct SlotStore {
    ptr: NonNull<u8>,
    capacity: u32, // number of slots
    layout: Layout,
}

impl SlotStore {
    fn new(num_slots: u32) -> Self {
        let total_bytes = num_slots as usize * BLOCK_SIZE;
        let layout =
            Layout::from_size_align(total_bytes.max(BLOCK_SIZE), BLOCK_SIZE).expect("bad layout");
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("SlotStore allocation failed");
        Self {
            ptr,
            capacity: num_slots,
            layout,
        }
    }

    /// Returns a typed SlotPtr to the start of slot `idx`.
    fn slot_ptr(&self, idx: u32) -> SlotPtr {
        debug_assert!(idx < self.capacity);
        let ptr = unsafe { self.ptr.as_ptr().add(idx as usize * BLOCK_SIZE) };
        SlotPtr(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Returns a slice view of slot `idx`.
    fn slot_slice(&self, idx: u32) -> &[u8] {
        debug_assert!(idx < self.capacity);
        unsafe { std::slice::from_raw_parts(self.slot_ptr(idx).as_mut_ptr(), BLOCK_SIZE) }
    }
}

impl Drop for SlotStore {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

// ---------------------------------------------------------------------------
// CacheStats — Cell<u64> counters (single-threaded, no atomics)
// ---------------------------------------------------------------------------

/// Snapshot of cache statistics.
#[derive(Debug, Clone, Copy)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub dedup_hits: u64,
    pub evictions: u64,
    pub evict_fail_all_pinned: u64,
    pub bypasses: u64,
    pub prefetch_hits: u64,
    /// Physical IO reads completed (miss loads + prefetch loads + bypass reads).
    /// This is the true NVMe IO count, unlike misses which undercounts
    /// because prefetch reads show up as later hits.
    pub phys_reads: u64,
}

struct CacheStats {
    hits: Cell<u64>,
    misses: Cell<u64>,
    dedup_hits: Cell<u64>,
    evictions: Cell<u64>,
    evict_fail_all_pinned: Cell<u64>,
    bypasses: Cell<u64>,
    prefetch_hits: Cell<u64>,
    phys_reads: Cell<u64>,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: Cell::new(0),
            misses: Cell::new(0),
            dedup_hits: Cell::new(0),
            evictions: Cell::new(0),
            evict_fail_all_pinned: Cell::new(0),
            bypasses: Cell::new(0),
            prefetch_hits: Cell::new(0),
            phys_reads: Cell::new(0),
        }
    }

    fn inc_hits(&self) {
        self.hits.set(self.hits.get() + 1);
    }
    fn inc_misses(&self) {
        self.misses.set(self.misses.get() + 1);
    }
    fn inc_dedup_hits(&self) {
        self.dedup_hits.set(self.dedup_hits.get() + 1);
    }
    fn inc_evictions(&self) {
        self.evictions.set(self.evictions.get() + 1);
    }
    fn inc_evict_fail(&self) {
        self.evict_fail_all_pinned
            .set(self.evict_fail_all_pinned.get() + 1);
    }
    fn inc_bypasses(&self) {
        self.bypasses.set(self.bypasses.get() + 1);
    }
    fn inc_prefetch_hits(&self) {
        self.prefetch_hits.set(self.prefetch_hits.get() + 1);
    }
    fn inc_phys_reads(&self) {
        self.phys_reads.set(self.phys_reads.get() + 1);
    }

    fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.get(),
            misses: self.misses.get(),
            dedup_hits: self.dedup_hits.get(),
            evictions: self.evictions.get(),
            evict_fail_all_pinned: self.evict_fail_all_pinned.get(),
            bypasses: self.bypasses.get(),
            prefetch_hits: self.prefetch_hits.get(),
            phys_reads: self.phys_reads.get(),
        }
    }
}

// ---------------------------------------------------------------------------
// ProbeResult — outcome of scanning a set for a vid
// ---------------------------------------------------------------------------

enum ProbeResult {
    /// Found READY entry — already pinned, return CacheGuard
    Hit { slot_idx: u32 },
    /// Found LOADING entry — caller should await WaitForReady
    Loading {
        slot_idx: u32,
        load_gen: u32,
    },
    /// Not found — caller should evict and load
    Miss,
}

// ---------------------------------------------------------------------------
// AdjacencyPool — the public cache
// ---------------------------------------------------------------------------

/// Per-core set-associative block cache for adjacency reads.
///
/// 8-way set-associative with clock eviction. Slots are pre-allocated
/// as a single contiguous 4KB-aligned buffer that doubles as IO target.
///
/// **CacheGuard discipline**: guards must not escape the expansion scope.
/// Decode neighbors from `guard.data()`, then drop the guard immediately.
/// Holding guards across await points or storing them in result sets will
/// pin slots and cause `evict_fail_all_pinned` to climb.
pub struct AdjacencyPool {
    state: RefCell<PoolState>,
    slot_store: SlotStore,
    num_sets: u32,
    stats: CacheStats,
}

impl AdjacencyPool {
    /// Create a new cache pool.
    ///
    /// `capacity_bytes` is rounded down to the nearest multiple of
    /// `BLOCK_SIZE * SET_WAYS` and the number of sets is rounded to a power of 2.
    pub fn new(capacity_bytes: usize) -> Self {
        let num_slots = (capacity_bytes / BLOCK_SIZE) as u32;
        let mut num_sets = (num_slots / SET_WAYS).max(1);
        // Round down to power of 2
        num_sets = 1 << (31 - num_sets.leading_zeros());
        let total_slots = num_sets * SET_WAYS;

        let entries: Vec<Entry> = (0..total_slots).map(|_| Entry::empty()).collect();

        Self {
            state: RefCell::new(PoolState {
                entries,
                prefetch: PrefetchChannel::new(),
            }),
            slot_store: SlotStore::new(total_slots),
            num_sets,
            stats: CacheStats::new(),
        }
    }

    /// Map vid → set index using Fibonacci hashing.
    fn set_index(&self, vid: u32) -> u32 {
        let bits = self.num_sets.trailing_zeros();
        if bits == 0 {
            return 0; // single set
        }
        let h = (vid as u64).wrapping_mul(FIB_HASH);
        (h >> (64 - bits)) as u32
    }

    /// Base slot index for a given set.
    fn set_base(&self, set_idx: u32) -> u32 {
        set_idx * SET_WAYS
    }

    /// Probe the set for `vid`. If found READY, pins it. If LOADING, returns
    /// the slot info for WaitForReady. If not found, returns Miss.
    ///
    /// Caller must hold NO RefCell borrow before calling.
    fn probe_set(&self, vid: u32) -> ProbeResult {
        let set_idx = self.set_index(vid);
        let base = self.set_base(set_idx);
        let mut state = self.state.borrow_mut();

        for way in 0..SET_WAYS {
            let idx = base + way;
            let entry = &mut state.entries[idx as usize];
            if entry.vid != vid {
                continue;
            }
            match entry.state {
                SlotState::Ready => {
                    entry.referenced = true;
                    if entry.prefetched {
                        entry.prefetched = false;
                        self.stats.inc_prefetch_hits();
                    }
                    entry.pin_count += 1;
                    self.stats.inc_hits();
                    return ProbeResult::Hit { slot_idx: idx };
                }
                SlotState::Loading => {
                    self.stats.inc_dedup_hits();
                    return ProbeResult::Loading {
                        slot_idx: idx,
                        load_gen: entry.load_gen,
                    };
                }
                SlotState::Empty => {}
            }
        }
        ProbeResult::Miss
    }

    /// Find an empty slot or evict one within the set. Returns slot index or None.
    ///
    /// Clock eviction within set, effort-capped to 2 × SET_WAYS iterations.
    /// No global clock_hand — each eviction sweeps from way 0. For 8-way sets
    /// the sweep is at most 16 iterations, always inline.
    fn find_or_evict(&self, vid: u32) -> Option<u32> {
        let set_idx = self.set_index(vid);
        let base = self.set_base(set_idx);
        let mut state = self.state.borrow_mut();
        Self::find_or_evict_inner(&mut state, &self.stats, base)
    }

    /// Inner eviction logic operating on an already-borrowed PoolState.
    /// Used by both `find_or_evict` and `prefetch_hint` (which already holds borrow_mut).
    fn find_or_evict_inner(
        state: &mut PoolState,
        stats: &CacheStats,
        base: u32,
    ) -> Option<u32> {
        // Pass 1: any empty slot?
        for way in 0..SET_WAYS {
            let idx = base + way;
            if state.entries[idx as usize].state == SlotState::Empty {
                return Some(idx);
            }
        }

        // Pass 2: clock sweep, max 2 × SET_WAYS iterations
        let max_iter = 2 * SET_WAYS;

        for i in 0..max_iter {
            let way = i % SET_WAYS;
            let idx = base + way;
            let entry = &mut state.entries[idx as usize];

            // Skip LOADING (IO in flight)
            if entry.state == SlotState::Loading {
                continue;
            }
            // Skip pinned (Seastar rule) or hub-pinned
            if entry.pin_count > 0 || entry.pinned {
                continue;
            }
            // Second chance
            if entry.referenced {
                entry.referenced = false;
                continue;
            }

            // Evict this slot
            entry.waiters.wake_all();
            entry.reset();
            stats.inc_evictions();
            return Some(idx);
        }

        // All slots pinned or loading
        stats.inc_evict_fail();
        None
    }

    /// Check if a vid is currently cached (any state except Empty).
    pub fn is_resident(&self, vid: u32) -> bool {
        let set_idx = self.set_index(vid);
        let base = self.set_base(set_idx);
        let state = self.state.borrow();
        for way in 0..SET_WAYS {
            let idx = (base + way) as usize;
            if state.entries[idx].vid == vid && state.entries[idx].state != SlotState::Empty {
                return true;
            }
        }
        false
    }

    /// Get a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// Reset all entries to EMPTY. Used for cold_per_query benchmarks.
    ///
    /// Panics if any entry is pinned (CacheGuard held) or in LOADING state.
    /// Caller must ensure no in-flight IO or held guards.
    /// Reset all non-pinned entries to EMPTY. Used for per-query cold benchmarks.
    ///
    /// **Caller must ensure no in-flight IO**: stop the prefetch worker and
    /// await its JoinHandle before calling this. Panics if any non-pinned entry
    /// is in LOADING state or has a held CacheGuard (pin_count > 0).
    pub fn clear(&self) {
        let mut state = self.state.borrow_mut();
        for entry in state.entries.iter_mut() {
            // Hub-pinned entries survive clear
            if entry.pinned {
                continue;
            }
            assert!(
                entry.state != SlotState::Loading,
                "clear() called with LOADING entry (vid={}). \
                 Stop prefetch worker and await its JoinHandle before calling clear().",
                entry.vid
            );
            assert!(
                entry.pin_count == 0,
                "clear() called with pinned entry (vid={}, pin_count={})",
                entry.vid, entry.pin_count
            );
            entry.waiters.wake_all();
            entry.reset();
        }
        // Drain any pending prefetch hints
        state.prefetch.drain();
    }

    /// Pre-load and pin pages as non-evictable hub pages.
    ///
    /// Each page is loaded via `get_or_load`, then marked `pinned = true`.
    /// Set-aware constraint: at most `MAX_PINNED_PER_SET` (4) entries per set
    /// can be pinned — excess pages in a crowded set are silently skipped.
    /// Returns the number of pages actually pinned.
    pub async fn pin_pages(&self, page_ids: &[u32], io: &IoDriver) -> io::Result<u32> {
        // Dedup page_ids to avoid wasted IO and double-counting
        let mut seen = std::collections::HashSet::with_capacity(page_ids.len());
        let mut pinned_count = 0u32;

        for &page_id in page_ids {
            if !seen.insert(page_id) {
                continue; // duplicate
            }

            // Check set saturation and already-pinned BEFORE issuing IO
            {
                let set_idx = self.set_index(page_id);
                let base = self.set_base(set_idx);
                let state = self.state.borrow();
                let mut set_pinned = 0u32;
                let mut already_pinned = false;
                for way in 0..SET_WAYS {
                    let entry = &state.entries[(base + way) as usize];
                    if entry.pinned {
                        set_pinned += 1;
                    }
                    if entry.vid == page_id && entry.pinned {
                        already_pinned = true;
                        break;
                    }
                }
                if already_pinned || set_pinned >= MAX_PINNED_PER_SET {
                    continue; // already pinned or set saturated — no IO
                }
            }

            // Load the page into cache
            let guard = self.get_or_load(page_id, io).await?;
            drop(guard); // release CacheGuard pin_count

            // Mark the entry as hub-pinned
            let set_idx = self.set_index(page_id);
            let base = self.set_base(set_idx);
            let mut state = self.state.borrow_mut();
            for way in 0..SET_WAYS {
                let idx = (base + way) as usize;
                if state.entries[idx].vid == page_id
                    && state.entries[idx].state == SlotState::Ready
                {
                    state.entries[idx].pinned = true;
                    pinned_count += 1;
                    break;
                }
            }
        }
        Ok(pinned_count)
    }

    /// Total number of slots in the pool.
    pub fn total_slots(&self) -> u32 {
        self.num_sets * SET_WAYS
    }

    /// Async lookup-or-load. Returns a CacheGuard pinning the block.
    ///
    /// On hit: returns immediately (no IO).
    /// On dedup: awaits existing LOADING flight, then re-probes.
    /// On miss: evicts if needed, reads from disk into slot, transitions to READY.
    /// On evict failure (all pinned): bypasses cache with direct uncached read.
    ///
    /// **Important**: Drop the returned CacheGuard as soon as you're done
    /// decoding the adjacency block. Do not hold it across await points or
    /// store it in result sets.
    pub async fn get_or_load(&self, vid: u32, io: &IoDriver) -> io::Result<CacheGuard<'_>> {
        // Guard 1: overflow_retries bounds the reprobe loop. If we can't
        // register as a waiter after MAX_OVERFLOW_RETRIES attempts (waiter
        // array full every time), fall back to bypass read. This prevents
        // soft livelock under extreme waiter pressure (prefetch window saturating
        // a hot set). The cost is one duplicate IO — acceptable for progress.
        const MAX_OVERFLOW_RETRIES: u32 = 2;
        let mut overflow_retries: u32 = 0;

        loop {
            match self.probe_set(vid) {
                ProbeResult::Hit { slot_idx } => {
                    return Ok(CacheGuard {
                        pool: self,
                        slot_idx,
                        bypass_buf: None,
                    });
                }
                ProbeResult::Loading {
                    slot_idx,
                    load_gen,
                } => {
                    // Dedup: await existing flight, then re-probe
                    let wait = WaitForReady {
                        pool: self,
                        slot_idx,
                        vid,
                        load_gen,
                        registered: false,
                    };
                    let did_wait = wait.await;

                    if !did_wait {
                        // WaitForReady returned immediately: overflow (waiter
                        // array full), gen mismatch, or IO-failed rollback.
                        //
                        // Guard 2: yield to executor before re-probing. monoio
                        // has no public submit/flush API, but its executor does
                        // IO poll during scheduling. YieldOnce (wake_by_ref +
                        // Pending) returns us to the run queue; the executor
                        // processes CQEs before re-scheduling us.
                        //
                        // DO NOT REMOVE THIS YIELD — without it, overflow causes
                        // a busy-loop that starves IO completion and hangs the core.
                        YieldOnce::new().await;

                        // Guard 1: after MAX_OVERFLOW_RETRIES immediate returns,
                        // fall back to bypass (uncached direct read). This trades
                        // one duplicate IO for guaranteed progress — prevents soft
                        // livelock when a hot set's waiter array stays saturated.
                        overflow_retries += 1;
                        if overflow_retries > MAX_OVERFLOW_RETRIES {
                            self.stats.inc_bypasses();
                            let buf = io.read_adj_block(vid).await?;
                            return Ok(CacheGuard {
                                pool: self,
                                slot_idx: u32::MAX,
                                bypass_buf: Some(buf),
                            });
                        }
                    }
                    // Normal dedup (did_wait=true): IO completed, re-probe
                    // for CacheGuard. No overflow counting needed.

                    continue; // re-probe to get CacheGuard
                }
                ProbeResult::Miss => {
                    self.stats.inc_misses();
                    break;
                }
            }
        }

        // Miss path: evict and load
        let slot_idx = match self.find_or_evict(vid) {
            Some(idx) => idx,
            None => {
                // All ways pinned/loading — bypass cache with direct read
                self.stats.inc_bypasses();
                self.stats.inc_phys_reads();
                let buf = io.read_adj_block(vid).await?;
                return Ok(CacheGuard {
                    pool: self,
                    slot_idx: u32::MAX, // sentinel: bypass mode
                    bypass_buf: Some(buf),
                });
            }
        };

        // Transition to LOADING
        let current_gen = {
            let mut state = self.state.borrow_mut();
            let entry = &mut state.entries[slot_idx as usize];
            entry.vid = vid;
            entry.state = SlotState::Loading;
            entry.load_gen = entry.load_gen.wrapping_add(1);
            entry.referenced = true;
            entry.pin_count = 0;
            entry.load_gen
        };

        // LoadGuard: rolls back LOADING → EMPTY on drop if not disarmed
        let mut load_guard = LoadGuard {
            pool: self,
            slot_idx,
            vid,
            load_gen: current_gen,
            disarmed: false,
        };

        // Read directly into slot buffer (no memcpy, no flight pool)
        let slot_ptr = self.slot_store.slot_ptr(slot_idx);
        io.read_adj_block_direct(vid, slot_ptr).await?;
        self.stats.inc_phys_reads();

        // IO succeeded — disarm guard, transition to READY
        load_guard.disarm();
        {
            let mut state = self.state.borrow_mut();
            let entry = &mut state.entries[slot_idx as usize];
            entry.state = SlotState::Ready;
            entry.pin_count = 1;
            entry.waiters.wake_all();
        }

        Ok(CacheGuard {
            pool: self,
            slot_idx,
            bypass_buf: None,
        })
    }

    // -----------------------------------------------------------------------
    // Prefetch API
    // -----------------------------------------------------------------------

    /// Enqueue a prefetch hint for `vid`. Sync (no await, no spawn).
    ///
    /// 1. If vid is already READY or LOADING in its set: return (no-op).
    /// 2. If the prefetch channel is full: return silently (backpressure).
    /// 3. Evict a slot, set it to LOADING with `prefetched=true`, push to channel.
    /// 4. Channel push auto-wakes the prefetch worker if queue was empty.
    pub fn prefetch_hint(&self, vid: u32) {
        let set_idx = self.set_index(vid);
        let base = self.set_base(set_idx);
        let mut state = self.state.borrow_mut();

        // Paused: silently drop new hints (quiesce for per-query clear)
        if state.prefetch.paused {
            return;
        }

        // Check if vid is already present (READY or LOADING)
        for way in 0..SET_WAYS {
            let idx = (base + way) as usize;
            let entry = &mut state.entries[idx];
            if entry.vid == vid && entry.state != SlotState::Empty {
                if entry.state == SlotState::Ready {
                    entry.referenced = true;
                }
                return;
            }
        }

        // Channel full → drop hint silently
        if state.prefetch.is_full() {
            return;
        }

        // Find or evict a slot
        let slot_idx = match Self::find_or_evict_inner(&mut state, &self.stats, base) {
            Some(idx) => idx,
            None => return, // all pinned/loading
        };

        // Transition to LOADING with prefetched flag
        let entry = &mut state.entries[slot_idx as usize];
        entry.vid = vid;
        entry.state = SlotState::Loading;
        entry.load_gen = entry.load_gen.wrapping_add(1);
        entry.referenced = true;
        entry.prefetched = true;
        entry.pin_count = 0;
        let load_gen = entry.load_gen;

        // Push to channel (auto-wakes worker if was empty)
        state.prefetch.push(PrefetchEntry {
            vid,
            slot_idx,
            load_gen,
        });
    }

    /// Check if `vid` is currently in LOADING state.
    pub fn is_loading(&self, vid: u32) -> bool {
        let set_idx = self.set_index(vid);
        let base = self.set_base(set_idx);
        let state = self.state.borrow();
        for way in 0..SET_WAYS {
            let idx = (base + way) as usize;
            if state.entries[idx].vid == vid && state.entries[idx].state == SlotState::Loading {
                return true;
            }
        }
        false
    }

    /// Returns true if any entry is in LOADING state.
    pub fn has_loading(&self) -> bool {
        let state = self.state.borrow();
        state.entries.iter().any(|e| e.state == SlotState::Loading)
    }

    /// Pause/unpause prefetch hint acceptance.
    /// When paused, `prefetch_hint()` silently drops new hints.
    /// The worker stays alive but idles (no new work enters the channel).
    pub fn pause_prefetch(&self, paused: bool) {
        let mut state = self.state.borrow_mut();
        state.prefetch.paused = paused;
    }

    /// Drain pending prefetch hints without stopping the worker.
    /// The worker stays alive but idles (empty channel).
    ///
    /// IMPORTANT: Drained entries may correspond to slots already set to LOADING.
    /// Since the prefetch task will never be spawned for these, we must reset
    /// the orphaned LOADING slots to prevent `has_loading()` from spinning forever.
    pub fn drain_prefetch(&self) {
        let mut state = self.state.borrow_mut();
        while let Some(pe) = state.prefetch.pop() {
            let entry = &mut state.entries[pe.slot_idx as usize];
            // Only reset if this is still the same LOADING entry we queued
            if entry.vid == pe.vid && entry.load_gen == pe.load_gen
                && entry.state == SlotState::Loading
            {
                entry.waiters.wake_all();
                entry.reset();
            }
        }
    }

    /// Signal the prefetch worker to stop. Wake it if it's sleeping.
    pub fn stop_prefetch(&self) {
        let mut state = self.state.borrow_mut();
        state.prefetch.stopped = true;
        if let Some(w) = state.prefetch.waker.take() {
            w.wake();
        }
    }

    /// Spawn the per-core prefetch worker as a monoio task.
    ///
    /// `prefetch_budget`: max concurrent prefetch IOs. This is the prefetch-
    /// specific cap — prevents prefetch from starving `get_or_load` for adj_sem
    /// permits. Total inflight IO is bounded by min(adj_sem, search + prefetch_budget).
    ///
    /// Guideline: set prefetch_budget to 2–4. Higher values increase inflight
    /// depth but also increase p99 due to permit contention with the search path.
    ///
    /// Call once per core. Stop via `pool.stop_prefetch()` + `handle.await`.
    pub fn spawn_prefetch_worker(
        pool: std::rc::Rc<AdjacencyPool>,
        io: std::rc::Rc<IoDriver>,
        prefetch_budget: usize,
    ) -> monoio::task::JoinHandle<()> {
        let pf_sem = std::rc::Rc::new(crate::io::LocalSemaphore::new(prefetch_budget));
        monoio::spawn(prefetch_worker(pool, io, pf_sem))
    }
}

// ---------------------------------------------------------------------------
// PrefetchWait — future that waits for work in the prefetch channel
// ---------------------------------------------------------------------------

/// Polls the prefetch channel. Returns `true` if work available, `false` if stopped.
struct PrefetchWait<'a> {
    pool: &'a AdjacencyPool,
}

impl<'a> Future for PrefetchWait<'a> {
    type Output = bool;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        let this = self.get_mut();
        let mut state = this.pool.state.borrow_mut();
        if state.prefetch.stopped {
            return Poll::Ready(false);
        }
        if !state.prefetch.is_empty() {
            return Poll::Ready(true);
        }
        // Store waker and wait
        state.prefetch.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

// ---------------------------------------------------------------------------
// prefetch_worker — per-core singleton async task, bounded concurrency
// ---------------------------------------------------------------------------

/// Complete a single prefetch IO: acquire prefetch permit, read, transition.
///
/// Holds `pf_sem` permit for the duration of the IO — this is the cap that
/// prevents prefetch from starving `get_or_load` for adj_sem permits.
///
/// Two-level budget:
///   pf_sem  → bounds max concurrent prefetch IOs (e.g., 4)
///   adj_sem → bounds total IO depth (search + prefetch, e.g., 32)
/// So search always has at least (adj_capacity - prefetch_budget) permits.
async fn prefetch_one_read(
    pool: std::rc::Rc<AdjacencyPool>,
    io: std::rc::Rc<IoDriver>,
    pf_sem: std::rc::Rc<crate::io::LocalSemaphore>,
    pe: PrefetchEntry,
) {
    // Acquire prefetch budget permit first — if all budget slots are taken,
    // this suspends until one frees up. Bounded by prefetch_budget (e.g., 4),
    // not by channel size (16). No spawn storm.
    let _pf_permit = pf_sem.acquire().await;

    let slot_ptr = pool.slot_store.slot_ptr(pe.slot_idx);
    match io.read_adj_block_direct(pe.vid, slot_ptr).await {
        Ok(()) => {
            pool.stats.inc_phys_reads();
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[pe.slot_idx as usize];
            if entry.vid == pe.vid && entry.load_gen == pe.load_gen
                && entry.state == SlotState::Loading
            {
                entry.state = SlotState::Ready;
                entry.waiters.wake_all();
            }
        }
        Err(_) => {
            // IO failed — roll back LOADING → EMPTY
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[pe.slot_idx as usize];
            if entry.vid == pe.vid && entry.load_gen == pe.load_gen
                && entry.state == SlotState::Loading
            {
                entry.reset();
                entry.waiters.wake_all();
            }
        }
    }
    // _pf_permit drops here → releases one prefetch budget slot
}

/// Per-core prefetch worker. Consumes hints from the channel and spawns
/// bounded IO tasks.
///
/// Each spawned task must acquire a permit from `pf_sem` before issuing IO.
/// This limits active prefetch IOs to `prefetch_budget`, regardless of how
/// many hints are queued. Excess tasks suspend on pf_sem until a slot frees.
///
/// Two-level backpressure:
///   1. Channel capacity (16) — bounds how many hints can be queued
///   2. pf_sem (prefetch_budget) — bounds how many IOs are in flight
///   3. adj_sem — bounds total IO (search + prefetch)
async fn prefetch_worker(
    pool: std::rc::Rc<AdjacencyPool>,
    io: std::rc::Rc<IoDriver>,
    pf_sem: std::rc::Rc<crate::io::LocalSemaphore>,
) {
    loop {
        // Wait for work or stop signal
        let has_work = PrefetchWait { pool: &pool }.await;
        if !has_work {
            break;
        }

        // Drain all available entries, spawning a bounded task for each
        loop {
            let pe = {
                let mut state = pool.state.borrow_mut();
                state.prefetch.pop()
            };
            let pe = match pe {
                Some(pe) => pe,
                None => break,
            };

            monoio::spawn(prefetch_one_read(
                std::rc::Rc::clone(&pool),
                std::rc::Rc::clone(&io),
                std::rc::Rc::clone(&pf_sem),
                pe,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// CacheGuard — RAII pin/unpin
// ---------------------------------------------------------------------------

/// RAII guard that pins a cache slot. Decrements pin_count on drop.
///
/// **Must not escape the expansion scope.** Decode the adjacency block
/// neighbors, then drop this guard immediately. Holding guards pins slots
/// and starves the eviction policy.
///
/// In bypass mode (all ways were pinned), holds a temporary AlignedBuf
/// instead of referencing a slot. No pin_count change on drop.
pub struct CacheGuard<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    bypass_buf: Option<crate::aligned::AlignedBuf>,
}

impl<'a> CacheGuard<'a> {
    /// Access the cached 4KB block data.
    pub fn data(&self) -> &[u8] {
        if let Some(ref buf) = self.bypass_buf {
            buf.as_slice()
        } else {
            self.pool.slot_store.slot_slice(self.slot_idx)
        }
    }
}

impl<'a> Drop for CacheGuard<'a> {
    fn drop(&mut self) {
        if self.bypass_buf.is_some() {
            return; // bypass mode — no slot to unpin
        }
        let mut state = self.pool.state.borrow_mut();
        let entry = &mut state.entries[self.slot_idx as usize];
        debug_assert!(entry.pin_count > 0, "CacheGuard drop with pin_count=0");
        entry.pin_count -= 1;
    }
}

// ---------------------------------------------------------------------------
// LoadGuard — RAII cancel safety for LOADING → EMPTY rollback
// ---------------------------------------------------------------------------

/// Rolls back a LOADING entry to EMPTY if not disarmed.
/// Covers: IO error, future cancellation (drop), timeout.
struct LoadGuard<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    vid: u32,
    load_gen: u32,
    disarmed: bool,
}

impl<'a> LoadGuard<'a> {
    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl<'a> Drop for LoadGuard<'a> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        let mut state = self.pool.state.borrow_mut();
        let entry = &mut state.entries[self.slot_idx as usize];
        if entry.vid == self.vid
            && entry.load_gen == self.load_gen
            && entry.state == SlotState::Loading
        {
            entry.state = SlotState::Empty;
            entry.vid = u32::MAX;
            entry.waiters.wake_all();
        }
    }
}

// ---------------------------------------------------------------------------
// YieldOnce — cooperative yield point
// ---------------------------------------------------------------------------

/// Yields once to the executor, then completes on next poll.
/// Used after WaitForReady to prevent spin-loops when waiter array
/// overflows or gen mismatches cause immediate Ready return.
struct YieldOnce {
    yielded: bool,
}

impl YieldOnce {
    fn new() -> Self {
        Self { yielded: false }
    }
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.yielded {
            Poll::Ready(())
        } else {
            this.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForReady — custom Future for dedup waiting
// ---------------------------------------------------------------------------

/// Awaits a LOADING entry to become READY (or detect that the flight is gone).
/// Matches on (slot_idx, vid, load_gen) to detect stale flights.
///
/// Uses `registered` flag to ensure each waiter pushes its waker exactly once,
/// preventing unbounded growth from spurious wakes or repeated polling.
///
/// Returns `bool`:
/// - `true` = successfully registered and was woken (normal dedup path)
/// - `false` = returned immediately without sleeping (overflow, gen mismatch,
///   or entry already transitioned). Caller should count these toward the
///   overflow bypass limit.
struct WaitForReady<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    vid: u32,
    load_gen: u32,
    /// Only register waker once. Prevents duplicate push on spurious wake.
    /// Reset is unnecessary — after wake_all the next poll sees Ready.
    registered: bool,
}

impl<'a> Future for WaitForReady<'a> {
    /// `true` = actually waited (registered + woken), `false` = returned immediately.
    type Output = bool;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this.pool.state.borrow_mut();
        let entry = &mut state.entries[this.slot_idx as usize];

        // Check vid + gen match
        if entry.vid != this.vid || entry.load_gen != this.load_gen {
            // Flight gone (IO failed + reused, or evicted). Caller re-probes.
            return Poll::Ready(false);
        }

        match entry.state {
            SlotState::Ready => {
                // If we were registered, we were woken normally (true).
                // If not registered, entry transitioned before we could wait (false).
                Poll::Ready(this.registered)
            }
            SlotState::Loading => {
                if !this.registered {
                    // First poll: register waker. If array is full, return
                    // immediately — caller will yield then re-probe or bypass.
                    if !entry.waiters.push(cx.waker().clone()) {
                        return Poll::Ready(false); // overflow
                    }
                    this.registered = true;
                }
                // Subsequent polls (spurious wake): stay Pending, don't re-push.
                // The waker registered on first poll is still in the array
                // and will fire when wake_all runs.
                Poll::Pending
            }
            SlotState::Empty => {
                // IO failed, entry rolled back. Caller re-probes.
                Poll::Ready(false)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(num_sets: u32) -> AdjacencyPool {
        let cap = num_sets as usize * SET_WAYS as usize * BLOCK_SIZE;
        AdjacencyPool::new(cap)
    }

    #[test]
    fn pool_construction() {
        let pool = make_pool(4);
        assert_eq!(pool.num_sets, 4);
        assert_eq!(pool.total_slots(), 32);
        assert_eq!(pool.stats().hits, 0);
    }

    #[test]
    fn pool_power_of_two_rounding() {
        // 3 sets → rounds down to 2
        let cap = 3 * SET_WAYS as usize * BLOCK_SIZE;
        let pool = AdjacencyPool::new(cap);
        assert_eq!(pool.num_sets, 2);
        assert_eq!(pool.total_slots(), 16);
    }

    #[test]
    fn set_index_distribution() {
        let pool = make_pool(8);
        // Fibonacci hashing should spread sequential vids across sets
        let mut set_counts = [0u32; 8];
        for vid in 0..64u32 {
            let idx = pool.set_index(vid);
            assert!(idx < 8, "set_index out of range: {}", idx);
            set_counts[idx as usize] += 1;
        }
        // Each set should get roughly 8 (64 / 8), allow ±2 for hash non-uniformity
        for &c in &set_counts {
            assert!(
                c >= 6 && c <= 10,
                "Fibonacci hash distribution too skewed: {:?}",
                set_counts
            );
        }
    }

    #[test]
    fn find_or_evict_empty_slots() {
        let pool = make_pool(1);
        // All 8 slots should be empty — should always find one
        for vid in 0..8u32 {
            let slot = pool.find_or_evict(vid);
            assert!(slot.is_some(), "should find empty slot for vid {}", vid);
            // Mark as Ready so next call can't reuse
            let idx = slot.unwrap();
            let mut state = pool.state.borrow_mut();
            state.entries[idx as usize].vid = vid;
            state.entries[idx as usize].state = SlotState::Ready;
        }
    }

    #[test]
    fn find_or_evict_clock_eviction() {
        let pool = make_pool(1);
        // Fill all 8 slots with READY, unreferenced, unpinned
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = 0;
            }
        }

        // Should evict one
        let slot = pool.find_or_evict(100);
        assert!(slot.is_some());
        assert_eq!(pool.stats().evictions, 1);
    }

    #[test]
    fn find_or_evict_skips_pinned() {
        let pool = make_pool(1);
        // Fill all 8 slots: 7 pinned, 1 unpinned unreferenced
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = if way < 7 { 1 } else { 0 };
            }
        }

        // Should evict the unpinned one (way 7)
        let slot = pool.find_or_evict(100);
        assert!(slot.is_some());
        let evicted_idx = slot.unwrap();
        assert_eq!(evicted_idx, 7); // way 7 is the only unpinned
    }

    #[test]
    fn find_or_evict_skips_hub_pinned() {
        let pool = make_pool(1);
        // Fill all 8 slots: 7 hub-pinned (pinned=true), 1 unpinned
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = 0;
                entry.pinned = way < 7;
            }
        }

        // Should evict the non-hub-pinned one (way 7)
        let slot = pool.find_or_evict(100);
        assert!(slot.is_some());
        let evicted_idx = slot.unwrap();
        assert_eq!(evicted_idx, 7);
    }

    #[test]
    fn clear_preserves_hub_pinned() {
        let pool = make_pool(1);
        // Place 2 hub-pinned entries and 6 normal entries
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = true;
                entry.pinned = way < 2;
            }
        }

        pool.clear();

        // Hub-pinned entries should survive
        let state = pool.state.borrow();
        for way in 0..SET_WAYS {
            let entry = &state.entries[way as usize];
            if way < 2 {
                assert_eq!(entry.vid, way, "hub-pinned entry should survive clear");
                assert!(entry.pinned);
            } else {
                assert_eq!(entry.state, SlotState::Empty, "non-pinned entry should be cleared");
            }
        }
    }

    #[test]
    fn find_or_evict_all_pinned_fails() {
        let pool = make_pool(1);
        // Fill all 8 slots as pinned
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.pin_count = 1;
            }
        }

        let slot = pool.find_or_evict(100);
        assert!(slot.is_none());
        assert_eq!(pool.stats().evict_fail_all_pinned, 1);
    }

    #[test]
    fn find_or_evict_second_chance() {
        let pool = make_pool(1);
        // Fill all 8 slots: all referenced, unpinned
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = true;
                entry.pin_count = 0;
            }
        }

        // First sweep clears referenced bits, second sweep evicts
        let slot = pool.find_or_evict(100);
        assert!(slot.is_some());
        // Should have cleared referenced bits on first pass, evicted on second
        assert_eq!(pool.stats().evictions, 1);
    }

    #[test]
    fn probe_set_miss() {
        let pool = make_pool(1);
        match pool.probe_set(42) {
            ProbeResult::Miss => {}
            _ => panic!("expected Miss for empty pool"),
        }
    }

    #[test]
    fn probe_set_hit() {
        let pool = make_pool(1);
        // Manually place a READY entry
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Ready;
        }

        match pool.probe_set(vid) {
            ProbeResult::Hit { slot_idx } => {
                assert_eq!(slot_idx, base);
                // Should have pinned it
                let state = pool.state.borrow();
                assert_eq!(state.entries[base as usize].pin_count, 1);
                assert!(state.entries[base as usize].referenced);
                drop(state);
                // Unpin
                let guard = CacheGuard {
                    pool: &pool,
                    slot_idx,
                    bypass_buf: None,
                };
                drop(guard);
                let state = pool.state.borrow();
                assert_eq!(state.entries[base as usize].pin_count, 0);
            }
            _ => panic!("expected Hit"),
        }
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn probe_set_loading_dedup() {
        let pool = make_pool(1);
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Loading;
            entry.load_gen = 5;
        }

        match pool.probe_set(vid) {
            ProbeResult::Loading { slot_idx, load_gen } => {
                assert_eq!(slot_idx, base);
                assert_eq!(load_gen, 5);
            }
            _ => panic!("expected Loading"),
        }
        assert_eq!(pool.stats().dedup_hits, 1);
    }

    #[test]
    fn cache_guard_unpin_on_drop() {
        let pool = make_pool(1);
        {
            let mut state = pool.state.borrow_mut();
            state.entries[0].vid = 0;
            state.entries[0].state = SlotState::Ready;
            state.entries[0].pin_count = 1;
        }

        let guard = CacheGuard {
            pool: &pool,
            slot_idx: 0,
            bypass_buf: None,
        };
        drop(guard);

        let state = pool.state.borrow();
        assert_eq!(state.entries[0].pin_count, 0);
    }

    #[test]
    fn cache_guard_bypass_no_unpin() {
        let pool = make_pool(1);
        // Bypass guard should not touch any entry
        let guard = CacheGuard {
            pool: &pool,
            slot_idx: u32::MAX,
            bypass_buf: Some(crate::aligned::AlignedBuf::new(BLOCK_SIZE)),
        };
        let data = guard.data();
        assert_eq!(data.len(), 0); // AlignedBuf::new has len=0 (no data written)
        drop(guard); // should not panic
    }

    #[test]
    fn load_guard_rollback_on_drop() {
        let pool = make_pool(1);
        {
            let mut state = pool.state.borrow_mut();
            state.entries[0].vid = 42;
            state.entries[0].state = SlotState::Loading;
            state.entries[0].load_gen = 1;
        }

        // Simulate cancel: drop LoadGuard without disarming
        {
            let _guard = LoadGuard {
                pool: &pool,
                slot_idx: 0,
                vid: 42,
                load_gen: 1,
                disarmed: false,
            };
        } // drops here

        let state = pool.state.borrow();
        assert_eq!(state.entries[0].state, SlotState::Empty);
        assert_eq!(state.entries[0].vid, u32::MAX);
    }

    #[test]
    fn load_guard_no_rollback_when_disarmed() {
        let pool = make_pool(1);
        {
            let mut state = pool.state.borrow_mut();
            state.entries[0].vid = 42;
            state.entries[0].state = SlotState::Loading;
            state.entries[0].load_gen = 1;
        }

        {
            let mut guard = LoadGuard {
                pool: &pool,
                slot_idx: 0,
                vid: 42,
                load_gen: 1,
                disarmed: false,
            };
            guard.disarm();
        } // drops here

        let state = pool.state.borrow();
        // Should NOT have rolled back
        assert_eq!(state.entries[0].state, SlotState::Loading);
        assert_eq!(state.entries[0].vid, 42);
    }

    #[test]
    fn load_guard_stale_gen_no_rollback() {
        let pool = make_pool(1);
        {
            let mut state = pool.state.borrow_mut();
            state.entries[0].vid = 42;
            state.entries[0].state = SlotState::Loading;
            state.entries[0].load_gen = 2; // gen moved past guard's gen
        }

        {
            let _guard = LoadGuard {
                pool: &pool,
                slot_idx: 0,
                vid: 42,
                load_gen: 1, // stale load_gen
                disarmed: false,
            };
        }

        let state = pool.state.borrow();
        // Should NOT roll back (gen mismatch)
        assert_eq!(state.entries[0].state, SlotState::Loading);
    }

    #[test]
    fn slot_store_alignment() {
        let store = SlotStore::new(4);
        for i in 0..4u32 {
            let ptr = store.slot_ptr(i);
            assert_eq!(ptr.as_mut_ptr() as usize % BLOCK_SIZE, 0, "slot {} not aligned", i);
        }
    }

    #[test]
    fn slot_store_write_read() {
        let store = SlotStore::new(2);
        // Write pattern to slot 0
        let ptr = store.slot_ptr(0);
        unsafe {
            std::ptr::write_bytes(ptr.as_mut_ptr(), 0xAB, BLOCK_SIZE);
        }
        let data = store.slot_slice(0);
        assert!(data.iter().all(|&b| b == 0xAB));

        // Slot 1 should still be zeroed
        let data1 = store.slot_slice(1);
        assert!(data1.iter().all(|&b| b == 0));
    }

    #[test]
    fn is_resident() {
        let pool = make_pool(1);
        assert!(!pool.is_resident(42));

        let set_idx = pool.set_index(42);
        let base = pool.set_base(set_idx);
        {
            let mut state = pool.state.borrow_mut();
            state.entries[base as usize].vid = 42;
            state.entries[base as usize].state = SlotState::Ready;
        }
        assert!(pool.is_resident(42));
    }

    /// Acceptance Gate 1: hit path cost.
    ///
    /// A cache hit (probe_set → CacheGuard → drop) must be cheap enough that
    /// it's negligible next to a distance computation (~100-500ns for 768d).
    /// Target: median < 100ns per hit cycle.
    #[test]
    fn acceptance_gate_hit_path_cost() {
        let pool = make_pool(16); // 128 slots across 16 sets
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);

        // Seed a READY entry
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Ready;
            entry.referenced = false;
            entry.pin_count = 0;
        }

        // Warmup
        for _ in 0..1000 {
            match pool.probe_set(vid) {
                ProbeResult::Hit { slot_idx } => {
                    let guard = CacheGuard {
                        pool: &pool,
                        slot_idx,
                        bypass_buf: None,
                    };
                    let _data = guard.data();
                    drop(guard);
                }
                _ => panic!("expected hit"),
            }
        }

        // Measure
        let iterations = 100_000u64;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            match pool.probe_set(vid) {
                ProbeResult::Hit { slot_idx } => {
                    let guard = CacheGuard {
                        pool: &pool,
                        slot_idx,
                        bypass_buf: None,
                    };
                    let _data = guard.data();
                    drop(guard);
                }
                _ => panic!("expected hit"),
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let per_hit_ns = elapsed_ns / iterations;

        eprintln!(
            "\n=== ACCEPTANCE GATE 1: Hit Path Cost ===\n\
             iterations:   {}\n\
             total:        {} us\n\
             per hit:      {} ns\n\
             target:       < 100 ns\n\
             verdict:      {}",
            iterations,
            elapsed_ns / 1000,
            per_hit_ns,
            if per_hit_ns < 100 { "PASS" } else { "FAIL (but check: debug build adds overhead)" }
        );

        // In release mode, this should be < 50ns. In debug mode with bounds
        // checks and RefCell overhead, allow up to 500ns.
        assert!(
            per_hit_ns < 500,
            "hit path too slow: {} ns/hit (even debug build should be < 500ns)",
            per_hit_ns
        );
    }

    /// Acceptance gate: clock eviction stays bounded under full-set pressure.
    ///
    /// Fill a set to capacity, then repeatedly evict + insert new entries.
    /// Verify that eviction cost per cycle stays bounded (no degradation).
    #[test]
    fn acceptance_gate_eviction_bounded_cost() {
        let pool = make_pool(1); // single set, 8 ways

        // Fill all 8 slots: READY, unreferenced, unpinned
        {
            let mut state = pool.state.borrow_mut();
            for way in 0..SET_WAYS {
                let entry = &mut state.entries[way as usize];
                entry.vid = way;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = 0;
            }
        }

        // Warmup
        for vid in 100..200u32 {
            if let Some(idx) = pool.find_or_evict(vid) {
                let mut state = pool.state.borrow_mut();
                let entry = &mut state.entries[idx as usize];
                entry.vid = vid;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = 0;
            }
        }

        // Measure: evict + replace cycles
        let iterations = 50_000u64;
        let start = std::time::Instant::now();
        for i in 0..iterations {
            let vid = 1000 + i as u32;
            if let Some(idx) = pool.find_or_evict(vid) {
                let mut state = pool.state.borrow_mut();
                let entry = &mut state.entries[idx as usize];
                entry.vid = vid;
                entry.state = SlotState::Ready;
                entry.referenced = false;
                entry.pin_count = 0;
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let per_evict_ns = elapsed_ns / iterations;

        eprintln!(
            "\n=== ACCEPTANCE GATE: Eviction Bounded Cost ===\n\
             iterations:   {}\n\
             total:        {} us\n\
             per evict:    {} ns\n\
             target:       < 200 ns (8-way scan is O(16) max)\n\
             verdict:      {}",
            iterations,
            elapsed_ns / 1000,
            per_evict_ns,
            if per_evict_ns < 200 { "PASS" } else { "FAIL (debug build overhead?)" }
        );

        assert!(
            per_evict_ns < 1000,
            "eviction too slow: {} ns/cycle (even debug should be < 1us)",
            per_evict_ns
        );
    }

    // -----------------------------------------------------------------------
    // Helper: noop waker for manual polling
    // -----------------------------------------------------------------------

    fn noop_waker() -> Waker {
        use std::task::RawWaker;
        use std::task::RawWakerVTable;

        fn noop(_: *const ()) {}
        fn noop_clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(noop_clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    /// Gate 4: Repeated-poll stress — waker array doesn't grow from spurious wakes.
    ///
    /// Set up a LOADING entry, create N WaitForReady futures, poll each one
    /// multiple times (simulating spurious wakes). The waiter array must stay
    /// at exactly N entries (not N × polls).
    #[test]
    fn acceptance_gate_repeated_poll_no_growth() {
        let pool = make_pool(1);
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);

        // Set up a LOADING entry
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Loading;
            entry.load_gen = 1;
        }

        let num_waiters = 3usize; // < MAX_WAITERS_PER_ENTRY
        let polls_per_waiter = 50usize; // many spurious wakes

        // Create WaitForReady futures and poll them repeatedly
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut futures: Vec<_> = (0..num_waiters)
            .map(|_| WaitForReady {
                pool: &pool,
                slot_idx: base,
                vid,
                load_gen: 1,
                registered: false,
            })
            .collect();

        // Poll each future many times — simulates spurious wakes
        for _round in 0..polls_per_waiter {
            for f in futures.iter_mut() {
                let pinned = Pin::new(f);
                let result = pinned.poll(&mut cx);
                assert_eq!(result, Poll::Pending, "should still be LOADING");
            }
        }

        // Check: waiter array should have exactly num_waiters entries,
        // NOT num_waiters × polls_per_waiter
        {
            let state = pool.state.borrow();
            let entry = &state.entries[base as usize];
            let waiter_count = entry.waiters.len();

            eprintln!(
                "\n=== GATE 4: Repeated-Poll No Growth ===\n\
                 waiters:     {}\n\
                 futures:     {}\n\
                 polls/each:  {}\n\
                 max capacity: {}\n\
                 verdict:     {}",
                waiter_count,
                num_waiters,
                polls_per_waiter,
                MAX_WAITERS_PER_ENTRY,
                if waiter_count == num_waiters { "PASS" } else { "FAIL" }
            );

            assert_eq!(
                waiter_count, num_waiters,
                "waiter array grew beyond registered count: {} (expected {})",
                waiter_count, num_waiters
            );
        }

        // Now transition to READY and wake all — futures should complete
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.state = SlotState::Ready;
            entry.waiters.wake_all();
        }

        // Poll again — should all be Ready now
        for f in futures.iter_mut() {
            let pinned = Pin::new(f);
            let result = pinned.poll(&mut cx);
            assert_eq!(result, Poll::Ready(true), "should be Ready(true) after wake_all");
        }

        // Waiter array should be empty after wake_all
        {
            let state = pool.state.borrow();
            let entry = &state.entries[base as usize];
            assert_eq!(entry.waiters.len(), 0, "waiters should be empty after wake_all");
        }
    }

    /// Gate 5: Cancel storm — dropping waiters mid-flight doesn't corrupt state.
    ///
    /// Create N waiters on a LOADING entry, drop half of them, then complete
    /// the IO. Verify: no panics, waiter array is cleared, state is clean,
    /// subsequent get_or_load probes work normally.
    #[test]
    fn acceptance_gate_cancel_storm() {
        let pool = make_pool(1);
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);

        // Set up LOADING entry
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Loading;
            entry.load_gen = 1;
        }

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Create 4 waiters (fills WaiterArray exactly)
        let mut futures: Vec<_> = (0..MAX_WAITERS_PER_ENTRY)
            .map(|_| WaitForReady {
                pool: &pool,
                slot_idx: base,
                vid,
                load_gen: 1,
                registered: false,
            })
            .collect();

        // Poll all to register
        for f in futures.iter_mut() {
            let pinned = Pin::new(f);
            assert_eq!(pinned.poll(&mut cx), Poll::Pending);
        }

        // Verify all registered
        {
            let state = pool.state.borrow();
            assert_eq!(
                state.entries[base as usize].waiters.len(),
                MAX_WAITERS_PER_ENTRY,
                "all waiters should be registered"
            );
        }

        // Drop half the futures (simulating timeout/abort)
        let surviving = futures.split_off(MAX_WAITERS_PER_ENTRY / 2);
        drop(futures); // dropped futures leave zombie wakers in array — that's OK

        // Complete the IO: transition LOADING → READY, wake_all
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.state = SlotState::Ready;
            entry.pin_count = 0;
            entry.waiters.wake_all();
        }

        // Waiter array should be clear after wake_all
        {
            let state = pool.state.borrow();
            let entry = &state.entries[base as usize];
            assert_eq!(entry.waiters.len(), 0, "waiters should be empty after wake_all");
        }

        // Surviving futures should resolve cleanly
        for mut f in surviving {
            let pinned = Pin::new(&mut f);
            let result = pinned.poll(&mut cx);
            assert_eq!(result, Poll::Ready(true), "surviving waiter should see Ready(true)");
        }

        // State should be clean — subsequent probe_set should work
        match pool.probe_set(vid) {
            ProbeResult::Hit { slot_idx } => {
                assert_eq!(slot_idx, base);
                // Clean up: drop the guard
                let guard = CacheGuard {
                    pool: &pool,
                    slot_idx,
                    bypass_buf: None,
                };
                drop(guard);
            }
            other => panic!("expected Hit after cancel storm, got {:?}",
                match other {
                    ProbeResult::Miss => "Miss",
                    ProbeResult::Loading { .. } => "Loading",
                    ProbeResult::Hit { .. } => "Hit",
                }
            ),
        }

        // Also verify: a new LOADING cycle on the same slot works cleanly
        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.state = SlotState::Loading;
            entry.load_gen = entry.load_gen.wrapping_add(1);
            // No stale waiters should be present
            assert_eq!(entry.waiters.len(), 0);
        }

        eprintln!(
            "\n=== GATE 5: Cancel Storm ===\n\
             total waiters:     {}\n\
             cancelled (dropped): {}\n\
             survived:          {}\n\
             state after:       clean (READY, no stale waiters)\n\
             verdict:           PASS",
            MAX_WAITERS_PER_ENTRY,
            MAX_WAITERS_PER_ENTRY / 2,
            MAX_WAITERS_PER_ENTRY - MAX_WAITERS_PER_ENTRY / 2,
        );
    }

    /// Gate 4b: WaiterArray full — graceful degradation.
    ///
    /// If more than MAX_WAITERS_PER_ENTRY futures try to wait, the overflow
    /// ones should get Poll::Ready (forcing re-probe), not panic or block.
    #[test]
    fn acceptance_gate_waiter_overflow() {
        let pool = make_pool(1);
        let vid = 42u32;
        let set_idx = pool.set_index(vid);
        let base = pool.set_base(set_idx);

        {
            let mut state = pool.state.borrow_mut();
            let entry = &mut state.entries[base as usize];
            entry.vid = vid;
            entry.state = SlotState::Loading;
            entry.load_gen = 1;
        }

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Fill WaiterArray to capacity
        let mut saturating: Vec<_> = (0..MAX_WAITERS_PER_ENTRY)
            .map(|_| WaitForReady {
                pool: &pool,
                slot_idx: base,
                vid,
                load_gen: 1,
                registered: false,
            })
            .collect();

        for f in saturating.iter_mut() {
            let pinned = Pin::new(f);
            assert_eq!(pinned.poll(&mut cx), Poll::Pending);
        }

        // Now create one more — should get Poll::Ready (overflow, re-probe)
        let mut overflow = WaitForReady {
            pool: &pool,
            slot_idx: base,
            vid,
            load_gen: 1,
            registered: false,
        };
        let result = Pin::new(&mut overflow).poll(&mut cx);
        assert_eq!(
            result,
            Poll::Ready(false),
            "overflow waiter should get Ready(false) — not registered"
        );

        // Array should still have exactly MAX_WAITERS_PER_ENTRY entries
        {
            let state = pool.state.borrow();
            assert_eq!(
                state.entries[base as usize].waiters.len(),
                MAX_WAITERS_PER_ENTRY
            );
        }

        eprintln!(
            "\n=== GATE 4b: WaiterArray Overflow ===\n\
             capacity:         {}\n\
             registered:       {}\n\
             overflow result:  Ready (forced re-probe)\n\
             verdict:          PASS",
            MAX_WAITERS_PER_ENTRY,
            MAX_WAITERS_PER_ENTRY,
        );
    }
}
