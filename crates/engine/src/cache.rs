//! AdjacencyPool: per-core set-associative block cache for adjacency reads.
//!
//! Design contracts:
//! 1. No hot-path allocation — all slots pre-allocated, waiter is Option<Waker>
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
    pin_count: u32,    // live CacheGuards
    waiter: Option<Waker>,
}

impl Entry {
    fn empty() -> Self {
        Self {
            vid: u32::MAX,
            state: SlotState::Empty,
            load_gen: 0,
            referenced: false,
            pin_count: 0,
            waiter: None,
        }
    }

    fn reset(&mut self) {
        self.vid = u32::MAX;
        self.state = SlotState::Empty;
        self.referenced = false;
        self.pin_count = 0;
        // load_gen is NOT reset — monotonically increasing
        // waiter is NOT cleared here — find_or_evict caller handles it
    }
}

// ---------------------------------------------------------------------------
// PoolState — mutable interior behind RefCell
// ---------------------------------------------------------------------------

struct PoolState {
    entries: Vec<Entry>,
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
}

struct CacheStats {
    hits: Cell<u64>,
    misses: Cell<u64>,
    dedup_hits: Cell<u64>,
    evictions: Cell<u64>,
    evict_fail_all_pinned: Cell<u64>,
    bypasses: Cell<u64>,
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

    fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.get(),
            misses: self.misses.get(),
            dedup_hits: self.dedup_hits.get(),
            evictions: self.evictions.get(),
            evict_fail_all_pinned: self.evict_fail_all_pinned.get(),
            bypasses: self.bypasses.get(),
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
            // Skip pinned (Seastar rule)
            if entry.pin_count > 0 {
                continue;
            }
            // Second chance
            if entry.referenced {
                entry.referenced = false;
                continue;
            }

            // Evict this slot
            if let Some(w) = entry.waiter.take() {
                w.wake();
            }
            entry.reset();
            self.stats.inc_evictions();
            return Some(idx);
        }

        // All slots pinned or loading
        self.stats.inc_evict_fail();
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
                    };
                    wait.await;
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

        // IO succeeded — disarm guard, transition to READY
        load_guard.disarm();
        {
            let mut state = self.state.borrow_mut();
            let entry = &mut state.entries[slot_idx as usize];
            entry.state = SlotState::Ready;
            entry.pin_count = 1;
            if let Some(w) = entry.waiter.take() {
                w.wake();
            }
        }

        Ok(CacheGuard {
            pool: self,
            slot_idx,
            bypass_buf: None,
        })
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
            if let Some(w) = entry.waiter.take() {
                w.wake();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForReady — custom Future for dedup waiting
// ---------------------------------------------------------------------------

/// Awaits a LOADING entry to become READY (or detect that the flight is gone).
/// Matches on (slot_idx, vid, load_gen) to detect stale flights.
///
/// Only registers the first waker — subsequent polls don't overwrite,
/// reducing unnecessary waker clones in the single-threaded executor.
///
/// Returns `()` always — gen mismatch / IO failure cause immediate return,
/// caller re-probes. This is control flow, not an error.
struct WaitForReady<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    vid: u32,
    load_gen: u32,
}

impl<'a> Future for WaitForReady<'a> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this.pool.state.borrow_mut();
        let entry = &mut state.entries[this.slot_idx as usize];

        // Check vid + gen match
        if entry.vid != this.vid || entry.load_gen != this.load_gen {
            // Flight gone (IO failed + reused, or evicted). Caller re-probes.
            return Poll::Ready(());
        }

        match entry.state {
            SlotState::Ready => Poll::Ready(()),
            SlotState::Loading => {
                // Only register first waker — don't overwrite existing
                if entry.waiter.is_none() {
                    entry.waiter = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            SlotState::Empty => {
                // IO failed, entry rolled back. Caller re-probes.
                Poll::Ready(())
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
}
