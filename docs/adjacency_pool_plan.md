# AdjacencyPool: Per-Core Block Cache

## Context

The disk beam search issues an io_uring read for every adjacency expansion — no caching.
AdjacencyPool adds a per-core fixed-capacity cache. Single API change:
`io.read_adj_block(vid)` → `pool.get_or_load(vid, io)`.

---

## Research Distillation

### From RocksDB HyperClockCache (`~/repos/rocksdb/cache/clock_cache.h`)

**64-bit packed metadata per slot** — the core innovation:
```
acquire_counter(30b) | release_counter(30b) | hit(1b) | occupied(1b) | shareable(1b) | visible(1b)
```
- RefCount = acquire − release. Lookup = single fetch_add on acquire. Release = single
  fetch_add on release. No CAS on the hot path.
- Four states: Empty → Construction → Visible → Invisible. Only shareable→construction
  needs CAS (must verify zero refs).

**Eviction effort cap** (`clock_cache.cc:568`): stops if `(freed+1) * 30 <= seen_pinned`.
Prevents unbounded CPU when most entries are pinned. We must have this.

**Priority countdown**: entries get initial score by priority (HIGH=3, LOW=1). Clock
decrements on sweep, evicts when 0. More useful than a single second-chance bit when
we later add typed priorities (adjacency=3, vector=1).

**Charge accounting**: `usage_` (all entries) + `standalone_usage_` (not in table) +
`GetPinnedUsage()`. We need at minimum `usage` and `pinned_usage`.

**What to adopt**: effort-capped eviction, priority countdown (future), charge tracking.
**What to skip**: split acquire/release counters (overkill for single-threaded), dynamic
table resizing (we're fixed-capacity).

### From Moka (`~/repos/moka/src/common/frequency_sketch.rs`)

**4-bit CountMinSketch for TinyLFU admission**:
```rust
table: Box<[u64]>,  // each u64 = 16 × 4-bit counters
```
- 4 hash functions (depth=4), counters cap at 15
- Aging: right-shift all u64s by 1, masks off top bit → halves all counters
- Aging trigger: when total increments reach `10 × capacity`
- Admission: `candidate.freq > victim.freq` (strictly greater)

**Batched policy application**: ops go to channel, applied when 64+ pending or 300ms.
Decouples hot-path mutation from policy state. Elegant but unnecessary for per-core
single-threaded cache (we can update policy inline).

**What to adopt**: frequency sketch design (future Opt-A admission filter).
**What to skip for now**: W-TinyLFU window/main partitioning (adds complexity, defer).

### From Seastar (`~/repos/seastar/include/seastar/core/slab.hh`)

**Pin/unpin via refcnt** — the cleanest pattern for eviction safety:
```cpp
lock_item(): refcnt++; if was 0, remove from LRU  // pinned = invisible to evictor
unlock_item(): refcnt--; if now 0, re-insert to LRU  // unpinned = evictable again
```
Slab page eviction: `ASSERT(desc.refcnt() == 0)` — can only evict unpinned pages.

**Watermark-based reclaim** (`memory.hh:361`): `min_free_pages` trigger → async
reclaimer fiber runs at higher priority than query fibers. Lesson: eviction should
never stall the read path. For us, eviction within a set is O(16) max — always inline.

**Per-core memory isolation** is absolute: cross-core alloc/dealloc has "severe penalty."
Validates our RefCell-only, no-atomics design.

**Non-threadsafe `lw_shared_ptr`**: avoids atomic refcount for per-core use. Same
reasoning as our Cell<u64> counters instead of AtomicU64.

**What to adopt**: pin/unpin pattern (CacheGuard = lock, drop = unlock), refcnt == 0
check before eviction. **What to skip**: slab multi-size classes (all our blocks are 4KB).

### From SIEVE Paper (`~/Documents/divergence2.1/sieve_cache.pdf`)

**Algorithm** (~20 lines):
```
Hit:  set visited = 1  (no list reordering)
Miss: while hand.visited == 1: clear, advance hand
      evict hand entry
      insert new entry at FIFO head, visited = 0
```
- 1 bit metadata per entry. 42% miss reduction over FIFO. Comparable to ARC.
- **Quick demotion**: one-hit-wonders evicted fast (never get visited=1 before sweep).
- **Lazy promotion**: only at eviction time, not on every hit.
- **Weakness**: no scan resistance without ghost cache / admission filter.

**For 8-way sets**: SIEVE degenerates to plain clock (circular 8-entry buffer). The FIFO
insertion ordering that makes SIEVE special is lost in such small sets. → Use simple clock
within sets; SIEVE's value is for large, globally-ordered caches.

### From CLOCK-Pro Paper (`~/Documents/divergence2.1/clock-pro.pdf`)

Three-hand mechanism: HANDhot, HANDcold, HANDtest. Adaptive `m_c` tunes cold:hot ratio.
40-47% improvement over LRU on weak-locality workloads.

**For us**: overkill for 8-way sets. File under "future Opt-B" if we ever move to a
globally-ordered cache (thousands of entries with a single eviction hand).

### From TinyLFU Paper (`~/Documents/divergence2.1/`)

**Core concept**: frequency-based admission policy, orthogonal to eviction. On miss when
cache full: compare `new_item.freq > victim.freq`. If yes → admit. If no → reject.

**Frequency estimation**: Counting Bloom Filter with k=3 hash functions, 4-bit counters.
`estimate(item) = min(counter[h1], counter[h2], counter[h3])`. ~8 bytes per capacity entry.

**Doorkeeper** (Bloom filter for scan resistance): first access → insert to Doorkeeper
only (don't touch sketch). Second access → insert to sketch. One-hit-wonders never
pollute frequency estimates. Overhead: ~1 bit per sample item. Reduces memory by ~89%
vs naive approach.

**Aging**: every W=10×capacity accesses, divide all counters by 2 (right-shift), clear
Doorkeeper. Prevents stale frequencies from blocking newly popular items.

**W-TinyLFU**: Window (1% LRU) + Main (99% SLRU with TinyLFU admission). New items
enter Window first, get chance to build frequency before competing.

**Performance**: Zipf 0.9: TinyLFU ~42% hit vs LRU ~15%. Beats ARC/LIRS on all traces.
Memory: ~8-10 bytes/entry total (vs ARC 16 bytes, WLFU 99 bytes).

**What to adopt (future Opt-A)**: frequency sketch + doorkeeper as admission gate on
miss path. ~150 lines Rust. Most impactful upgrade for multi-tenant/scan workloads.
**What to skip**: W-TinyLFU window/main partitioning (diminishing returns for fixed 4KB).

### From VectorDB Paper (`~/Documents/divergence1.1/p4710-do.pdf`)

- HNSW Layer 0 (top): 57% cache hit. Layers 3+ (leaves): 99.98% hit.
- Different layers need different allocation — confirms split pools.
- Query reordering improves cache hit rate (batch queries touching same subgraph).

### Synthesis: 6 Things to Get Right

1. **Single-flight dedup**: one IO per block, others await (RocksDB + universal principle)
2. **Clock eviction with effort cap**: don't burn CPU on pinned entries (RocksDB)
3. **Pin/unpin via CacheGuard**: refcnt > 0 → invisible to evictor (Seastar slab)
4. **Per-core, no atomics**: RefCell + Cell only (Seastar memory model)
5. **Pre-allocated everything**: slots, flight buffers, entry metadata (Contract #1)
6. **Stats**: hit/miss/dedup/eviction/pinned_usage (RocksDB charge model)

**Defer**: TinyLFU admission (Opt-A), CLOCK-Pro/SIEVE global (Opt-B), dynamic pool
resizing (Opt-C), background eviction (Opt-D).

---

## Design

### 6 Hard Contracts

1. **No hot-path allocation** — all buffers pre-allocated at pool creation. No `VecDeque`, no `Vec::push` on the hot path. Waiter is `Option<Waker>` (single slot, first-writer-wins — don't overwrite existing).
2. **Per-core** — RefCell, not Mutex (monoio is single-threaded)
3. **State machine** — EMPTY → LOADING → READY (Cell/RefCell since single-threaded). Each LOADING carries a `load_gen` epoch for stale waiter detection.
4. **In-flight dedup** — LOADING entry: other coroutines await same completion via `WaitForReady(slot_idx, vid, load_gen)`. WaitForReady returns `()` — gen mismatch is control flow (re-probe), not error.
5. **Cancel safety** — `LoadGuard` rolls back LOADING → EMPTY on drop (future cancel, IO error, timeout). No hung LOADING entries.
6. **CacheGuard lifetime** — must not escape the expansion scope. Decode neighbors from `guard.data()`, then drop immediately. Holding guards pins slots and starves eviction.

### Data Structures

**Set-associative cache (8-way)**. Hash vid → set. Each set has 8 slots.
`num_sets` rounded to power of 2 for Fibonacci hashing.

```rust
const SET_WAYS: u32 = 8;

pub struct AdjacencyPool {
    state: RefCell<PoolState>,          // borrowed momentarily, NEVER across await
    slot_store: SlotStore,              // one contiguous 4KB-aligned allocation (also IO buffers)
    num_sets: u32,
    stats: CacheStats,                  // Cell<u64> counters
}

struct PoolState {
    entries: Vec<Entry>,                // length = num_sets × 8
    // No global clock_hand — each eviction sweeps from way 0.
    // For 8-way sets the sweep is at most 16 iterations, always inline.
}

struct Entry {
    vid: u32,                           // u32::MAX = empty
    state: SlotState,                   // Empty | Loading | Ready
    load_gen: u32,                      // incremented each LOADING transition
    referenced: bool,                   // clock second-chance bit
    pin_count: u32,                     // live CacheGuards
    waiter: Option<Waker>,             // single waker slot, first-writer-wins
}

#[derive(PartialEq)]
enum SlotState { Empty, Loading, Ready }

/// 4KB-aligned pointer to a slot buffer. Only created by SlotStore.
/// Prevents raw pointer misuse — IoDriver only accepts SlotPtr, not *mut u8.
pub(crate) struct SlotPtr(NonNull<u8>);

struct SlotStore {                      // one big alloc: capacity × 4096 bytes
    ptr: NonNull<u8>,
    capacity: u32,
    layout: Layout,
}

pub struct CacheGuard<'a> {             // RAII: decrements pin_count on drop
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    bypass_buf: Option<AlignedBuf>,     // Some = bypass mode (all ways pinned)
}

pub struct CacheStats {
    pub hits: Cell<u64>,
    pub misses: Cell<u64>,
    pub dedup_hits: Cell<u64>,
    pub evictions: Cell<u64>,
    pub evict_fail_all_pinned: Cell<u64>,  // all 8 ways pinned/loading
    pub bypasses: Cell<u64>,               // direct reads bypassing cache
}
```

### State Machine

```
EMPTY ──allocate──→ LOADING ──IO done──→ READY
  ↑                    │                   │
  │                    │ IO fail           │ clock evict (pin_count==0, referenced==false)
  └────────────────────┴───────────────────┘
```

- LOADING: not evictable, not duplicatable (dedup point)
- READY + pin_count > 0: not evictable (Seastar pattern)
- READY + referenced: second chance — clear bit, skip (clock)
- READY + !referenced + pin_count == 0: evict

### API

```rust
impl AdjacencyPool {
    pub fn new(capacity_bytes: usize) -> Self;
    pub async fn get_or_load(&self, vid: u32, io: &IoDriver) -> io::Result<CacheGuard<'_>>;
    pub fn is_resident(&self, vid: u32) -> bool;
    pub fn stats(&self) -> CacheStatsSnapshot;  // hit, miss, dedup, eviction, evict_fail, bypasses
}
```

### get_or_load Algorithm

RefCell borrows are always momentary — dropped before any `.await`.

**Step 1 — Probe** (sync, no await):
```
borrow state → scan 8 entries in set for vid
  READY:   set referenced=true, pin_count++, inc hits → return CacheGuard
  LOADING: capture (slot_idx, vid, load_gen), inc dedup_hits
           → drop borrow → await WaitForReady(slot_idx, vid, load_gen)
           → always re-probe (WaitForReady returns () — mismatch is control flow)
  not found: → Step 2
```

**Step 2 — Miss** (sync borrow + async IO):
```
find_or_evict within 8-way set → get slot_idx
  None (all pinned) → BYPASS: do io.read_adj_block(vid), return CacheGuard with bypass_buf
                       (inc bypasses counter, no slot pinned, no cache pollution)
  Some(slot_idx) → continue below

set entry = { vid, Loading, load_gen++, referenced=true, pin_count=0 }
drop borrow

let slot_ptr = slot_store.slot_ptr(slot_idx);  // SlotPtr (typed, not raw)
let _guard = LoadGuard { pool, slot_idx, vid, load_gen };  // rollback on drop

io.read_adj_block_direct(vid, slot_ptr).await  ← yield point

// IO succeeded — disarm guard, transition to Ready
_guard.disarm();
borrow state → entry.state = Ready, pin_count = 1, wake waiter
drop borrow → return CacheGuard
```

**Error path**: IO fails → LoadGuard drop fires → LOADING → EMPTY, wake waiter, return Err.
**Cancel path**: future dropped → LoadGuard drop fires → same rollback. No hung LOADING.
**Bypass path**: all ways pinned → direct uncached read, no error returned to caller.

### Clock Eviction Within Set (effort-capped)

```rust
fn find_or_evict(&self, vid: u32) -> Option<u32> {
    // Pass 1: any Empty slot? → return immediately
    // Pass 2: clock sweep from way 0, max 2 × SET_WAYS iterations (effort cap)
    //   skip Loading (IO in flight)
    //   skip pin_count > 0 (Seastar rule)
    //   referenced → clear bit, advance (second chance)
    //   !referenced → evict (reset entry, inc stats.evictions) → return slot
    // None → inc stats.evict_fail_all_pinned → return None (caller bypasses cache)
}
```

No global clock_hand — each eviction sweeps from way 0. For 8-way sets, the full
sweep is 16 iterations max, always inline.

`evict_fail_all_pinned` tracks pressure. If it climbs, cache is too small or query
concurrency too high. Caller **bypasses** cache with a direct uncached read instead
of returning an error — never fail-hard on eviction pressure.

### WaitForReady Future

Custom Future modeled after existing `SemAcquire` in `io.rs:78-98`.
Matches on `(vid, load_gen)` — not just `vid` — to avoid stale waiter matching
after IO failure + re-LOADING of the same vid or slot reuse.

Returns `()` always — gen mismatch / IO failure is **control flow** (caller re-probes),
not an error metric.

```rust
struct WaitForReady<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,          // exact slot, not vid search
    vid: u32,               // for validation
    load_gen: u32,          // epoch captured at registration time
}

impl Future for WaitForReady<'_> {
    type Output = ();
    fn poll(self, cx) -> Poll<()> {
        borrow state → check entry at slot_idx
          vid matches AND load_gen matches:
            Ready → Poll::Ready(())
            Loading → if waiter.is_none() { register }, Poll::Pending
          vid/gen mismatch OR Empty → Poll::Ready(())  // flight gone, re-probe
    }
}
```

First-writer-wins waker: only registers if `waiter.is_none()`, reducing unnecessary
waker clones. Per-core single-threaded means at most one coroutine polls at a time.
Multiple waiters cycle: wake one → it re-probes → next waiter registers on next poll.

Caller loops: after WaitForReady completes, always re-probe the set to get CacheGuard.

### IoDriver Addition

New method — reads directly into a `SlotPtr` (typed wrapper, not raw `*mut u8`):

```rust
// crates/engine/src/io.rs
pub(crate) async fn read_adj_block_direct(&self, vid: u32, dst: SlotPtr) -> io::Result<()>
```

`SlotPtr` is only constructable by `SlotStore`, guaranteeing 4KB alignment and
sufficient capacity. Prevents raw pointer misuse — external callers cannot pass
arbitrary pointers. The slot is exclusively owned by the LOADING entry (pin_count=0,
no CacheGuard outstanding), so no aliasing.

For io_uring registered buffers (future optimization): SlotStore allocation can be
registered as a fixed buffer set, giving kernel-side zero-copy.

### LoadGuard — Drop Safety for Cancel/Error

```rust
struct LoadGuard<'a> {
    pool: &'a AdjacencyPool,
    slot_idx: u32,
    vid: u32,
    gen: u32,
    disarmed: bool,
}

impl Drop for LoadGuard<'_> {
    fn drop(&mut self) {
        if self.disarmed { return; }
        // Rollback: LOADING → EMPTY, wake waiter so they re-probe
        let mut state = self.pool.state.borrow_mut();
        let entry = &mut state.entries[self.slot_idx as usize];
        if entry.vid == self.vid && entry.load_gen == self.gen
           && entry.state == SlotState::Loading {
            entry.state = SlotState::Empty;
            entry.vid = u32::MAX;
            if let Some(w) = entry.waiter.take() { w.wake(); }
        }
    }
}
```

Covers three cases:
1. **IO error** → `get_or_load` returns Err, guard drops, rollback fires
2. **Future cancelled** (timeout, abort, joinset drop) → guard drops, rollback fires
3. **IO success** → `guard.disarm()` called, drop is a no-op

This replaces the old "monoio is cooperative so cancellation can't happen" assumption,
which was wrong — Rust futures can be dropped at any `.await` point regardless of
the executor's cooperative scheduling.

---

## Files to Modify

| File | Action | ~Lines |
|------|--------|--------|
| `crates/engine/src/cache.rs` | **New**: AdjacencyPool, SlotStore, CacheGuard, LoadGuard, WaitForReady, tests | ~480 |
| `crates/engine/src/io.rs` | Add `read_adj_block_direct` method (raw ptr variant) | ~20 |
| `crates/engine/src/search.rs` | Add `pool` param, replace `io.read_adj_block` with `pool.get_or_load` | ~10 |
| `crates/engine/src/lib.rs` | Add `pub mod cache`, re-export | ~3 |
| `crates/engine/tests/disk_search.rs` | Create pool in tests, pass to search | ~10 |

**~520 lines total**

## Build Order

1. `SlotStore` + `Entry` (with `load_gen`) + `SlotState` + `CacheStats` (with `evict_fail_all_pinned`) — data defs, no async
2. `AdjacencyPool::new` — constructor, pre-allocation (no flight_pool)
3. `find_or_evict` — clock eviction (sync, testable without IO)
4. `probe_set` → `ProbeResult` enum — lookup logic (sync)
5. `CacheGuard` — RAII drop decrementing pin_count
6. `LoadGuard` — RAII drop rolling back LOADING → EMPTY + wake waiter
7. `IoDriver::read_adj_block_direct` — reads into raw `*mut u8` slot pointer
8. `WaitForReady` — custom Future matching `(slot_idx, vid, load_gen)`, single `Option<Waker>`
9. `get_or_load` — async fn: probe → (hit | dedup-wait | miss+LoadGuard+IO)
10. Unit tests: eviction, dedup, pin safety, stats, effort cap, cancel safety (drop LoadGuard)
11. Integration: update `search.rs` + `disk_search.rs` test

## Verification

1. `cargo test -p divergence-engine` — unit tests: eviction, dedup, pin, stats
2. `cargo test -p divergence-engine --test disk_search` — integration: disk search still matches in-memory
3. `cargo check --workspace` — no warnings
4. Run same query twice → second run shows higher hit count, lower miss count

## Future Opts (not in this PR)

- **Opt-A**: TinyLFU admission filter (4-bit CountMinSketch + Doorkeeper, ~200 lines). Bolt onto get_or_load miss path. Prevents scan pollution. Most impactful next upgrade.
- **Opt-B**: Global SIEVE/CLOCK-Pro eviction for large caches (thousands of slots). Replace set-associative.
- **Opt-C**: Dynamic pool resizing under memory pressure (Seastar watermark model).
- **Opt-D**: Priority countdown (RocksDB model): adjacency=3, vector=1 initial score.
