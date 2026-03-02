# Cache Research Distillation for AdjacencyPool

Distilled from: RocksDB source (`~/repos/rocksdb/`), Moka source (`~/repos/moka/`),
Seastar source (`~/repos/seastar/`), SIEVE paper, CLOCK-Pro paper, TinyLFU paper,
VectorDB papers, RocksDB wiki + HyperClockCache blog, Caffeine W-TinyLFU wiki,
ScyllaDB cache blog.

---

## 1. RocksDB HyperClockCache

**Source**: `~/repos/rocksdb/cache/clock_cache.h`, `clock_cache.cc`

### 64-bit packed metadata per slot (the core innovation)

```
Bits [0-29]:   AcquireCounter (30 bits)  — incremented on Lookup
Bits [30-59]:  ReleaseCounter (30 bits)  — incremented on Release
Bit 60:        HitFlag
Bit 61:        OccupiedFlag
Bit 62:        ShareableFlag (ref-counted, visible or invisible)
Bit 63:        VisibleFlag (findable by Lookup)
```

RefCount = AcquireCounter − ReleaseCounter. Lookup is a single `fetch_add` on acquire
(not CAS). Release is a single `fetch_add` on release. This eliminates CAS contention
on the hot path entirely.

### Four-state machine

- **Empty**: OccupiedFlag=0. Free slot.
- **UnderConstruction**: Occupied=1, Shareable=0, Visible=0. Thread has exclusive ownership.
- **Visible**: Occupied=1, Shareable=1, Visible=1. Normal live entry. Findable by Lookup.
- **Invisible**: Occupied=1, Shareable=1, Visible=0. Erased but refs still outstanding.

Transitions:
- Empty → UnderConstruction: atomic OR to set Occupied (optimistic takeover)
- UnderConstruction → Visible: set Shareable+Visible after data is written
- Visible → Invisible: atomic AND to clear Visible (on Erase)
- Invisible → UnderConstruction: CAS, only if refs==0 (reclaim slot)
- UnderConstruction → Empty: clear Occupied (abandon or cleanup)

Only the Invisible→UnderConstruction transition requires CAS (must verify zero refs).
All others are single atomics.

### Eviction effort cap (`clock_cache.cc:568`)

```cpp
if ((freed_count + 1) * effort_cap <= seen_pinned_count) {
    // abort eviction — too many pinned entries, wasting CPU
}
```
Default `effort_cap = 30`. Prevents unbounded scanning when cache is heavily pinned.

### Priority countdown

Initial clock score by priority: HIGH=3, MID=2, LOW=1. Clock decrements score on sweep.
Evict when score reaches 0. More nuanced than single second-chance bit — high-priority
entries survive 3 sweeps, low-priority survive 1.

### Open addressing with displacement tracking

Flat pre-allocated array. Double hashing probe. Each slot has `displacements` counter
tracking how many entries hash at-or-below this slot but reside at-or-above. Enables
deletion without tombstones. Accepts transient duplicates (rare in practice).

### Charge accounting

- `usage_`: total bytes of all entries
- `standalone_usage_`: entries not in hash table (memory reservations)
- `GetPinnedUsage()`: entries with refcount > 0
- `GetOccupancyCount()`: number of entries in table
- Per-entry: `total_charge = object_charge + metadata_overhead`

### Insert flow

1. Optimistic `occupancy_.FetchAdd(1)`
2. `ChargeUsageMaybeEvict()` — may trigger evictions to make room
3. If strict capacity and usage exceeds cap → revert, fail with MemoryLimit
4. `DoInsert()` → `FindSlot()` → CAS Empty→UnderConstruction → fill → Visible
5. If table full → create standalone entry on heap (not in table)

### Secondary Cache (two-tier)

Primary eviction → optionally demote to secondary (compressed/persistent).
Async lookup: `StartAsyncLookup()` → `WaitAll()` → `Result()`.
"Placeholder" entries (zero charge) act as dedup tokens: if another thread needs the
same block, it finds the placeholder and waits instead of issuing duplicate IO.

### Lessons for Divergence

- **Adopt**: effort-capped eviction, charge tracking (usage + pinned_usage), priority
  countdown model (adjacency=3 vs vector=1 in future)
- **Adopt**: visibility state machine semantics (UnderConstruction/Visible/Invisible) —
  the correctness value is the state separation, not the atomics. Our EMPTY/LOADING/READY
  maps to Empty/UnderConstruction/Visible. Without this discipline: "write data then set
  ready" races, or "evict a slot with outstanding references."
- **Skip**: split acquire/release counters (we're single-threaded, no atomic contention),
  dynamic resizing, displacement-tracked open addressing (we use set-associative)

---

## 2. Moka (Rust Cache Library)

**Source**: `~/repos/moka/src/common/frequency_sketch.rs`, `src/sync/base_cache.rs`

### 4-bit CountMinSketch for TinyLFU

```rust
pub struct FrequencySketch {
    sample_size: u32,        // capacity × 10 (window for aging)
    table_mask: u64,
    table: Box<[u64]>,       // each u64 = 16 × 4-bit counters
    size: u32,               // current increment count
}
```

- 4 hash functions (depth=4): each item hashes to 4 positions
- Counter range: 0–15 (4 bits, capped)
- Frequency estimate = min of 4 counters
- Table width = `capacity.next_power_of_two()`
- Memory: 8 bytes per capacity entry (one u64 per 16 counters)

### Aging (reset)

```rust
fn reset(&mut self) {
    for entry in self.table.iter_mut() {
        count += (*entry & ONE_MASK).count_ones();  // count odd counters
        *entry = (*entry >> 1) & RESET_MASK;        // halve all counters
    }
    self.size = (self.size >> 1) - (count >> 2);
}
```

- Triggered when `self.size >= self.sample_size` (sample_size = capacity × 10)
- Right-shift all u64s by 1 → halves all 4-bit counters
- Amortized O(1) per operation (reset is O(table_size) but happens every 10× capacity ops)
- `ONE_MASK = 0x1111_1111_1111_1111` (bit 0 of each 4-bit counter)
- `RESET_MASK = 0x7777_7777_7777_7777` (top 3 bits of each 4-bit counter)

### Admission decision

```rust
if victims.policy_weight >= candidate.policy_weight && candidate.freq > victims.freq {
    AdmissionResult::Admitted { victim_keys }
} else {
    AdmissionResult::Rejected
}
```

Candidate must have strictly greater frequency than aggregated victims. Max 5 retry
attempts before giving up.

### Batched policy application

Read/write ops go to bounded channels (size 384). Applied in batch when:
- Channel has 64+ items, OR
- 300ms elapsed since last application

Decouples hot-path from policy mutation. Elegant for multi-threaded cache, unnecessary
for our single-threaded per-core pool.

### MiniArc (lightweight refcounting)

```rust
struct ArcData<T> {
    ref_count: AtomicU32,   // 4 bytes instead of 8 (AtomicUsize)
    data: T,
}
```

For per-core single-threaded use, we don't even need atomics — plain Cell<u32> suffices.

### Lessons for Divergence

- **Adopt (future Opt-A)**: frequency sketch design — 4-bit counters, 4 hash functions,
  right-shift aging. Implementation is ~100 lines of Rust.
- **Adopt**: admission logic pattern (candidate.freq > victim.freq)
- **Skip**: batched channel application (we're single-threaded), W-TinyLFU window/main
  partitioning (adds complexity, diminishing returns for our fixed-size 4KB blocks)

---

## 3. Seastar Framework

**Source**: `~/repos/seastar/include/seastar/core/slab.hh`, `memory.hh`, `shared_ptr.hh`

### Per-core memory isolation (absolute)

Each lcore gets its own memory pool. Cross-core alloc/dealloc has "severe performance
penalty." Message passing between cores is explicit. No shared memory structures.

Validates: our RefCell-only design. No Mutex, no atomics for pool state.

### Slab allocator with LRU and pin/unpin

```cpp
void lock_item(Item* item) {
    auto& desc = get_slab_page_desc(item);
    auto& refcnt = desc.refcnt();
    if (++refcnt == 1) {
        // Remove page from LRU — pinned pages are invisible to evictor
        _slab_page_desc_lru.erase(iterator_to(desc));
    }
}

void unlock_item(Item* item) {
    auto& desc = get_slab_page_desc(item);
    auto& refcnt = desc.refcnt();
    if (--refcnt == 0) {
        // Re-insert to LRU — page is now evictable
        _slab_page_desc_lru.push_front(desc);
    }
}
```

Eviction: `ASSERT(desc.refcnt() == 0)` — only evict unpinned pages.

This is exactly the pattern for our CacheGuard: pin on get_or_load, unpin on drop.
Eviction skips entries with pin_count > 0.

### Watermark-based reclaim

```cpp
size_t min_free_memory();  // when free drops below this → reclaim
void set_min_free_pages(size_t pages);
```

Reclaimer runs as async fiber at higher priority than query fibers. Prevents eviction
from stalling reads. For us: eviction within an 8-way set is O(16) max — always inline,
no background fiber needed.

### Non-threadsafe smart pointers

`lw_shared_ptr<T>`: non-atomic refcount. 1 machine word. Optimized for per-core use.
Validates our Cell<u64>/Cell<u32> approach.

### IO queue with token bucket

Fair queue with pending token reservations for IO backpressure. Our LocalSemaphore
serves the same purpose (bounds inflight IO).

### Lessons for Divergence

- **Adopt**: pin/unpin pattern (CacheGuard = lock, Drop = unlock, eviction checks
  refcnt == 0)
- **Adopt**: per-core isolation as fundamental design axiom
- **Adopt**: watermark concept for "when should we evict proactively" (future opt)
- **Skip**: slab multi-size classes (all our blocks are fixed 4KB), cross-cpu freelist

---

## 4. SIEVE Eviction Algorithm

**Source**: `~/Documents/divergence2.1/sieve_cache.pdf` (NSDI 2024)

### Algorithm (~20 lines)

```
On cache hit:
    object.visited = 1          // no list reordering

On cache miss:
    if cache is full:
        while hand.visited == 1:
            hand.visited = 0    // clear second-chance
            hand = hand.next    // advance
        evict entry at hand
        hand = hand.next
    insert new object at HEAD of FIFO, visited = 0
```

### Key properties

- **Metadata**: 1 bit per entry (visited flag) + FIFO linked list
- **Quick demotion**: one-hit-wonders never get visited=1 before sweep → evicted fast
- **Lazy promotion**: objects promoted only at eviction time, not on every hit
- **No list reordering on hit**: just set a bit. LRU must move-to-front on every hit.

### Performance

- SIEVE vs FIFO: 42% miss ratio reduction (1559 traces, 7 datasets)
- SIEVE vs ARC: 3.7% lower miss ratio on average (mean, not median)
- SIEVE vs LRU: better on 45%+ of traces
- Throughput at 16 threads: 2× optimized LRU (no list manipulation contention)

### Critical difference from CLOCK

In CLOCK, retained objects stay in their circular position. In SIEVE, retained objects
stay at their original FIFO insertion position. New objects enter at head. This naturally
separates generations: old popular objects cluster near tail, new objects at head.

### Weakness

No scan resistance without ghost cache / admission filter. A large sequential scan can
pollute the cache. Mitigate with TinyLFU admission (future Opt-A).

### For 8-way sets

SIEVE degenerates to plain clock for small sets. The FIFO ordering that makes SIEVE
special is lost in 8-entry sets. → Use simple clock within sets for now. SIEVE's value
is for large globally-ordered caches (thousands of entries with a single eviction hand).

**Why set-associative over global open-addressing + single hand?**
Set-associative: probe bounded by ways (8), eviction O(2×ways) worst case, implementation
simple, no deletion/tombstone complexity. Global clock: potentially higher hit rate (no
conflict misses) but eviction scan unbounded without effort cap, and deletion requires
displacement tracking (RocksDB approach) or tombstones. We choose set-associative to
bound probe + eviction cost. Global SIEVE/CLOCK is a future option if conflict misses
(tracked via `evict_fail_all_pinned` + per-set conflict stats) prove to dominate.

### Lessons for Divergence

- **Adopt concept**: visited bit + sweep = minimal metadata eviction
- **Skip full SIEVE**: for set-associative (8-way), plain clock is equivalent
- **Future**: if we move to global eviction (thousands of slots), use SIEVE over CLOCK

---

## 5. CLOCK-Pro

**Source**: `~/Documents/divergence2.1/clock-pro.pdf` (USENIX 2005)

### Three-hand mechanism

- **HANDhot**: points to hot page with largest recency, terminates hot test
- **HANDcold**: searches for cold page to evict
- **HANDtest**: identifies test period duration for non-resident cold pages

### Adaptive cold page allocation

Dynamic `m_c` parameter adjusts cold page allocation based on reuse patterns.
If a non-resident cold page is accessed again (re-enters cache), `m_c` increases
(allocate more space for cold pages). If hot page demoted to cold, `m_c` may decrease.

### Scan resistance

Non-resident cold pages tracked in a "test" ring. If they're accessed again during the
test period, they're promoted to hot. Pages accessed only once never make it past cold.

### Performance

40–47% page fault reduction on weak-locality workloads vs CLOCK. On real programs,
up to 47% execution time reduction.

### Lessons for Divergence

- **Skip for now**: three-hand design is complex for 8-way sets
- **Future Opt-B**: if we move to global cache with thousands of entries, CLOCK-Pro
  gives adaptive hot/cold tuning that plain clock or SIEVE lack

---

## 6. TinyLFU

**Source**: `~/Documents/divergence2.1/` TinyLFU paper

### Core concept

Frequency-based admission policy, orthogonal to eviction policy. On cache miss when
cache is full:
1. Eviction policy selects a victim
2. TinyLFU compares: `new_item.freq > victim.freq` ?
3. If yes → admit new, evict victim. If no → reject new, keep victim.

### Frequency estimation: Counting Bloom Filter

- k=3 hash functions per item
- 4-bit counters (range 0–15)
- `estimate(item) = min(counter[h1(item)], counter[h2(item)], counter[h3(item)])`
- Memory: ~8 bytes per cache-capacity entry

### Doorkeeper (Bloom filter for scan resistance)

- Standard Bloom filter in front of frequency sketch
- First access to item → insert into Doorkeeper only (don't touch sketch)
- Second access → insert into sketch, increment counters
- Effect: one-hit-wonders never pollute the frequency sketch
- Cleared on every aging reset
- Overhead: ~1 bit per sample item
- Reduces memory by ~89% vs naive approach

### Aging / reset

- Trigger: every W accesses, where W = 10 × cache_capacity
- Operation: divide all counters by 2 (right-shift), clear Doorkeeper
- Amortized O(1) per access
- Prevents stale frequency counts from blocking newly popular items

### W-TinyLFU (Window variant)

- Window Cache (1% of total): simple LRU, admits everything
- Main Cache (99% of total): SLRU eviction, TinyLFU-gated admission
- New items enter Window first, get chance to build frequency
- Window victim competes with Main victim via TinyLFU

### Performance

- Zipf 0.9: TinyLFU ~42% hit vs LRU ~15% (3× improvement)
- Wikipedia: 5+ point improvement over LRU, matches or beats ARC/LIRS
- OLTP (DS1): W-TinyLFU ~60% vs ARC ~50%
- YouTube (dynamic): TinyLFU adapts to distribution shifts, LRU lags
- On ALL tested traces: W-TinyLFU ≥ ARC ≥ LRU

### Memory overhead

| Component | Overhead |
|-----------|----------|
| Frequency sketch | ~0.57 bytes/sample |
| Doorkeeper | ~0.125 bytes/sample (1 bit) |
| W-TinyLFU Window pointers | 8 bytes/cache entry |
| **Total** | **~8–10 bytes per cache entry** |

vs WLFU: 99 bytes/entry. vs ARC: 16 bytes/entry.

### Lessons for Divergence

- **Future Opt-A**: bolt TinyLFU admission onto miss path. On miss:
  1. Sketch.estimate(new_vid) vs Sketch.estimate(victim_vid)
  2. If new > victim → admit. Else → reject (don't cache this block).
  3. Doorkeeper prevents one-hit polluters from even entering sketch.
- **Implementation**: ~150 lines Rust (frequency sketch + doorkeeper + admission check)
- **When to add**: when profiling shows scan/cold-block pollution in the cache
  (multi-tenant workloads, broad initial searches, or sequential scans)

---

## 7. ScyllaDB Internal Cache

**Source**: ScyllaDB blog (2024)

### Per-core isolation eliminates all synchronization

Every CPU core owns its own cache and memtable. No locks, no atomics, no lock-free
algorithms needed. Simple data structures win because concurrency is eliminated at
architecture level.

### Object cache, not buffer cache

Caches deserialized row objects, not raw pages. No redundant parsing on hit.
For us: caching raw 4KB blocks is simpler (fixed size, no serialization on eviction).

### Dynamic memory management

Cache consumes all available memory, shrinks on demand when other subsystems need space.
Controllers prevent over-stealing. More sophisticated than static 70/30 split.

### Bypass Linux page cache entirely

Direct IO to block devices via Seastar. Reasons: page cache causes read amplification
(4KB minimum), sync blocking, context switches, double-caching. Validates our O_DIRECT
approach.

### Cache bypass hint

Large scans explicitly bypass cache to prevent pollution. For us: could add a bypass
flag on broad exploratory search phases.

### Lessons for Divergence

- **Validates**: per-core isolation, O_DIRECT, record-level (block-level) caching
- **Future**: dynamic pool sizing under pressure, cache bypass hints for scans

---

## 8. VectorDB Papers

### "Turbocharging Vector Databases" (VLDB 2025)

- HNSW Layer 0 (top): 57.49% cache hit ratio with 50% buffer
- Layer 3+ (leaves): 99.98% hit — temporal locality is extremely strong
- Different layers need different allocation policies
- Confirms split buffer pool (adjacency 70% / vectors 30%) is sound
- Query reordering can improve cache utilization significantly

---

## Synthesis: Priority-Ordered Lessons

### Must have (this PR)

1. **Single-flight dedup**: LOADING state + bounded, allocation-free waiter (single
   `Option<Waker>` slot, NOT an unbounded `VecDeque<Waker>`)
2. **Pin/unpin via CacheGuard**: pin_count > 0 → invisible to evictor (Seastar)
3. **Effort-capped clock eviction**: max iterations before giving up (RocksDB).
   `evict_scan_budget` is a hard cap — exceed it → fail-fast, return miss.
4. **Per-core, no atomics**: RefCell + Cell only (Seastar, ScyllaDB)
5. **Pre-allocated everything**: slots, metadata, no hot-path allocation (Contract #1)
6. **Counters**: hit/miss/dedup/eviction/evict_fail_all_pinned (RocksDB charge model)

### Should have (next PR)

7. **TinyLFU admission**: frequency sketch + doorkeeper (~150 lines). Prevents
   scan/cold-block pollution. Most impactful upgrade for multi-tenant workloads.

### Nice to have (future)

8. **Priority countdown**: adjacency=3, vector=1 initial score (RocksDB model)
9. **Global SIEVE eviction**: for large caches, replace set-associative (SIEVE paper)
10. **Dynamic pool sizing**: watermark-based pressure response (Seastar model)
11. **Cache bypass hints**: skip caching for broad scans (ScyllaDB pattern)
12. **CLOCK-Pro adaptive**: three-hand hot/cold tuning for shifting workloads
