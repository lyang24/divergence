# Prefetch Design for Divergence

Distilled from: PipeANN (OSDI'25, Guo & Lu), VeloANN (2026), glommio,
io_uring fixed-buffer experiments, and Divergence's current beam search.

---

## 1. The Problem PipeANN Solved (and Why It Matters to Us)

### Best-first search on SSD has two mismatches

**Issue 1: Ordered compute-I/O dependency.** Each search step batch-reads W
neighbors, waits for ALL W I/Os to finish, computes distances, then decides
the next W. Compute sits idle during I/O tail latency. Even at W=8, compute
is only 45% of I/O latency (PipeANN Fig 3a) — the overlap opportunity is
wasted.

**Issue 2: Synchronous I/O underutilizes the pipeline.** W=8 achieves 76%
I/O utilization, W=32 drops to 58% (Fig 3b). Waiting for a batch means the
slowest I/O in the batch determines when the next batch starts.

### PipeANN's key insight: pseudo-dependency of compute and I/O

In a graph index, neighbors can be decided from the *in-memory candidate
pool* alone — no need to wait for all ongoing I/Os. When the I/O pipeline
has free slots, immediately issue reads for the nearest unread candidate.
Overlapped with I/O, explore completed reads one at a time.

Result: 39-49% latency reduction vs DiskANN/Starling at 0.9 recall.
1.14-2.02x of in-memory Vamana latency. 1.35x throughput at 0.9 recall.

### What PipeANN does NOT have

- **No persistent cache.** Every query starts cold. Every expansion is a miss.
- **No singleflight dedup.** Single-query-per-thread model, no shared blocks.

Divergence has both. Our prefetch design must compose with AdjacencyPool.

---

## 2. PipeANN Algorithm (from paper + source)

### PipeSearch (Algorithm 2 in paper)

```
W ← 4  (starting pipeline width)
P ← InMemSearch(q, min(L, L_mem))    // approach phase: in-memory entry point
U ← ∅, Q ← ∅                         // U = unexplored set, Q = in-flight IOs

while P ⊄ E:                          // converge phase
    if Q.size() < W then              // pipeline not full
        V ← top-1 nearest to q in P, not in E
        Q.insert(V)                   // send read request (io_uring prep_read)

    v ← nearest vector in U           // explore one completed read
    E.insert(v), U.remove(v)
    for nbr in v.neighbors:
        dis ← PQ_distance(nbr, q)
        P.insert(<nbr, dis>)

    F ← finished I/Os in Q            // poll for completion (peek_batch_cqe)
    W ← AdaptPipelineWidth(P, F)
    U.insert(F), Q.remove(F)
```

### Key mechanics

1. **Issue-one-explore-one interleaving** (lines 11-14). The pipeline
   maintains up to W outstanding reads, but when multiple I/Os complete
   simultaneously, PipeANN processes them one at a time: issue one new I/O,
   explore one completed node, repeat. This prevents "neighbor accumulation"
   — each new I/O decision benefits from the neighbor info of the prior
   completed I/O. Important: "one at a time" refers to the processing
   discipline, NOT limiting outstanding reads to 1. The NVMe queue stays
   at depth W for throughput; the interleaving avoids stale decisions.

2. **Dynamic pipeline width** (§4.2):
   - Two phases: approach (high I/O waste, W=4 fixed) and converge (low waste, W grows).
   - Transition signal: `n_v` = estimated number of already-recalled vectors. When
     `n_v ≥ 5`, switch to converge and start increasing W.
   - `n_v` approximation: after exploring a vector, iterate the candidate pool to find
     the first vector whose read hasn't been issued. Its index is an upper bound on `n_v`.
   - Dynamic adjustment: when `(IOs landing in candidate pool) / (total finished IOs) > 0.9`,
     increment W by 1.

3. **Polling-based I/O** (§4.4): Uses `peek_batch_cqe` (non-blocking CQE poll) instead
   of interrupts. The saved interrupt time is used for compute overlap.

4. **Two-phase search**: In-memory mini-graph for entry point optimization (approach),
   then on-disk PipeSearch (converge).

### PipeANN numbers to remember

| Metric | Value | Source |
|--------|-------|--------|
| Latency reduction vs DiskANN | 39% (SIFT), 49% (SPACEV) | Fig 11, 0.9 recall |
| Latency vs in-memory Vamana | 2.02x (SIFT), 1.14x (DEEP) | Fig 15, 0.9 recall |
| I/O waste at W=8 | 1.11x avg I/O vs best-first | §5.5, +Pipe row |
| Throughput vs DiskANN | 1.35x average | Fig 12, 0.9 recall |
| Throughput vs ideal (W=1) | 85-88% | Fig 18, 0.9 recall |
| Accuracy drop | <1% at recall ≥ 0.9 (≥95.9% of DiskANN) | Fig 19 |
| Pipeline draining | not significant in later steps | §4.2.1 |

---

## 3. VeloANN Model (Inter-Query Prefetch)

VeloANN takes a different approach: **inter-query** overlap via coroutines.

### Core mechanism
```
B = ceil(α × I / T)   // batch size: α=scaling, I=io_latency, T=compute_per_node
```

Each core runs B concurrent search coroutines. When one coroutine hits a cache
miss, it suspends (`.await`); the runtime schedules another coroutine that may
have its I/O completed. The "prefetch" emerges from overlapping different queries'
I/O and compute phases.

### Cache-aware beam search
When the frontier has both cached and uncached candidates, VeloANN pivots to
process cached candidates first (no I/O wait). This is essentially free prefetch:
the uncached reads submitted earlier by this or other coroutines may complete
while cached candidates are being processed.

### Relevance to Divergence
Our monoio runtime already provides this: `disk_graph_search` is an async fn,
`pool.get_or_load(vid).await` is the suspend point. Multiple queries on the
same core naturally interleave. VeloANN's B=ceil(α×I/T) gives us the formula
for tuning coroutines-per-core.

**What VeloANN doesn't give us**: intra-query prefetch. Each query's beam search
is sequential — one expansion at a time. PipeANN addresses this.

---

## 4. Prefetch State Machine for Divergence

### Design: Prefetch Worker + Submit Queue (No Barrier)

Divergence combines AdjacencyPool (which PipeANN lacks) with PipeANN-style
intra-query pipeline. The critical constraint: **the search loop must never
wait for prefetches to complete.** Prefetch is speculative — the search loop
only awaits when it actually *needs* a block (via `get_or_load`).

Three APIs:

```rust
/// Enqueue a prefetch: transition EMPTY→LOADING and push (vid, slot_idx) to
/// the per-core prefetch channel. Returns immediately (sync, no await).
/// Does NOT pin — no CacheGuard returned.
/// If vid is READY or LOADING, this is a no-op (singleflight).
/// If all slots in the set are pinned, silently drops the hint.
pub fn prefetch_hint(&self, vid: u32)

/// Kick: ensure enqueued prefetches are submitted to io_uring.
/// Light-weight: just wakes the prefetch worker if it's sleeping.
/// Does NOT await completion. Returns immediately.
pub fn kick_prefetches(&self)

/// Get or load a block (existing API, unchanged).
/// If the block was prefetched and completed, this is a cache hit.
/// If the block is still LOADING (prefetch in flight), this awaits via
/// WaitForReady (singleflight dedup — no duplicate IO).
pub async fn get_or_load(&self, vid: u32, io: &IoDriver) -> io::Result<CacheGuard<'_>>
```

### Why NOT `flush_prefetches().await`

`flush_prefetches().await` in the expansion hot loop creates a **barrier**:
every expansion waits for all W prefetches to complete before starting
`get_or_load`. This defeats the purpose of prefetch — it turns "read ahead
while I'm computing" into "read ahead, wait for all of them, then read the
one I actually need."

Concrete damage:
- If W=4 prefetches take {50μs, 55μs, 80μs, 120μs}, flush blocks for 120μs
  even though the expansion's own block might complete in 60μs.
- This is exactly PipeANN's "batch-wait" anti-pattern (Fig 4a) that they
  designed PipeSearch to eliminate.
- p99 gets worse: the flush barrier adds the **tail latency of W reads**
  to every expansion, instead of just the tail of 1 read.

### Prefetch worker: 1 long-lived task per core

Instead of the search loop awaiting prefetch IO, a single background task
consumes the prefetch queue and awaits IO completions independently.

```
Search coroutine                  Prefetch worker (1 per core)
────────────────                  ────────────────────────────
prefetch_hint(v1)  →  push to channel
prefetch_hint(v2)  →  push to channel
kick_prefetches()  →  wake worker       recv(v1): sem.acquire, read_direct, READY, wake_all
                                        recv(v2): sem.acquire, read_direct, READY, wake_all
get_or_load(curr).await ─── if curr is:
   READY (prefetch completed) → cache hit, ~7ns
   LOADING (prefetch in flight) → WaitForReady, singleflight
   MISS (not prefetched) → normal miss path
compute(neighbors)
   ← while computing, worker processes more prefetches
```

The search loop and prefetch worker run as **concurrent coroutines on the
same monoio thread** (cooperative scheduling). When the search loop yields
(at `get_or_load().await`), the worker gets CPU time to process queued
reads. When the worker yields (at `read_direct().await`), the search loop
can resume if its IO completed.

### Submit queue design

```rust
const MAX_PREFETCH_QUEUE: usize = 16;  // 2 × SET_WAYS, bounded ring buffer

struct PrefetchEntry {
    vid: u32,
    slot_idx: u32,
    load_gen: u32,
}

/// SPSC channel: search coroutine produces, prefetch worker consumes.
/// Fixed-capacity ring buffer. No heap allocation after init.
/// Both ends are on the same thread — Cell/RefCell, no atomics.
struct PrefetchChannel {
    buf: [Option<PrefetchEntry>; MAX_PREFETCH_QUEUE],
    head: u8,    // consumer reads from head
    tail: u8,    // producer writes at tail
    waker: Option<Waker>,  // worker's waker for kick
}
```

### State machine interaction with singleflight

```
                       prefetch_hint(vid)             get_or_load(vid).await
                       ─────────────────              ──────────────────────

Slot state = EMPTY     Evict victim, set LOADING,     Same as today (miss path).
                       push to channel. Return.

Slot state = LOADING   No-op (singleflight:           Await WaitForReady as today.
                       IO already in flight).          Singleflight dedup works.

Slot state = READY     No-op (already cached).         Hit path. pin_count++.
                       Set referenced=true.            Return CacheGuard.
```

**Contract**: `prefetch_hint` never pins. Only `get_or_load` pins (returns CacheGuard).
This means prefetched-but-not-yet-consumed blocks are evictable if cache pressure
hits — correct behavior, not a bug.

**Non-negotiable eviction rule**: **LOADING slots are non-evictable.** The clock
eviction scan MUST skip entries with `state == Loading`. This is already the
case in `find_or_evict` (which only evicts `Ready` + `pin_count == 0` +
`!referenced` entries), but we state it explicitly here because prefetch makes
LOADING entries more common. If LOADING were evictable, we'd need cancel safety
for in-flight IO + wake_all + generation bump — complexity explosion for no gain.

### Why this composes with AdjacencyPool

1. **Singleflight**: prefetch_hint sets LOADING. When get_or_load arrives, it sees
   LOADING → WaitForReady → gets READY when worker completes IO. One IO, two
   consumers. Existing dedup mechanism handles it.

2. **Eviction safety**: prefetch_hint doesn't pin. LOADING is non-evictable
   (IO in flight). Once READY with pin_count=0, the block is evictable by clock.
   If evicted before get_or_load arrives, get_or_load sees MISS → loads normally.
   Wasted prefetch IO, but correct.

3. **No new allocation**: PrefetchChannel is inline fixed-capacity ring buffer.
   No per-hint heap allocation ever.

4. **No barrier**: The search loop never waits for prefetch completion. It only
   waits at `get_or_load`, and only if the specific block it needs is still
   LOADING. If the prefetch completed in time → cache hit. If not → singleflight
   wait (same latency as no-prefetch case, never worse).

### Implementation sketch

```rust
impl AdjacencyPool {
    pub fn prefetch_hint(&self, vid: u32) {
        let mut state = self.state.borrow_mut();
        let base = self.set_base(vid);

        // Check if already resident or loading
        for i in 0..SET_WAYS {
            let idx = base + i;
            let entry = &state.entries[idx as usize];
            if entry.vid == vid {
                match entry.state {
                    SlotState::Ready => { /* already cached */ return; }
                    SlotState::Loading => { return; } // singleflight — IO in flight
                    SlotState::Empty => unreachable!(),
                }
            }
        }

        // Channel full? Drop hint (bounded, never blocks)
        if state.prefetch_channel.is_full() { return; }

        // Find victim slot (may fail if all pinned/loading — give up silently)
        let slot_idx = match Self::find_or_evict_inner(&mut state, &self.stats, base) {
            Some(idx) => idx,
            None => { self.stats.inc_evict_fail(); return; }
        };

        // Transition EMPTY → LOADING, enqueue for worker
        let entry = &mut state.entries[slot_idx as usize];
        entry.vid = vid;
        entry.state = SlotState::Loading;
        entry.load_gen += 1;
        entry.referenced = true;

        state.prefetch_channel.push(PrefetchEntry {
            vid,
            slot_idx,
            load_gen: entry.load_gen,
        });
        // RefCell borrow drops here
    }

    pub fn kick_prefetches(&self) {
        let state = self.state.borrow();
        if let Some(w) = state.prefetch_channel.waker.as_ref() {
            w.wake_by_ref();
        }
    }
}

/// Prefetch worker: spawned once per core at pool creation.
/// Runs forever, sleeping when channel is empty.
async fn prefetch_worker(pool: Rc<AdjacencyPool>, io: Rc<IoDriver>) {
    loop {
        // Wait for work (suspend until kick or channel non-empty)
        PrefetchWait::new(&pool).await;

        // Drain available entries (batch: process all that are queued)
        while let Some(pe) = pool.state.borrow_mut().prefetch_channel.pop() {
            let slot_ptr = pool.slot_store.slot_ptr(pe.slot_idx);
            match io.read_adj_block_direct(pe.vid, slot_ptr).await {
                Ok(()) => {
                    let mut state = pool.state.borrow_mut();
                    let entry = &mut state.entries[pe.slot_idx as usize];
                    if entry.vid == pe.vid && entry.load_gen == pe.load_gen {
                        entry.state = SlotState::Ready;
                        entry.waiters.wake_all();
                    }
                }
                Err(_) => {
                    let mut state = pool.state.borrow_mut();
                    let entry = &mut state.entries[pe.slot_idx as usize];
                    if entry.vid == pe.vid && entry.load_gen == pe.load_gen {
                        entry.reset();
                        entry.waiters.wake_all();
                    }
                }
            }
        }
    }
}
```

**Scheduling behavior**: The worker's `read_adj_block_direct().await` yields
to the monoio executor. During this yield, the search coroutine can run
(process the current expansion's compute phase, issue more prefetch_hints).
When the search coroutine yields at `get_or_load().await`, the worker can
run (complete prefetch IOs, transition LOADING→READY). This is the natural
cooperative interleaving that gives us PipeANN-style overlap without any
explicit scheduling.

**SQE batching**: The worker's reads and the search loop's get_or_load reads
share the same io_uring instance (monoio is single-ring per thread). SQEs
from both coroutines accumulate in the same submission queue. monoio submits
them together at the next `io_uring_enter()`. Batching is preserved.

**Lifetime**: Pool and IoDriver are wrapped in `Rc` (zero-cost on single
thread — just refcount). The worker task is spawned via `monoio::spawn`
once at pool creation and lives for the pool's lifetime.

---

## 5. Prefetch Window Selection Rules

### Why NOT "approach phase = expansion 1-10"

PipeANN's approach/converge split makes sense for their system: no cache,
cold start, in-memory entry-point optimization. Our EXP-0 data shows the
opposite reality:

- **Entry phase is <1%**: first top-k result enters beam at expansion 1 (median)
- **1st_tk ~ 1**: we're already in "converge" territory from expansion 2 onward
- **54% waste is in convergence**, not approach

The "expansion 1-10 = approach" concept doesn't exist on our flat NSW graph.
Window selection must use signals from the running search, not a fixed
expansion-number cutoff.

### Four signals (all cheaply available, no extra IO)

| Signal | What it tells us | How to read it | Cost |
|--------|------------------|----------------|------|
| **Inflight utilization** | Is IO pipeline saturated? | `inflight = capacity - available; util = inflight / capacity` | 2 Cell reads |
| **Frontier LOADING ratio** | Are we already waiting on IO? | Count LOADING in top-2W candidates via `pool.probe_state(vid)` | 2W probes (~16 Cell reads) |
| **Frontier stall** | Is search converging (diminishing returns)? | `consecutive_no_improve` counter: # pops that didn't improve worst top-k | 1 comparison |
| **Online waste ratio** | Are prefetches landing in useful expansions? | `prefetch_consumed / prefetch_issued` (per-query running counters) | 2 Cell reads |

### Algorithm: Signal-Driven Prefetch Window

```rust
const W_MAX: usize = 8;            // hard cap (bounded by prefetch queue)
const INFLIGHT_PRESSURE: f32 = 0.8; // >80% permits used → back off
const STALL_THRESHOLD: u32 = 3;     // 3 consecutive non-improving pops → shrink
const WASTE_THRESHOLD: f32 = 0.5;   // >50% prefetches wasted → shrink

fn select_prefetch_window(
    io: &IoDriver,
    pool: &AdjacencyPool,
    candidates: &CandidateHeap,
    consecutive_stalls: u32,
    prefetch_issued: u64,
    prefetch_consumed: u64,  // prefetch_hint'd blocks later accessed by get_or_load
) -> usize {
    // Signal 1: Pipeline pressure — don't queue into a full pipe
    let capacity = io.adj_capacity();   // total permits (e.g., 32)
    let available = io.available_adj_permits(); // free permits right now
    if available == 0 { return 0; } // pipeline full, no prefetch

    let inflight = capacity - available;
    let utilization = inflight as f32 / capacity as f32;
    let pressure_cap = if utilization > INFLIGHT_PRESSURE {
        1  // pipeline is hot — at most 1 speculative read
    } else {
        W_MAX
    };

    // Signal 2: Frontier LOADING ratio — if most top candidates
    // are already LOADING, prefetch adds nothing (dedup no-ops)
    let mut loading_count = 0;
    let check_count = (pressure_cap * 2).min(16);
    for cand in candidates.peek_top(check_count) {
        if pool.is_loading(cand.id.0) {
            loading_count += 1;
        }
    }
    let loading_cap = if loading_count > check_count / 2 {
        1  // most candidates already in flight
    } else {
        pressure_cap
    };

    // Signal 3: Frontier stall — convergence detected, shrink window
    let stall_cap = if consecutive_stalls >= STALL_THRESHOLD {
        (loading_cap / 2).max(1) // halve, but never zero
    } else {
        loading_cap
    };

    // Signal 4: Online waste ratio — too many prefetches not consumed
    let final_w = if prefetch_issued > 10 {
        let waste = 1.0 - (prefetch_consumed as f32 / prefetch_issued as f32);
        if waste > WASTE_THRESHOLD {
            (stall_cap / 2).max(1)
        } else {
            stall_cap
        }
    } else {
        stall_cap // not enough data yet, trust other signals
    };

    final_w.min(available)
}
```

### Prefetch target selection

Each expansion, BEFORE the `get_or_load` await:

```
1. w = select_prefetch_window(...)
2. Peek top-(w+1) candidates from CandidateHeap (without popping)
3. Skip [0] (that's the current candidate, about to be get_or_load'd)
4. For [1..w]: if not resident in cache → prefetch_hint(vid)
5. kick_prefetches()             — wake worker (non-blocking, ~10ns)
6. get_or_load(candidate).await  — expand current candidate
     ↑ this .await yields to executor → worker gets CPU → processes queue
```

**No barrier**: Steps 4-5 are sync (no await). The search loop only blocks
at step 6, and only for the one block it actually needs. The prefetch worker
runs concurrently during step 6's await, completing prefetch IOs. By the
time the next expansion needs those blocks, they may already be READY (hit)
or still LOADING (singleflight wait — same cost as no-prefetch).

The prefetch window controls how many *speculative candidates* we read
ahead, NOT the number of outstanding IO operations. Outstanding IO depth
is bounded by `adj_inflight` semaphore permits — the worker acquires a
permit for each read.

### Interaction with inter-query overlap

Per-core coroutine count (VeloANN model) provides inter-query overlap.
Prefetch provides intra-query overlap. They compose naturally:

- Prefetch fills the IO pipeline within a single query
- When a query's prefetches are all in flight and it needs to `.await`,
  the runtime switches to another query's ready computation
- The two mechanisms use the same semaphore budget

Total pipeline depth = `adj_inflight` permits (e.g., 32). Split between
active queries (inter) and prefetch lookahead (intra). With B=4 queries
and W=4 prefetch window: 4×(1+4) = 20 max concurrent reads, leaving 12
permits headroom for burst.

---

## 6. Revised Beam Search with Prefetch

```rust
pub async fn disk_graph_search_pipe(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    pool: &AdjacencyPool,     // has prefetch worker running
    io: &IoDriver,
    bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    // ... seed from entry_set (same as today) ...

    let mut consecutive_stalls: u32 = 0;
    let mut prev_worst_topk = f32::MAX;

    while let Some(candidate) = candidates.pop() {
        if nearest.len() >= ef && candidate.distance > nearest.furthest().unwrap().distance {
            break;
        }

        perf.expansions += 1;

        // ── Prefetch: select window + enqueue hints (all sync, no await) ──
        let w = select_prefetch_window(
            io, pool, &candidates,
            consecutive_stalls,
            perf.prefetch_issued,
            perf.prefetch_consumed,
        );
        for next_cand in candidates.peek_top(w) {
            if !pool.is_resident(next_cand.id.0) {
                pool.prefetch_hint(next_cand.id.0);
                perf.prefetch_issued += 1;
            }
        }
        pool.kick_prefetches();  // wake worker (non-blocking)

        // ── Expand current candidate ──
        // This .await is the ONLY yield point. During this yield:
        //   - prefetch worker gets CPU time to process queued reads
        //   - other queries on this core can also make progress
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => { perf.wasted_expansions += 1; continue; }
        };

        // ── Compute phase (same as today, all sync) ──
        let neighbors = decode_adj_block(guard.data());
        // ... score neighbors, update heaps ...
        drop(guard);

        // ── Update stall counter for window adaptation ──
        let current_worst = nearest.furthest().map(|s| s.distance).unwrap_or(f32::MAX);
        if current_worst < prev_worst_topk {
            consecutive_stalls = 0;
            prev_worst_topk = current_worst;
        } else {
            consecutive_stalls += 1;
        }
    }

    // ... return results ...
}
```

### Timing diagram: one expansion with prefetch

```
Search coroutine        IO ring (shared)        Prefetch worker
────────────────        ────────────────        ───────────────
hint(v1), hint(v2)
kick_prefetches()
get_or_load(curr)
  → sem.acquire ──────→ SQE: read curr
  → yield (Pending)                             wake: recv(v1)
                                                  → sem.acquire → SQE: read v1
                                                  → yield
                        ┌─ io_uring_enter() ──┐
                        │ submit: curr, v1    │
                        │ wait for CQEs...    │
                        └─────────────────────┘
                        CQE: curr done ────────→
  ← wake (curr READY)                          CQE: v1 done ──→
  pin, return guard                               → v1 READY, wake_all
compute(neighbors)                                recv(v2) → read v2...
  ... (sync, no yield)
drop(guard)
hint(v3), hint(v4)      ← next expansion
```

### What changes from current `disk_graph_search`

1. **+6 lines in loop**: prefetch hint loop + kick + window selection
2. **+4 lines**: stall counter for adaptive window
3. **+1 background task**: prefetch worker (spawned once at pool creation)
4. **CandidateHeap needs `peek_top(n)`**: view top-N without popping
5. **AdjacencyPool needs**: `prefetch_hint()`, `kick_prefetches()`, `is_resident()`, `is_loading()`
6. **IoDriver needs**: `available_adj_permits()`, `adj_capacity()`
7. **SearchPerfContext needs**: `prefetch_issued`, `prefetch_consumed`

The existing search loop structure is preserved. The only yield point
remains `get_or_load().await` — prefetch adds zero additional await points
to the hot loop. The `disk_graph_search` (no prefetch) remains the baseline —
`disk_graph_search_pipe` is a separate function for A/B comparison.

---

## 7. Four Micro-Experiments

### Required metrics for ALL experiments

Every experiment must output this full table per configuration:

| Metric | Source | Why it kills you if missing |
|--------|--------|-----------------------------|
| p50 / p99 / p999 total_us | SearchPerfContext timers | p50 down + p99 up = net loss, invisible without p999 |
| io_wait% | `io_wait_ns / total_ns` | Prefetch should shrink this; if it doesn't, prefetch is useless |
| compute% | `compute_ns / total_ns` | If compute% drops, you're losing CPU to prefetch overhead |
| avg inflight depth | sample `adj_inflight - available_permits` at each expansion | Histogram, not just mean — reveals queueing |
| prefetch_waste% | `(prefetch_issued - prefetch_consumed) / prefetch_issued` | The number that tells you if prefetch is reading garbage |
| eviction_churn / query | `(evictions_after - evictions_before) / num_queries` | Prefetch evicting useful cached blocks = cache thrashing |
| recall@k | Standard | Prefetch must not degrade accuracy |

### EXP-P0: Steady-State Prefetch Impact (Mixed Hot/Cold)

**Question**: Does prefetch help or hurt under realistic steady-state access
patterns? (Cold-cache experiments overstate prefetch's value.)

**Method**:
- Dataset: Cohere 100K, O_DIRECT
- Cache NOT reset between queries (persistent across all 200 queries)
- Queries drawn from Zipf distribution (α=1.0, ~80/20 hot/cold split)
- Use best W from EXP-P1 (run P0 after P1)
- Compare: W=0 (baseline) vs W=best
- Measure: full metrics table, especially:
  - **eviction_churn / query**: does prefetch evict hot blocks?
  - **p99 / p999**: does prefetch make tail latency worse under warm cache?
  - **hit rate delta**: does prefetch improve or degrade cache effectiveness?

**What to look for**:
- If eviction_churn rises significantly with prefetch on, the speculative
  reads are polluting the cache (evicting hot blocks to load blocks that may
  never be consumed). This is the "scan pollution" problem.
- If p99 stays flat or improves, prefetch is safe for production.
- If p99 degrades while p50 improves, prefetch needs admission control
  (only prefetch when waste ratio is low — feed into §5 signals).

**Implementation**: Remove `pool.clear()` between queries. Add Zipf query
generator (simple: `(rand::random::<f64>().powf(1.0/alpha) * N) as usize`).

### EXP-P1: Fixed-Window Prefetch Latency Sweep

**Question**: What is the relationship between prefetch window size and
latency/tail-latency under cold cache?

**Method**:
- Dataset: Cohere 100K (same as EXP-0/C/W/T)
- Fixed ef=200 (201 blocks/query)
- Run 100 queries with W ∈ {0, 1, 2, 4, 8} (fixed, no adaptation)
- Cold cache between queries (new pool per query)
- O_DIRECT mandatory (100K fits in page cache otherwise)
- Measure: full metrics table above

**What to look for**:
- Is there a sweet spot W where p50 drops without p99 rising?
- Does the inflight depth histogram show the pipeline is actually being used?
- At what W does prefetch_waste% start climbing?
- Does eviction_churn track W linearly, or is there a knee?

**Implementation**: `disk_graph_search_pipe` with fixed W (bypass
`select_prefetch_window`, just return constant). Reuse existing harness.

### EXP-P2: Cache-Aware vs Cache-Blind Prefetch

**Question**: Does checking `is_resident()` before prefetching matter
under warm cache?

**Method**:
- Same setup as EXP-P1, but warm cache (run 100 queries, then measure next 100)
- Use best W from EXP-P1
- Compare two strategies:
  - **Blind**: always call `prefetch_hint` for top-W candidates
  - **Aware**: skip candidates where `pool.is_resident(vid) == true`
- Measure: full metrics table, especially eviction_churn and hit rate delta

**What to look for**:
- Does blind prefetch cause eviction churn (evict cached block to re-load same block)?
- Is the hit rate delta significant (>5%) or noise?
- Under cold cache, blind and aware should be identical — verify this as sanity check

**Implementation**: Boolean flag `cache_aware: bool` in search params.

### EXP-P3: Adaptive Window vs Fixed Window

**Question**: Does signal-driven adaptation (§5) outperform fixed W?

**Method**:
- Same setup as EXP-P1 (cold cache, O_DIRECT)
- Three arms:
  - **Fixed W=best** (from EXP-P1)
  - **Adaptive** (`select_prefetch_window` with all 4 signals)
  - **Fixed W=best, B=4** (4 concurrent queries per core — inter-query overlap)
  - **Adaptive, B=4** (both mechanisms)
- Sweep across B ∈ {1, 4, 8} for each W strategy
- Measure: full metrics table

**What to look for**:
- Does the sweet spot W shift when B changes? (More queries = more pipeline
  contention = W should shrink)
- Does adaptive actually react to stalls and waste, or is it just noise?
- Is there a (W, B) combination where p999 degrades despite p50 improving?

**Implementation**: `select_prefetch_window` + multi-query harness (spawn B
search coroutines per monoio thread).

### What these experiments do NOT test

- Throughput under load (requires saturating the NVMe — EXP-P4, later)
- Large datasets (100K is small — io latency dominated by page cache without O_DIRECT)
- Cross-core cache interaction (per-core pools are independent by design)

For EXP-P1 to be meaningful on 100K vectors, we MUST use O_DIRECT and
cold cache to force real NVMe reads. Otherwise the "IO" is just page cache
memcpy and prefetch adds no measurable value.

---

## 8. Registered Buffers Verdict

### Summary

| Aspect | Finding |
|--------|---------|
| IOPS improvement | Reported ~10% for 4KB random reads (fio benchmarks). **Unverified on our hardware.** |
| monoio support | **None**. No `register_buffers`, no `READ_FIXED` op. |
| Implementation cost | Must use raw `io-uring` crate, bypass monoio's buf ownership model |
| RLIMIT_MEMLOCK | Registered buffers count against locked memory limit (default 64KB on many systems) |
| UIO_MAXIOV | Hard kernel limit: 1024 registered buffers max |
| glommio approach | Single pool via buddy allocator, automatic READ_FIXED, but glommio owns the runtime |

### Decision: **Do not adopt registered buffers for MVP.**

**Reasoning** (based on engineering cost, not speculative performance):

1. **monoio has no support.** We'd need to either:
   - Fork monoio and add `register_buffers` + `READ_FIXED` op support
   - Use raw `io-uring` crate alongside monoio (two submission queues on one
     ring = fragile, hard to reason about ownership)
   - Switch to glommio (different runtime, different async model, different
     buf ownership — rewrite of io.rs + cache.rs)

   All three options are high-risk engineering changes for uncertain gain.

2. **SlotStore already delivers the primary buffer benefit.** Our buffers are:
   - Pre-allocated at pool creation (no hot-path alloc)
   - 4KB-aligned (matches O_DIRECT alignment requirement)
   - Stable addresses (no realloc, no move)
   - Reused across queries (no per-read allocation)

   Registered buffers' main benefit is avoiding the kernel's `pin_user_pages`
   on each IO. With O_DIRECT + aligned buffers, the kernel pin path is already
   fast. The delta is real but unquantified on our specific hardware/kernel.

3. **Performance gain is unverified.** The "~10%" number comes from fio
   benchmarks under specific conditions. We have not measured it on our
   NVMe (Samsung PM9A3) with our access pattern (4KB random reads, 4-32
   queue depth, O_DIRECT). It could be 2% or 15%. We should not claim a
   specific number in this document.

4. **Future path is clean.** SlotStore's contiguous allocation is already
   register-ready: `io_uring_register_buffers(&[iovec { base: slot_store.ptr,
   len: capacity * 4096 }])`. When runtime support appears, the change is
   ~20 lines in io.rs — no cache architecture change.

### When to reconsider

- After prefetch/overlap is stable and NVMe p99 is in a measured steady state
- Run a targeted microbench: `read(4KB) × N` IOPS comparison, regular READ vs
  READ_FIXED using raw io-uring crate (not through monoio). This gives us our
  own number instead of relying on fio benchmarks.
- If monoio adds `register_buffers` support natively
- If we switch to glommio for other reasons (priority rings, etc.)
- At billion-scale, where per-read overhead compounds across 56 cores × 200K QPS

---

## 9. Implementation Priority

```
Phase 1 (done):        AdjacencyPool ✓
Phase 2 (next):        Prefetch worker + prefetch_hint + kick_prefetches
                       + CandidateHeap::peek_top() + IoDriver::adj_capacity()
                       (~150 lines: worker, channel, 3 pool methods)
Phase 3:               EXP-P1 (fixed W sweep, cold cache)
                       → find sweet spot W, check p50/p99/p999
Phase 4 (if P1 wins):  EXP-P0 (steady-state Zipf, warm cache)
                       → check eviction churn, p99 safety under load
Phase 5 (if P0 safe):  EXP-P3 (B × W sweep, inter+intra query overlap)
                       → find contention sweet spot
Phase 6 (last):        Adaptive window (§5 signal logic)
                       → only if EXP-P3 shows W needs to vary with B
```

**Rule**: Do NOT implement adaptive window (Phase 6) before having the
EXP-P1 baseline curve. Tuning parameters without a measured curve is noise.

---

## Appendix A: PipeANN vs VeloANN vs Divergence Comparison

| | PipeANN | VeloANN | Divergence (target) |
|--|---------|---------|---------------------|
| **Overlap type** | Intra-query | Inter-query | Both |
| **Cache** | None | Per-core (hash) | Per-core (set-assoc, clock) |
| **Singleflight** | N/A | N/A | Yes (WaiterArray) |
| **IO model** | io_uring, poll-based | io_uring, coroutine | io_uring via monoio, coroutine |
| **Pipeline width** | Dynamic (4→32) | B=ceil(α×I/T) queries | W adaptive + B queries |
| **Entry point** | In-memory Vamana | Graph layer 0 | TBD (HNSW upper layers in DRAM) |
| **Waste mitigation** | One-at-a-time issue | Cache-aware pivot | prefetch_hint (evictable) |

### Key difference from PipeANN

PipeANN has no cache, so every block is a miss. Its pipeline always reads
from disk. In Divergence, warm cache means many expansions are cache hits
(~50% in our EXP-C data). Prefetch's value is primarily for the **cold
misses** — the blocks not yet in cache. This means:

- Our prefetch window can be smaller (W=2-4 vs PipeANN's 4-32)
- Cache-aware prefetch (skip resident blocks) is important for us
- The approach/converge phase split matters less (cache absorbs approach waste)

### Key difference from VeloANN

VeloANN relies on inter-query overlap alone — when one query stalls on IO,
another runs. This works well at high QPS but provides zero benefit for a
single query's latency. PipeANN-style intra-query prefetch reduces single-
query latency, which is critical for tail latency SLAs.

Divergence should use both: VeloANN's B-query batching for throughput,
PipeANN's prefetch for per-query latency.
