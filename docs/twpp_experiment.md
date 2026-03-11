# EXP-TWPP: Trace-Weighted Page Packing

**Date**: 2026-03-10
**Dataset**: Cohere 100K, dim=768, cosine, k=100
**Hardware**: i4i.xlarge (1 core), NVMe SSD, O_DIRECT, io_uring

---

## Problem

Page-packed adjacency (v3 layout) packs variable-degree adjacency records into 4KB pages.
The packing order determines which VIDs share a page. When two VIDs that are frequently
accessed in the same query share a page, one IO serves both — reducing physical reads and
cache misses.

BFS reorder (our baseline) packs nodes in breadth-first order from the entry set. This
captures graph locality but is query-agnostic — it doesn't know which nodes are actually
co-accessed during search.

## Hypothesis

A trace-driven packing that directly observes which VIDs are co-expanded in the same query
will produce better page co-location than BFS reorder, reducing cache miss rate, physical
IO count, and query latency — without affecting recall.

## Result: NEGATIVE

**TWPP does not generalize to unseen queries.** When evaluated with proper train/test
separation, TWPP shows no meaningful improvement over BFS. The dramatic gains in the
initial (flawed) run were entirely due to train/test leakage.

---

## Prior Art

- **PageANN** (VLDB 2024): topology-aware page packing with inline compressed neighbor
  codes. Packs graph neighbors onto the same page so next-hop scoring requires zero extra
  IO. Uses static graph structure, not query traces.
- **DiskANN**: BFS reorder for cache locality. Our BFS baseline follows this approach.

## Design

### Signal: Co-Expansion Pairs

During beam search, when VID `u` is expanded and one of its neighbors `v` is later also
expanded, we call `(u, v)` a **co-expansion pair**. The parent-child relationship is:
`u` pushed `v` into the candidate heap (via neighbor discovery), and both were expanded.

We track `parent_of[v] = u` during search. When `v` is expanded and `parent_of[v]` was
already expanded, we record `(u, v)` with weight +1. Weights are accumulated across all
trace queries with symmetric keys `(min(u,v), max(u,v))`.

**Note on `parent_of` overwrite**: If multiple expanded nodes push the same neighbor `v`
into the candidate heap, `parent_of[v]` records only the *last* pusher before `v` is
expanded. This is a heuristic — the true "causal parent" may be an earlier pusher. In
practice this rarely matters because (a) most nodes are pushed by only one parent (the
first discoverer), and (b) the symmetric key means the co-expansion pair is recorded
regardless of direction.

### TraceRecorder

```rust
pub struct TraceRecorder {
    pub co_expand: HashMap<(u32, u32), u32>,  // symmetric key -> weight
    pub node_counts: HashMap<u32, u32>,        // vid -> total expansions
}
```

### Greedy Page Packer

`twpp_reorder_graph()` in `crates/storage/src/adjacency.rs`:

1. **Sort VIDs by expansion count** (descending).
2. **Seed pages** with the most-expanded VIDs.
3. **Greedy fill**: For each remaining VID, score against open pages by summing co-expand
   weights to placed members. Place on highest-scoring page with room.
4. **Fallback**: VIDs with no co-expand signal assigned sequentially.

Uses byte-accurate record sizing via `packed_record_size(degree) = 2 + degree * 4`.

## Implementation

| File | Change |
|------|--------|
| `crates/engine/src/search.rs` | `TraceRecorder`, `disk_graph_search_pipe_v3_traced()`, inner function refactor with `parent_of` tracking |
| `crates/storage/src/adjacency.rs` | `packed_record_size()`, `twpp_reorder_graph()` |
| `crates/storage/src/lib.rs` | Exports |
| `crates/engine/src/lib.rs` | Exports |
| `crates/engine/tests/disk_search.rs` | `exp_twpp` benchmark test |

## Experiment Structure

```
Phase 1: Trace collection
  - 200 queries [0..200), cold (cache cleared per query), BFS layout
  - Validation: traced == untraced (identical results + expansion count)

Phase 2: TWPP reorder computation
  - Feed co_expand + node_counts to twpp_reorder_graph()
  - Write TWPP v3 layout (same graph, different page assignment)

Phase 3: Cold A/B benchmark
  - 100 HOLDOUT queries [200..300), cache cleared per query
  - Disjoint from trace queries — no train/test leakage

Phase 3b: Unique pages cross-check
  - 50 holdout queries, traced + cold
  - Compare unique_pages/q (from adj_index) to phys/q

Phase 4: Warm A/B benchmark
  - Warmup on DISJOINT queries [300..320), measure [200..300)
  - No overlap between warmup and measured queries
```

## Trace Statistics

| Metric | Value |
|--------|-------|
| Unique co-expand pairs | 33,722 |
| Unique VIDs expanded | 21,528 / 100,000 (21.5%) |
| Total expansions | 40,196 |
| Max node expansion count | 63 |
| Total co-expand weight | 39,959 |
| Pages: BFS 1,537, TWPP 1,536 | -1 delta |

## Results (Hardened — Proper Train/Test Split)

### Cold Benchmark (100 holdout queries [200..300), per-query cache clear)

| Layout | Recall | p50 (ms) | p99 (ms) | QPS | miss/q | phys/q | pf_i/q | pf_c/q | sf/q |
|--------|--------|----------|----------|-----|--------|--------|--------|--------|------|
| BFS | 0.956 | 8.4 | 11.6 | 104.1 | 24.5 | 136.6 | 113.3 | 109.9 | 40.8 |
| TWPP | 0.956 | 8.0 | 10.3 | 109.0 | 25.0 | 139.2 | 115.6 | 110.2 | 41.3 |

Cache totals (cold):
- BFS:  hits=17644 misses=2451 bypasses=0 evictions=96     evict_fail=0 pf_hits=10986 phys=13693
- TWPP: hits=17595 misses=2500 bypasses=0 evictions=153    evict_fail=0 pf_hits=11021 phys=13965

### Unique Pages Cross-Check (50 holdout queries, traced cold)

| Layout | unique_pages/q | phys/q | delta (prefetch waste) |
|--------|---------------|--------|------------------------|
| BFS | 134.7 | 136.6 | 1.9 |
| TWPP | 135.0 | 138.9 | 4.0 |

Under cold, phys/q closely matches unique_pages/q — accounting is clean.
TWPP touches the SAME number of unique pages as BFS. No locality benefit.

### Warm Benchmark (warmup [300..320), measure [200..300))

| Layout | Recall | p50 (ms) | p99 (ms) | QPS | miss/q | phys/q | pf_i/q | pf_c/q | sf/q |
|--------|--------|----------|----------|-----|--------|--------|--------|--------|------|
| BFS-warm | 0.956 | 7.4 | 10.1 | 132.4 | 20.2 | 114.5 | 94.3 | 91.8 | 38.5 |
| TWPP-warm | 0.956 | 7.3 | 9.4 | 136.2 | 19.7 | 116.8 | 97.1 | 92.5 | 37.8 |

Cache totals (warm):
- BFS:  hits=21646 misses=2473 bypasses=0 evictions=13658 evict_fail=0 pf_hits=11125 phys=13914
- TWPP: hits=21723 misses=2396 bypasses=0 evictions=13922 evict_fail=0 pf_hits=11214 phys=14174

### Pool Configuration (identical for both)

| Param | Value |
|-------|-------|
| pool_bytes | 1,048,576 (1 MB) |
| num_sets | 32 |
| total_slots | 256 |
| SET_WAYS | 8 |

## Analysis: Why TWPP Failed

### 1. Train/Test Leakage Was the Entire Effect

The initial (flawed) experiment used the SAME queries for both trace collection and
benchmarking. When you benchmark on the queries you trained on:
- The packer places those queries' co-expanded nodes together
- Those exact queries see perfect page co-location by construction
- The "improvement" is just memorization, not generalization

With proper holdout queries, TWPP and BFS have **identical** performance within noise.

### 2. Sparse Coverage (21.5% of VIDs)

200 trace queries expand ~40K VIDs total but only 21.5K unique VIDs. The remaining 78.5%
of VIDs have no trace signal and fall back to sequential placement — same as BFS. Even the
covered VIDs have weak signal: max node count is 63 (out of 200 queries), and most nodes
are expanded only 1-2 times.

### 3. Co-Expansion Patterns Are Query-Specific

The beam search path through the graph depends on query vector direction. Two different
queries starting from the same entry points rapidly diverge into different graph regions.
Co-expansion pairs from query A have little predictive value for query B — the paths
through the graph are too different.

This is fundamentally different from PageANN's approach of packing graph *neighbors*
together, which exploits the structural property that expanding node u always reads u's
neighbor list (regardless of query). That's a static, universal pattern. Co-expansion
is a query-dependent, non-universal pattern.

### 4. BFS Already Captures the Real Locality

BFS reorder places graph-proximate nodes on the same page. Since beam search follows
graph edges, this already provides decent locality. The unique_pages/q cross-check
confirms: BFS and TWPP touch the same number of unique pages (~135/q out of 201 lookups).
BFS achieves this without any query traces.

## Comparison: Initial Run vs Hardened Run

| Metric | Initial (leaked) | Hardened (holdout) | Interpretation |
|--------|-------------------|---------------------|----------------|
| BFS cold p50 | 8.3 ms | 8.4 ms | Consistent |
| TWPP cold p50 | 6.1 ms | 8.0 ms | Leaked run was memorized |
| TWPP cold phys/q | 74.1 | 139.2 | Leaked run: packer placed train-query nodes together |
| BFS warm p50 | 7.3 ms | 7.4 ms | Consistent |
| TWPP warm p50 | 4.6 ms | 7.3 ms | Leaked: warm amplified train-query memorization |
| TWPP warm phys/q | 35.5 | 116.8 | No generalization |

## Lessons Learned

1. **Always use holdout queries.** Train/test leakage in layout optimization is subtle:
   the "layout" doesn't look like a model, but any data-driven reordering that sees
   benchmark queries will overfit to them.

2. **Query-dependent signals don't generalize for static layouts.** A layout is fixed at
   index build time. Optimizing it for specific queries helps those queries but not others.
   Static graph properties (neighbor adjacency, BFS layers) are more robust.

3. **Cross-check with unique_pages/q.** This metric would have caught the problem
   immediately: if TWPP truly improved locality, unique_pages/q would be lower (same
   VIDs, fewer distinct pages). It wasn't.

4. **Prefetch stats as sanity check.** Both layouts had nearly identical prefetch
   issued/consumed — confirming the difference (or lack thereof) was pure locality,
   not a prefetch artifact.

## What PageANN Does Differently

PageANN's topology-aware packing works because it exploits a *universal* property:
expanding node u always reads u's adjacency list, and u's neighbors are the most likely
next expansions. Placing u's adjacency record on the same page as its neighbors' records
means one IO naturally serves multiple expansions.

This is a structural property of the graph, not a query-dependent signal. It generalizes
to all queries because it's built from the graph topology, not query traces.

Our TWPP approach tried to learn *which* graph paths are most common, but at 100K scale
with 768d cosine, the query distribution is too spread out for any fixed set of traces
to predict future queries.

## Files

| File | Purpose |
|------|---------|
| `crates/engine/src/search.rs` | TraceRecorder, traced search variant |
| `crates/storage/src/adjacency.rs` | twpp_reorder_graph(), packed_record_size() |
| `crates/engine/tests/disk_search.rs` | exp_twpp benchmark |

## Reproduction

```bash
# On EC2 i4i instance with NVMe
BENCH_DIR=/mnt/nvme/bench \
COHERE_DIR=/mnt/nvme/divergence/data/cohere_100k \
COHERE_N=100000 \
cargo test --release -p divergence-engine --test disk_search exp_twpp \
  -- --nocapture --ignored
```
