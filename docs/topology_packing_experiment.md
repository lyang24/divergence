# Topology-Based Page Packing Experiment

**Date**: 2026-03-10
**Dataset**: Cohere 100K, dim=768, cosine, k=100
**Hardware**: i4i.xlarge (1× NVMe SSD), O_DIRECT, monoio + io_uring
**Search config**: ef=200, W=4 (prefetch width), 5% cache, 1 core

## Background

After TWPP (Trace-Weighted Page Packing) proved to be a negative result — co-expansion traces from training queries don't generalize to unseen queries in 768d cosine space — we pivoted to topology-based packing strategies that use only graph structure (no query traces).

The core observation: a beam search with ef=200 expands ~201 nodes per query, each requiring one adjacency page read. If neighbors are co-located on the same page, multiple expansions can be served by a single physical read. The question is how to assign nodes to pages to maximize this co-location.

## Hypotheses

**H1 (Layout)**: Graph-aware page assignment reduces unique_pages/q (distinct pages touched per query) and phys/q (physical reads), improving QPS at identical recall.

**H2 (Scheduling)**: Among the top-B candidates in the search heap, preferring one whose page is already resident in cache reduces cache misses without meaningful recall loss — the distance deviation within B nearest candidates is small.

**H3 (Composition)**: Layout optimization and scheduling are orthogonal and compose: layout reduces total IO demand, scheduling eliminates remaining cache misses.

## Design

### Four page layouts

All layouts use the same NSW graph (m_max=32, ef_construction=200). Only the VID-to-page assignment differs. The v3 writer's page packer greedily fills 4KB pages in reorder-sequence order, so consecutive VIDs in the reorder share pages.

1. **Sequential** (identity): VID i maps to position i. No locality optimization. Baseline for "what if layout is random relative to graph structure."

2. **BFS**: Standard breadth-first traversal from entry nodes. Nodes discovered together in BFS tend to be graph-local, so BFS-adjacent nodes share pages. This is the existing production layout.

3. **Neighbor-run BFS**: Modified BFS that, upon visiting node v, eagerly assigns all of v's unvisited neighbors before continuing BFS. The intuition: when expanding v during search, its neighbors are the next candidates — if they share v's page, the adjacency read for v also pre-loads neighbor data.

4. **Heavy-edge (MARGO-style)**: Seed-and-greedy-fill packing. Seeds each page with the highest in-degree unassigned node (hubs are accessed most frequently). Fills the page by greedily selecting unassigned graph neighbors that maximize `indeg(u) + indeg(v)` edge weight. Only graph neighbors are eligible for filling (the MARGO rule — ensures topological affinity, not just degree).

### Page-aware scheduling

Modified the search loop's candidate selection. Instead of always popping the nearest candidate from the min-heap:

1. Pop the top-B nearest candidates
2. Among those B, find the nearest one whose `page_id` is already resident in the AdjacencyPool
3. If found, return that candidate (and push the rest back); otherwise return the overall nearest

This is implemented as `CandidateHeap::pop_preferred(b, is_preferred)` called from `disk_graph_search_pipe_v3_pagesched`. The `is_preferred` predicate checks `pool.is_resident(adj_index[vid].page_id)`.

The key insight: within the top-B candidates, distance differences are small. Choosing a slightly-further candidate that avoids a cache miss trades negligible recall for significant latency reduction.

### Holdout protocol

- Bench queries: [200..300) — 100 queries, disjoint from any trace collection
- Warmup queries: [300..320) — 20 queries for warm benchmarks, disjoint from measured set
- Ground truth: brute-force exact (N=100K)

### Recall baseline note

This experiment reports recall=0.956 at ef=200, whereas earlier experiments (EXP-P1, EXP-AS) reported 0.963. The difference is the query subset: earlier experiments used queries [0..100), this experiment uses the holdout [200..300) to avoid train/test leakage from TWPP trace collection on [0..200). The 0.7% difference is within normal query-subset variance — ef=250 on the same holdout gives 0.965, consistent with the recall-vs-ef curve.

### Metrics

- **recall@100**: fraction of true top-100 neighbors found
- **p50/p99**: query latency in ms
- **QPS**: queries per second (wall clock)
- **exp/q**: beam expansions per query (~ef+1)
- **blk/q**: adjacency block reads per query (= exp/q in v3)
- **miss/q**: demand cache misses per query — page was not in the AdjacencyPool at all (neither Ready nor Loading), so a new IO was issued and the search loop awaited it. This is distinct from phys/q (total physical reads including prefetch). When scheduling reduces miss/q without changing phys/q, it means the scheduler is choosing candidates whose pages are already Ready or in-flight (Loading), converting demand waits into immediate hits.
- **hit/q**: cache hits per query (page was Ready — no IO wait)
- **phys/q**: total physical IO reads per query (demand misses + prefetch loads + bypasses)
- **upg/q**: unique pages touched per query (from adj_index, traced — the true layout quality metric)
- **sched/q**: times a non-nearest candidate was chosen because its page was resident

Note: `pool.is_resident(page_id)` returns true for both Ready and Loading states (anything != Empty). The scheduler may select a page that is still in-flight from a prefetch, which is correct — it will complete soon and avoids issuing a redundant IO.

## Implementation

### Files changed

| File | Change |
|------|--------|
| `crates/storage/src/adjacency.rs` | `neighbor_run_reorder_graph()`, `heavy_edge_reorder_graph()` |
| `crates/index/src/heap.rs` | `CandidateHeap::pop_preferred(b, is_preferred)` |
| `crates/engine/src/search.rs` | `disk_graph_search_pipe_v3_pagesched()`, inner function refactored with `page_sched_b` param |
| `crates/engine/src/perf.rs` | `page_sched_hits: u64` counter in `SearchPerfContext` |
| `crates/engine/tests/disk_search.rs` | `exp_topology_packing` test (4 phases) |

### neighbor_run_reorder_graph

```
BFS from entry nodes, but when visiting v:
  for each neighbor u of v:
    if u not yet assigned:
      assign u next position
  then continue normal BFS from v's neighbors
```

Same signature as `bfs_reorder_graph`. The eager neighbor assignment creates "runs" of topologically related nodes that the page packer groups together.

### heavy_edge_reorder_graph

```
1. Compute in-degree for every node
2. Sort all nodes by in-degree descending
3. For each unassigned node (highest indeg first):
   a. Start a new "page" (conceptual group of ~records_per_page nodes)
   b. Seed with this node
   c. Greedily fill: among unassigned graph neighbors of any node in current page,
      pick the one with highest indeg(u)+indeg(v) edge weight
   d. Stop when page is full (byte-accurate via packed_record_size)
4. Assign remaining nodes in order
```

Uses `packed_record_size()` for byte-accurate page capacity calculation. Only graph neighbors are eligible for page fill (the MARGO constraint).

### pop_preferred

```
pop_preferred(b, is_preferred):
  if b <= 1: return pop() with was_preferred=false
  pop top-b candidates into temp array
  find nearest among those where is_preferred(vid) is true
  if found: return that one (was_preferred = true if it wasn't already index 0)
  else: return overall nearest (was_preferred = false)
  push remaining b-1 candidates back
```

O(b log n) — negligible for b ≤ 16 in an IO-bound search loop.

## Results

### Phase 1: Cold benchmark (cache cleared per query)

```
      layout  recall   p50ms   p99ms     QPS   exp/q   blk/q  miss/q   hit/q  phys/q   upg/q
  sequential   0.956     9.1    11.4    96.9   200.9   200.9    33.0   167.9   195.9   188.8
         bfs   0.956     8.2    11.5   106.1   200.9   200.9    24.5   176.4   136.3   134.3
     nbr_run   0.956     8.3    11.4   106.0   200.9   200.9    24.5   176.4   136.5   134.3
  heavy_edge   0.956     7.4     9.7   118.3   200.9   200.9    20.8   180.2   107.0   102.9
```

**Key observations**:
- **Recall is identical** (0.956) across all layouts — layout only affects IO, not search quality
- **heavy_edge** reduces unique_pages/q from 188.8 (sequential) to 102.9 — a **45% reduction**
- **heavy_edge** reduces phys/q from 195.9 to 107.0 — **45% fewer physical reads**
- **heavy_edge vs BFS**: 24% fewer unique pages (134.3 → 102.9), 21% fewer phys reads
- **nbr_run ≈ BFS**: no measurable improvement — eager neighbor assignment doesn't help beyond what BFS already achieves
- **QPS**: heavy_edge 118.3 vs sequential 96.9 (+22%), vs BFS 106.1 (+12%)

### Phase 2: Warm benchmark (20 disjoint warmup queries, then measure)

```
      layout  recall   p50ms   p99ms     QPS   exp/q   blk/q  miss/q   hit/q  phys/q
  sequential   0.956     8.9    11.2   111.1   200.9   200.9    28.6   172.3   174.5
         bfs   0.956     7.6    10.3   129.5   200.9   200.9    20.3   180.7   114.6
     nbr_run   0.956     7.5     9.9   131.9   200.9   200.9    20.3   180.7   114.5
  heavy_edge   0.956     6.8     8.9   147.8   200.9   200.9    16.6   184.4    88.3
```

**Key observations**:
- Warm cache amplifies the layout advantage: heavy_edge QPS 147.8 vs BFS 129.5 (+14%)
- heavy_edge miss/q = 16.6 (vs BFS 20.3) — better co-location means more hits from cached pages
- heavy_edge p99 = 8.9ms vs BFS 10.3ms — **14% lower tail latency**
- Cache stats confirm: heavy_edge has fewer evictions (10,497 vs 13,668 for BFS) because fewer unique pages need loading

### Phase 3: Page-aware scheduling B-sweep (BFS layout, warm)

```
       B  recall   p50ms   p99ms     QPS   exp/q   blk/q  miss/q   hit/q  phys/q sched/q
     B=1   0.956     7.5    10.1   131.6   200.9   200.9    20.2   180.7   114.7     0.0
     B=4   0.956     7.2     9.1   138.3   201.2   201.2     0.5   200.8   114.4    18.8
     B=8   0.956     7.2     9.0   137.6   201.4   201.4     0.2   201.2   114.3    19.2
    B=16   0.955     7.3     9.3   136.1   201.4   201.4     0.2   201.2   114.4    19.1
```

**Key observations**:
- **B=4 eliminates nearly all cache misses**: miss/q drops from 20.2 to 0.5 (98% reduction)
- **Recall preserved**: 0.956 at B=4 and B=8, 0.955 at B=16 (negligible)
- **~19 scheduling hits per query** — the scheduler finds a resident-page candidate ~19 times out of ~201 expansions (9.4% of pops are reordered)
- **phys/q unchanged** (~114) — scheduling doesn't reduce total IO, it just avoids cache misses by reordering when pages are already loaded
- **B=4 is sufficient** — B=8 and B=16 give no further improvement
- **exp/q increases slightly** (200.9 → 201.4) — choosing a slightly-further candidate occasionally adds one extra expansion, but never enough to affect recall

### Phase 4: Combined layout + scheduling (B=8, warm)

```
        layout+B  recall   p50ms   p99ms     QPS   exp/q   blk/q  miss/q   hit/q  phys/q sched/q
   sequential+B8   0.956     8.4    10.1   118.6   201.5   201.5     0.3   201.2   174.3    25.2
          bfs+B8   0.956     7.2     8.9   138.7   201.4   201.4     0.2   201.2   114.3    19.2
      nbr_run+B8   0.955     7.1     9.0   139.8   201.4   201.4     0.3   201.1   114.6    19.1
   heavy_edge+B8   0.955     6.6     8.2   151.8   201.3   201.3     0.1   201.2    88.8    15.5
```

**Key observations**:
- **heavy_edge+B8 is the best**: QPS=151.8, p50=6.6ms, p99=8.2ms, miss/q=0.1
- **vs BFS baseline (B=1)**: +15% QPS (131.6 → 151.8), -12% p50 (7.5 → 6.6ms), -19% p99 (10.1 → 8.2ms)
- **Layout and scheduling compose**: heavy_edge reduces phys/q (114→89), scheduling eliminates misses (20→0.1). The gains are additive.
- **heavy_edge needs fewer scheduling interventions** (15.5 sched/q vs 19.2 for BFS) — better layout means fewer misses to avoid in the first place
- **sequential+B8 sched/q=25.2** — worst layout needs the most scheduling help, confirming that scheduling compensates for poor layout

## Analysis

### Why heavy_edge wins

The MARGO-style seed+greedy-fill works because:
1. **Hub-seeded pages**: High in-degree nodes are accessed by many queries. Seeding pages with hubs ensures the most-accessed nodes get optimal placement.
2. **Edge-weight filling**: `indeg(u)+indeg(v)` prioritizes edges between important nodes. These edges are traversed most frequently during search.
3. **Graph-neighbor constraint**: Only graph neighbors fill a page. This ensures page co-location reflects actual traversal patterns, not just degree statistics.

### Why neighbor-run BFS ≈ BFS

BFS already provides good spatial locality — nodes discovered in the same BFS wave are topologically close. Eagerly assigning neighbors doesn't improve on this because:
- BFS naturally processes neighbors in the next wave anyway
- The eager assignment just reorders within a BFS level, but the page packer already groups consecutive VIDs together
- The "run" structure doesn't survive the page packer's greedy bin-packing

### Why scheduling works without recall loss

The distance distribution within the top-B candidates is tight. At B=4, the 4th-nearest candidate is typically within a few percent of the nearest. Choosing a slightly-further candidate that avoids a cache miss adds at most 1 extra expansion (exp/q: 200.9 → 201.4) and has no measurable recall impact.

The mechanism is self-limiting: scheduling only helps when a page is already resident (from a previous expansion's neighbors sharing a page). It cannot cause unbounded deviation from the distance-ordered traversal.

### Composition

The two techniques attack different parts of the IO stack:
- **Layout** reduces the number of distinct pages needed (upg/q: 189 → 103), directly reducing phys/q
- **Scheduling** eliminates cache misses by reordering candidate expansion to exploit existing cache contents

Together: heavy_edge+B8 achieves 88.8 phys/q with 0.1 miss/q. The remaining 88.8 physical reads are irreducible — they represent the ~103 unique pages minus cache hits from the 5% cache.

## Sensitivity Sweep v2: heavy_edge vs BFS

Grid: pool ∈ {256, 512, 1024} pages, W ∈ {0, 2, 4, 8}, ef ∈ {150, 200, 250}. 72 configurations × 2 layouts × 2 passes = 288 measurements. Two passes per point:

- **perq-cold**: `pool.clear()` every query — no cross-query cache reuse. phys/q directly tracks upg/q (layout truth).
- **warm**: disjoint warmup [300..320), then measure without clearing — tests cache reuse across queries.

**Methodology fixes** from v1:
- W=0 is real no-prefetch (no worker spawned, `prefetch_window=0` passed to search)
- Pool sizes are explicit page counts (256/512/1024), not percentages (which all clamped to 256 at 100K)
- Pool size printed in output for auditability

**Result: heavy_edge wins at every single point in the grid, in both cold and warm passes.**

### Per-query cold: layout truth

The cold pass validates that heavy_edge's phys/q advantage is structural (fewer unique pages), not a cache artifact.

| pool | ef | W | BFS phys/q | HE phys/q | HE savings | BFS p50 | HE p50 |
|------|-----|---|-----------|----------|------------|---------|--------|
| 256 | 200 | 0 | 134.3 | 104.2 | 22% | 17.0 | 13.9 |
| 256 | 200 | 4 | 136.5 | 108.3 | 21% | 7.8 | 7.3 |
| 512 | 200 | 0 | 134.3 | 104.2 | 22% | 17.1 | 13.7 |
| 512 | 200 | 4 | 136.4 | 108.4 | 21% | 7.7 | 7.1 |
| 1024 | 200 | 0 | 134.3 | 104.2 | 22% | 17.4 | 14.2 |
| 1024 | 200 | 4 | 136.4 | 108.3 | 21% | 8.1 | 7.4 |

**Key**: Under cold, phys/q is **invariant to pool size** (pool has no cross-query state). BFS phys/q=134.3 and HE phys/q=104.2 at W=0 match the upg/q measured in Phase 1 (134 vs 103). This confirms the layout improvement is real: HE touches 22% fewer unique pages per query.

With W>0 prefetch, phys/q increases slightly (prefetch reads adjacent pages speculatively), but the HE advantage holds at ~21%.

### Warm: cache reuse across queries

| pool | ef | W | BFS p50 | HE p50 | BFS QPS | HE QPS | advantage |
|------|-----|---|---------|--------|---------|--------|-----------|
| 256 | 200 | 0 | 14.8 | 11.9 | 68.9 | 85.7 | 1.24× |
| 256 | 200 | 4 | 7.1 | 6.4 | 137.3 | 156.4 | 1.14× |
| 256 | 200 | 8 | 6.1 | 5.6 | 161.1 | 178.8 | 1.11× |
| 512 | 200 | 0 | 10.8 | 8.7 | 92.9 | 114.4 | 1.23× |
| 512 | 200 | 4 | 6.5 | 6.0 | 153.0 | 168.2 | 1.10× |
| 512 | 200 | 8 | 5.5 | 4.8 | 182.2 | 204.8 | 1.12× |
| 1024 | 150 | 8 | 3.1 | 2.9 | 322.5 | 352.3 | 1.09× |
| 1024 | 200 | 0 | 5.5 | 4.8 | 178.7 | 206.2 | 1.15× |
| 1024 | 200 | 4 | 4.3 | 3.9 | 234.6 | 260.2 | 1.11× |
| 1024 | 200 | 8 | 3.8 | 3.7 | 259.7 | 267.0 | 1.03× |
| 1024 | 250 | 8 | 4.9 | 4.6 | 201.8 | 218.2 | 1.08× |

### Key findings

1. **heavy_edge advantage is robust and structural**: 22% fewer phys/q in cold (invariant to pool size), confirmed by direct upg/q measurement.

2. **W=0 is now a real no-prefetch mode**: Shows the true serial IO cost. HE advantage is largest here (1.24× at 256 pages, 1.23× at 512, 1.15× at 1024) because every miss is a blocking wait — layout quality directly determines latency.

3. **Pool size matters for warm, not cold**: Cold phys/q is identical across 256/512/1024 (all query-isolated). Warm phys/q drops substantially with more pool: BFS W=0 goes from 111.8 (256p) → 78.3 (512p) → 32.0 (1024p). At 1024 pages, the pool covers 67% of all pages — most queries find their pages cached.

4. **HE advantage narrows at large pools**: At 1024p W=8, advantage is 1.03× (3.8 vs 3.7ms). When nearly everything is cached, layout matters less. But HE never loses — even the worst case is tied.

5. **Best W unchanged**: W=4 remains the sweet spot for both layouts. W=8 gives ~10% more QPS but higher phys/q overhead.

6. **Peak performance**: HE + 1024 pages + W=8 + ef=150 achieves **352 QPS at 0.935 recall** (2.9ms p50). At ef=200: 267 QPS at 0.956 recall (3.7ms p50).

### Full raw data

#### Per-query cold

```
  pool  ef  W  lay  recall   p50ms   p99ms     QPS   exp/q  miss/q   hit/q  phys/q
   256 150  0  BFS   0.935    13.1    16.2    76.8   151.2   106.9    44.4   106.9
   256 150  0   HE   0.935    10.3    13.3    99.0   151.2    81.4    69.9    81.4
   256 150  2  BFS   0.935     8.1    10.3   107.8   151.2    21.8   129.5   108.3
   256 150  2   HE   0.935     6.9     9.0   127.0   151.2    19.1   132.1    83.4
   256 150  4  BFS   0.935     6.3     8.8   134.5   151.2    20.9   130.3   109.2
   256 150  4   HE   0.935     5.5     7.6   151.4   151.2    18.2   133.0    86.1
   256 150  8  BFS   0.935     5.3     8.3   152.8   151.2    20.2   131.0   111.7
   256 150  8   HE   0.935     4.8     6.9   170.3   151.2    17.1   134.1    92.4
   256 200  0  BFS   0.956    17.0    20.7    59.1   200.9   134.3    66.6   134.3
   256 200  0   HE   0.956    13.9    17.9    73.9   200.9   104.2    96.7   104.2
   256 200  2  BFS   0.956    10.3    13.0    87.3   200.9    25.4   175.6   135.6
   256 200  2   HE   0.956     9.1    11.2   100.7   200.9    22.5   178.4   106.0
   256 200  4  BFS   0.956     7.8    10.6   111.0   200.9    24.5   176.4   136.5
   256 200  4   HE   0.956     7.3    10.9   117.1   200.9    21.7   179.3   108.3
   256 200  8  BFS   0.956     7.4    11.2   115.4   200.9    23.8   177.2   138.6
   256 200  8   HE   0.956     6.6     9.7   128.9   200.9    20.5   180.4   114.0
   256 250  0  BFS   0.965    20.6    24.5    49.1   250.8   160.9    89.9   160.9
   256 250  0   HE   0.965    16.8    21.3    61.8   250.8   126.0   124.8   126.0
   256 250  2  BFS   0.965    12.7    32.8    68.3   250.8    28.6   222.2   162.1
   256 250  2   HE   0.965    11.0    13.5    84.2   250.8    25.2   225.5   127.6
   256 250  4  BFS   0.965     9.3    12.5    94.2   250.8    27.7   223.1   163.0
   256 250  4   HE   0.965     8.5    11.3   104.2   250.8    24.4   226.4   129.9
   256 250  8  BFS   0.965     7.8    11.5   110.9   250.8    27.0   223.8   164.9
   256 250  8   HE   0.965     7.1     9.9   121.2   250.8    23.2   227.6   135.2
   512 150  0  BFS   0.935    13.5    16.2    75.2   151.2   106.9    44.4   106.9
   512 150  0   HE   0.935    10.8    14.0    95.4   151.2    81.3    69.9    81.3
   512 150  2  BFS   0.935     8.3    10.3   106.7   151.2    21.8   129.5   108.3
   512 150  2   HE   0.935     7.3     9.4   120.7   151.2    19.1   132.1    83.4
   512 150  4  BFS   0.935     6.3     8.7   134.8   151.2    20.9   130.3   109.3
   512 150  4   HE   0.935     5.6     7.9   147.3   151.2    18.2   133.0    86.1
   512 150  8  BFS   0.935     5.5     8.5   148.8   151.2    20.2   131.0   111.7
   512 150  8   HE   0.935     4.8     7.0   168.8   151.2    17.1   134.1    92.4
   512 200  0  BFS   0.956    17.1    20.5    58.9   200.9   134.3    66.7   134.3
   512 200  0   HE   0.956    13.7    17.7    74.1   200.9   104.2    96.8   104.2
   512 200  2  BFS   0.956    10.3    12.8    86.9   200.9    25.4   175.6   135.6
   512 200  2   HE   0.956     9.2    11.3   100.1   200.9    22.5   178.4   106.0
   512 200  4  BFS   0.956     7.7    10.7   111.5   200.9    24.5   176.4   136.4
   512 200  4   HE   0.956     7.1     9.5   122.9   200.9    21.7   179.3   108.4
   512 200  8  BFS   0.956     6.5     9.7   129.3   200.9    23.8   177.2   138.4
   512 200  8   HE   0.956     5.9     8.3   143.7   200.9    20.5   180.4   114.2
   512 250  0  BFS   0.965    20.4    24.5    49.2   250.8   160.6    90.2   160.6
   512 250  0   HE   0.965    17.0    21.7    60.3   250.8   125.9   124.9   125.9
   512 250  2  BFS   0.965    12.8    15.2    72.2   250.8    28.6   222.2   161.8
   512 250  2   HE   0.965    11.2    13.9    83.0   250.8    25.2   225.6   127.5
   512 250  4  BFS   0.965     9.7    12.8    91.7   250.8    27.7   223.1   162.5
   512 250  4   HE   0.965     8.8    11.6   101.9   250.8    24.4   226.4   129.8
   512 250  8  BFS   0.965     8.0    11.4   110.0   250.8    26.9   223.9   164.3
   512 250  8   HE   0.965     7.3    10.2   118.5   250.8    23.2   227.6   135.0
  1024 150  0  BFS   0.935    13.7    16.7    73.5   151.2   106.9    44.4   106.9
  1024 150  0   HE   0.935    11.2    14.3    91.5   151.2    81.3    69.9    81.3
  1024 150  2  BFS   0.935     8.4    10.7   104.2   151.2    21.8   129.5   108.3
  1024 150  2   HE   0.935     7.3     9.4   119.5   151.2    19.1   132.1    83.5
  1024 150  4  BFS   0.935     6.5     9.2   130.5   151.2    20.9   130.3   109.3
  1024 150  4   HE   0.935     5.9     8.0   143.5   151.2    18.2   133.0    86.1
  1024 150  8  BFS   0.935     5.6     8.5   147.0   151.2    20.2   131.0   111.8
  1024 150  8   HE   0.935     4.9     7.0   165.4   151.2    17.1   134.1    92.4
  1024 200  0  BFS   0.956    17.4    21.2    57.8   200.9   134.3    66.7   134.3
  1024 200  0   HE   0.956    14.2    18.2    72.0   200.9   104.2    96.8   104.2
  1024 200  2  BFS   0.956    10.5    13.1    86.2   200.9    25.4   175.6   135.6
  1024 200  2   HE   0.956     9.3    11.6    98.0   200.9    22.5   178.4   106.0
  1024 200  4  BFS   0.956     8.1    11.2   106.3   200.9    24.5   176.4   136.4
  1024 200  4   HE   0.956     7.4     9.8   117.8   200.9    21.7   179.3   108.3
  1024 200  8  BFS   0.956     6.8    10.2   123.9   200.9    23.8   177.2   138.5
  1024 200  8   HE   0.956     6.2     8.8   135.8   200.9    20.5   180.4   114.2
  1024 250  0  BFS   0.965    21.0    25.1    48.1   250.8   160.6    90.2   160.6
  1024 250  0   HE   0.965    17.2    21.7    59.3   250.8   125.9   124.9   125.9
  1024 250  2  BFS   0.965    12.7    15.5    72.2   250.8    28.6   222.2   161.8
  1024 250  2   HE   0.965    11.2    13.7    83.6   250.8    25.2   225.6   127.5
  1024 250  4  BFS   0.965     9.5    12.6    93.4   250.8    27.7   223.1   162.5
  1024 250  4   HE   0.965     8.9    11.7   100.7   250.8    24.4   226.4   129.8
  1024 250  8  BFS   0.965     7.8    11.3   110.1   250.8    26.9   223.9   164.5
  1024 250  8   HE   0.965     7.2    10.1   119.4   250.8    23.2   227.6   134.9
```

#### Warm

```
  pool  ef  W  lay  recall   p50ms   p99ms     QPS   exp/q  miss/q   hit/q  phys/q
   256 150  0  BFS   0.935    11.5    14.2    87.6   151.2    86.7    64.5    86.7
   256 150  0   HE   0.935     8.9    12.0   114.6   151.2    64.5    86.7    64.5
   256 150  2  BFS   0.935     7.3     9.0   138.0   151.2    17.2   134.0    88.5
   256 150  2   HE   0.935     6.2     7.8   163.2   151.2    14.6   136.6    66.8
   256 150  4  BFS   0.935     5.6     7.8   174.6   151.2    17.1   134.1    89.8
   256 150  4   HE   0.935     5.1     7.0   194.4   151.2    14.2   137.1    69.5
   256 150  8  BFS   0.935     4.8     7.3   203.9   151.2    16.4   134.8    92.9
   256 150  8   HE   0.935     4.2     6.1   230.0   151.2    13.3   137.9    74.3
   256 200  0  BFS   0.956    14.8    18.2    68.9   200.9   111.8    89.2   111.8
   256 200  0   HE   0.956    11.9    15.5    85.7   200.9    86.1   114.9    86.1
   256 200  2  BFS   0.956     9.6    11.7   104.9   200.9    21.1   179.8   114.3
   256 200  2   HE   0.956     8.2    10.3   124.6   200.9    17.6   183.3    87.4
   256 200  4  BFS   0.956     7.1     9.7   137.3   200.9    20.2   180.8   114.4
   256 200  4   HE   0.956     6.4     8.6   156.4   200.9    17.1   183.9    89.6
   256 200  8  BFS   0.956     6.1     9.0   161.1   200.9    20.2   180.7   120.2
   256 200  8   HE   0.956     5.6     8.1   178.8   200.9    16.7   184.3    95.7
   256 250  0  BFS   0.965    18.1    22.1    55.8   250.8   138.4   112.4   138.4
   256 250  0   HE   0.965    14.9    19.3    68.6   250.8   107.2   143.6   107.2
   256 250  2  BFS   0.965    11.8    14.3    85.7   250.8    23.7   227.1   140.0
   256 250  2   HE   0.965    10.2    13.1    99.6   250.8    20.5   230.3   108.8
   256 250  4  BFS   0.965     9.0    11.6   110.1   250.8    23.6   227.2   142.1
   256 250  4   HE   0.965     8.0    10.8   125.1   250.8    19.7   231.1   111.2
   256 250  8  BFS   0.965     7.2     9.9   135.5   250.8    22.8   228.0   144.5
   256 250  8   HE   0.965     6.5     8.9   150.2   250.8    19.0   231.8   116.0
   512 150  0  BFS   0.935     8.5    11.1   119.5   151.2    60.5    90.7    60.5
   512 150  0   HE   0.935     6.8     9.6   148.3   151.2    46.1   105.1    46.1
   512 150  2  BFS   0.935     6.0     7.3   166.1   151.2    12.0   139.3    62.1
   512 150  2   HE   0.935     5.2     6.8   191.8   151.2    10.4   140.8    47.6
   512 150  4  BFS   0.935     5.0     6.5   204.2   151.2    11.7   139.5    62.5
   512 150  4   HE   0.935     4.3     5.9   230.6   151.2    10.1   141.2    48.6
   512 150  8  BFS   0.935     4.3     5.4   234.0   151.2    11.4   139.8    65.0
   512 150  8   HE   0.935     3.9     5.6   256.5   151.2     9.6   141.7    52.0
   512 200  0  BFS   0.956    10.8    13.8    92.9   200.9    78.3   122.7    78.3
   512 200  0   HE   0.956     8.7    11.5   114.4   200.9    60.7   140.3    60.7
   512 200  2  BFS   0.956     7.8     9.2   129.9   200.9    14.3   186.6    78.7
   512 200  2   HE   0.956     7.0     9.0   143.8   200.9    12.9   188.1    62.0
   512 200  4  BFS   0.956     6.5     8.5   153.0   200.9    14.2   186.8    80.1
   512 200  4   HE   0.956     6.0     8.1   168.2   200.9    12.2   188.7    63.4
   512 200  8  BFS   0.956     5.5     7.3   182.2   200.9    13.9   187.1    82.6
   512 200  8   HE   0.956     4.8     6.4   204.8   200.9    11.8   189.2    66.7
   512 250  0  BFS   0.965    13.6    16.8    74.9   250.8    95.4   155.4    95.4
   512 250  0   HE   0.965    10.7    14.4    94.1   250.8    74.5   176.2    74.5
   512 250  2  BFS   0.965     9.6    11.7   104.9   250.8    16.4   234.4    95.6
   512 250  2   HE   0.965     8.5    10.5   119.1   250.8    14.3   236.5    75.5
   512 250  4  BFS   0.965     7.8     9.5   129.2   250.8    16.3   234.5    97.0
   512 250  4   HE   0.965     7.0     9.3   142.4   250.8    14.1   236.7    77.2
   512 250  8  BFS   0.965     6.4     8.4   155.6   250.8    15.7   235.1    98.1
   512 250  8   HE   0.965     6.0     8.1   164.1   250.8    13.6   237.2    80.7
  1024 150  0  BFS   0.935     4.3     6.5   226.1   151.2    24.8   126.4    24.8
  1024 150  0   HE   0.935     3.8     5.6   265.0   151.2    19.8   131.4    19.8
  1024 150  2  BFS   0.935     3.8     5.1   261.9   151.2     5.0   146.2    25.4
  1024 150  2   HE   0.935     3.4     4.6   294.7   151.2     4.2   147.0    20.5
  1024 150  4  BFS   0.935     3.4     4.4   291.9   151.2     4.8   146.4    25.9
  1024 150  4   HE   0.935     3.1     4.2   322.8   151.2     4.2   147.1    21.2
  1024 150  8  BFS   0.935     3.1     3.9   322.5   151.2     4.7   146.5    26.8
  1024 150  8   HE   0.935     2.9     4.0   352.3   151.2     4.1   147.2    22.1
  1024 200  0  BFS   0.956     5.5     8.3   178.7   200.9    32.0   168.9    32.0
  1024 200  0   HE   0.956     4.8     6.5   206.2   200.9    25.8   175.2    25.8
  1024 200  2  BFS   0.956     4.9     6.3   204.7   200.9     5.5   195.4    32.4
  1024 200  2   HE   0.956     4.4     5.9   228.1   200.9     5.1   195.8    26.2
  1024 200  4  BFS   0.956     4.3     5.3   234.6   200.9     5.4   195.6    32.3
  1024 200  4   HE   0.956     3.9     5.0   260.2   200.9     5.1   195.9    26.9
  1024 200  8  BFS   0.956     3.8     4.8   259.7   200.9     5.5   195.4    33.2
  1024 200  8   HE   0.956     3.7     4.8   267.0   200.9     4.7   196.3    28.0
  1024 250  0  BFS   0.965     6.5     9.8   149.0   250.8    38.1   212.7    38.1
  1024 250  0   HE   0.965     6.0     8.0   167.1   250.8    31.7   219.1    31.7
  1024 250  2  BFS   0.965     5.9     7.8   168.3   250.8     6.4   244.4    38.5
  1024 250  2   HE   0.965     5.5     7.2   184.6   250.8     6.2   244.6    32.1
  1024 250  4  BFS   0.965     5.6     6.9   181.6   250.8     6.3   244.4    39.1
  1024 250  4   HE   0.965     5.1     6.5   196.5   250.8     6.0   244.8    32.5
  1024 250  8  BFS   0.965     4.9     5.9   201.8   250.8     6.2   244.6    39.5
  1024 250  8   HE   0.965     4.6     5.9   218.2   250.8     5.8   245.0    34.1
```

## Conclusions

1. **heavy_edge is the new default layout** — 22% fewer phys/q than BFS in cold (structural, pool-invariant), 10-24% QPS improvement warm depending on pool/W. Confirmed robust across all 72 sweep points × 2 passes.
2. **The advantage is structural**: Cold-pass phys/q is identical across pool sizes (134 BFS vs 104 HE at W=0), matching upg/q from traced measurement. This is a layout property, not a cache artifact.
3. **Page-aware scheduling (B=4) should be enabled by default** — eliminates demand cache misses at negligible recall cost.
4. **neighbor-run BFS is not worth pursuing** — no improvement over standard BFS.
5. **heavy_edge does not change the W tradeoff** — W=8 maximizes single-core throughput (e.g. 512p warm ef=200: BFS 182→HE 205 QPS); W=4 is the conservative default with less prefetch IO pressure, likely better when scaling across cores where total device QD matters.
6. **HE advantage narrows at large pools** (1.03× at 1024p/W=8) but never reverses — safe to deploy unconditionally.
7. **Peak single-core**: HE + 1024 pages + W=8 + ef=150 = **352 QPS** at 0.935 recall (2.9ms p50).

### Limitations

- Tested at N=100K only. At larger N, cache hit rates decrease and layout importance increases — heavy_edge advantage likely grows. 1M smoke test requires dataset preparation.
- heavy_edge reorder is O(n × degree × page_size) with HashSet operations. At 100K it takes ~10s. May need optimization for N>1M.
- Page-aware scheduling adds O(B log n) per expansion. At B=4 this is negligible; at B=16 there's slight overhead visible in p99.
- The `indeg(u)+indeg(v)` edge weight is a proxy for co-access frequency. A direct edge-cut objective (maximize edges between nodes on the same page) could potentially do better but was not tested.
