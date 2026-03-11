# Topology-Based Page Packing: 3 Experiments

## Context

TWPP (trace-weighted page packing) was a negative result: co-expansion patterns learned
from query traces don't generalize to unseen queries (21.5% VID coverage, query-specific
graph paths in 768d cosine). BFS reorder is the current baseline.

These 3 experiments use **static graph structure only** — no query traces, no learning.
They should generalize because they exploit universal properties of the graph, not
query-dependent access patterns.

All experiments use the existing v3 framework: `write_packed_adjacency`, `load_adj_index`,
`AdjacencyPool`, `IoDriver`, `disk_graph_search_pipe_v3`. Same dataset (Cohere 100K,
dim=768, cosine, k=100), same holdout evaluation protocol from hardened TWPP experiment.

---

## Experiment 3: Page-Aware Expansion Scheduling (no relayout)

### Idea

Even with the same BFS layout, reduce physical reads by biasing expansion order: among
the top-B candidates in the heap, prefer the one whose page is already resident in cache.
This is a search-time optimization — no build-time or layout changes.

### Why it might work

When multiple candidates have similar distances, expanding a resident-page candidate first
is free (cache hit). The non-resident candidate gets expanded later, by which time its
prefetch may have completed. This converts "miss then hit" into "hit then hit" without
sacrificing result quality — as long as the distance deviation is bounded.

### Algorithm

In `disk_graph_search_pipe_v3_inner`, replace the simple `candidates.pop()` with:

```
1. Pop top-B candidates from heap (B=8, same buffer as prefetch lookahead)
2. Among them, find the best-distance candidate whose page_id is resident
3. If found, expand that one; push the rest back
4. If none are resident, expand the best-distance candidate (normal behavior)
```

Bounded deviation: we only consider the top-B candidates, so the expanded candidate
is at most B positions away from the true best. With B=8 and ef=200, this is a 4%
deviation in rank — negligible for recall but potentially significant for cache hits.

### Implementation

**File**: `crates/engine/src/search.rs`

New function: `disk_graph_search_pipe_v3_pagesched`

The core change is ~15 lines replacing the `candidates.pop()` call:

```rust
// Page-aware candidate selection: among top-B, prefer resident pages
let candidate = if page_sched_b > 0 {
    let b = page_sched_b.min(8);
    let count = candidates.peek_nearest(&mut lookahead[..b]);

    // Find best-distance candidate with resident page
    let mut best_resident: Option<(usize, f32)> = None;
    for i in 0..count {
        let cand_vid = lookahead[i].id.0 as usize;
        if cand_vid < num_vectors {
            let pid = adj_index[cand_vid].page_id;
            if pool.is_resident(pid) {
                if best_resident.is_none() || lookahead[i].distance < best_resident.unwrap().1 {
                    best_resident = Some((i, lookahead[i].distance));
                }
            }
        }
    }

    match best_resident {
        Some((idx, _)) => {
            // Pop the chosen candidate; push others back
            // peek_nearest already pushed them all back, so we need to
            // pop all, take idx, push rest back
            let chosen = lookahead[idx];
            // Remove chosen from heap — it was already pushed back by peek_nearest
            // We need a remove-by-id or a different approach
            // Simpler: pop one-by-one, skip the chosen, push rest back
            ...
        }
        None => candidates.pop().unwrap(),
    }
} else {
    candidates.pop().unwrap()
};
```

**Problem with CandidateHeap API**: `peek_nearest` pops-and-pushes-back. To select a
non-top-1 candidate, we need to pop multiple, take one, push the rest back. This is
O(B log N) — fine for B=8.

Better approach — add `pop_preferred` to CandidateHeap:

```rust
/// Pop the nearest candidate whose page is resident, looking at up to `b` candidates.
/// If none are resident, pops the nearest (distance-best).
/// Returns (candidate, was_resident).
pub fn pop_page_preferred(
    &mut self,
    b: usize,
    is_resident: impl Fn(u32) -> bool, // vid -> page resident?
) -> Option<(ScoredId, bool)>
```

Implementation: pop up to `b`, scan for first/best resident, pick it, push rest back.

### New perf counter

Add `page_sched_hits: u64` to `SearchPerfContext` — counts how many times a non-top-1
candidate was chosen because its page was resident. This directly measures the benefit.

### Benchmark

Add to `exp_twpp` (rename to `exp_topology_packing` or add as separate `exp_page_sched`):
- BFS layout, holdout queries [200..300), warmup [300..320)
- Compare B=0 (baseline) vs B=4 vs B=8 vs B=16
- Metrics: recall, p50, p99, phys/q, miss/q, page_sched_hits/q

### Risk

Recall loss from distance deviation. With B=8, we might expand a candidate that's
0.001 further away than the true best. In practice the top-8 candidates are often
near-identical in distance (all on the convergence frontier), so the deviation should
be negligible.

---

## Experiment 1: Neighbor-Aware BFS Packing

### Idea

Standard BFS reorder assigns consecutive new VIDs in BFS layer order. This puts
BFS-nearby nodes on the same page, but doesn't specifically ensure that a node's
**direct neighbors** share its page.

Neighbor-aware packing modifies BFS to greedily co-locate neighbors: when visiting
node u, immediately try to place u's unassigned neighbors on the same page as u.

### Why it might work

When node u is expanded, the search reads u's adjacency record and discovers u's
neighbors. The most likely next expansions are u's neighbors (they're close in the
graph AND close in distance to the query). If those neighbors' adjacency records
are on the same page, the page is already cached — the next expansion is a cache hit.

This is the key insight from PageANN: pack adjacency records of graph-neighbors
together. Unlike TWPP, this is a **structural** property — it holds for ALL queries,
not just queries you've seen before.

### Algorithm

```
neighbor_aware_reorder(n, entry_set, neighbors_fn) -> Vec<u32>:
    old_to_new = [MAX; n]
    queue = deque()
    next_id = 0
    page_budget = 4096  // bytes remaining in current page

    // Seed BFS from entry points
    for ep in entry_set:
        if not assigned(ep):
            assign(ep, next_id++)
            queue.push_back(ep)

    while queue not empty:
        u = queue.pop_front()

        // Greedily assign u's unassigned neighbors to current page
        for nbr in neighbors(u):
            if not assigned(nbr):
                record_size = packed_record_size(degree(nbr))
                if record_size <= page_budget:
                    assign(nbr, next_id++)
                    page_budget -= record_size
                    queue.push_back(nbr)
                // else: skip, will be assigned later when it's popped from queue

        // If page is full or u exhausted, check if we need a new page
        // New page starts when next node's record won't fit
        // (handled implicitly by write_packed_adjacency's page-filling logic)

    // Unreachable nodes get sequential assignment
    for i in 0..n:
        if not assigned(i):
            assign(i, next_id++)

    old_to_new
```

**Key difference from BFS**: BFS assigns new VIDs in strict BFS queue order. This
variant assigns u's neighbors eagerly when visiting u, before moving to the next
BFS queue entry. This means u's neighbors get consecutive VIDs — and consecutive
VIDs get packed onto the same page by `write_packed_adjacency`.

**Subtlety**: the reorder only controls the VID assignment. The actual page packing
is done by `write_packed_adjacency` which fills pages greedily in new-VID order.
So to co-locate neighbors, we need their new VIDs to be consecutive. The algorithm
above achieves this by assigning neighbors immediately.

### Implementation

**File**: `crates/storage/src/adjacency.rs`

New function: `neighbor_aware_reorder_graph`

```rust
pub fn neighbor_aware_reorder_graph<'a>(
    n: usize,
    entry_set: &[u32],
    neighbors_fn: impl Fn(u32) -> &'a [u32],
) -> Vec<u32>
```

~30 lines. Same signature as `bfs_reorder_graph`. The key difference is the inner
loop that eagerly assigns neighbors before continuing BFS.

**Note**: We need degree info to track page budget. Pass `neighbors_fn` which gives
both neighbors and degree (degree = neighbors.len()).

### Benchmark

Same holdout protocol. Compare:
- `sequential` (identity reorder, control)
- `bfs` (current baseline)
- `neighbor_aware` (this experiment)

Metrics: recall (must be identical), p50, p99, phys/q, miss/q, unique_pages/q.

### Risk

Lower: this is a pure reorder, no recall impact. The only question is magnitude.
With avg degree ~26 and page capacity ~38 records, a node + all its neighbors often
fit in one page. The question is whether BFS already achieves most of this co-location
by accident (BFS neighbors tend to be assigned nearby VIDs).

---

## Experiment 2: Heavy-Edge Packing (MARGO-style)

### Idea

Weight graph edges by structural importance, then greedily pack pages to maximize
total edge weight within each page. Edges between nodes on the same page represent
"free" neighbor lookups (page already cached when expanding either endpoint).

### Edge weight: `indeg(v)` (hub attraction)

Start with the simplest useful weight: `w(u->v) = indeg(v)`. Edges pointing into
high-in-degree nodes (hubs) are more important because hubs appear in many search
paths. Undirected weight: `w_u(u,v) = indeg(u) + indeg(v)`.

This is query-agnostic and captures the key MARGO insight: prioritize edges on
likely-traversed paths. Hubs are structurally identifiable without any queries.

### Algorithm

```
heavy_edge_reorder(n, neighbors_fn) -> Vec<u32>:
    // 1. Compute in-degrees
    indeg = [0; n]
    for u in 0..n:
        for v in neighbors(u):
            indeg[v] += 1

    // 2. Build edge list with undirected weights, sorted descending
    edges = []
    for u in 0..n:
        for v in neighbors(u):
            if u < v:  // undirected: count each edge once
                w = indeg[u] + indeg[v]
                edges.push((w, u, v))
    edges.sort_descending_by_weight()

    // 3. Greedy page packing
    old_to_new = [MAX; n]
    next_id = 0
    page_budget = 4096

    for (w, u, v) in edges:
        // Try to place both endpoints on current page
        for node in [u, v]:
            if not assigned(node):
                rec = packed_record_size(degree(node))
                if rec <= page_budget:
                    assign(node, next_id++)
                    page_budget -= rec
                else:
                    // Page full, start new page
                    page_budget = 4096
                    if rec <= page_budget:
                        assign(node, next_id++)
                        page_budget -= rec

    // 4. Remaining unassigned nodes: fill sequentially
    for i in 0..n:
        if not assigned(i):
            assign(i, next_id++)

    old_to_new
```

**MARGO's "don't add non-neighbors" rule**: The edge-seeded approach naturally
follows this. Each page is seeded by a heavy edge (u,v), and then filled by other
endpoints of heavy edges that share a node with u or v. This produces pages that
are locally dense subgraphs — exactly what we want.

### Implementation

**File**: `crates/storage/src/adjacency.rs`

New function: `heavy_edge_reorder_graph`

```rust
pub fn heavy_edge_reorder_graph<'a>(
    n: usize,
    neighbors_fn: impl Fn(u32) -> &'a [u32],
) -> Vec<u32>
```

~50 lines. The edge sort is O(E log E) where E = sum of degrees ~ n * avg_deg.
For 100K nodes with avg degree 26, E ~ 2.6M. Sort takes < 1 second.

### Benchmark

Same holdout protocol. Compare all 4 reorders:
- `sequential` (control)
- `bfs` (current baseline)
- `neighbor_aware` (Experiment 1)
- `heavy_edge` (this experiment)

### Risk

Medium. The edge-sorted approach might produce fragmented pages (high-weight edges
scattered across the graph → each page has just 2-3 related nodes, rest are filler).
The "page seed + greedy fill" variant below might work better:

**Variant**: Instead of iterating edges, iterate pages. For each page:
1. Seed with the unassigned node with highest in-degree
2. Greedily add the unassigned neighbor of any page member that maximizes
   `sum of w_u to existing page members` (same as TWPP but with static weights)
3. Only consider graph neighbors of page members (MARGO rule)

This variant is more expensive (O(n * avg_page_size * avg_degree)) but produces
denser pages. Implement this variant if the simple edge-walk underperforms.

---

## Shared Benchmark Infrastructure

### Test function: `exp_topology_packing`

Single test that runs all experiments. Structure:

```
Phase 0: Build NSW index, write vectors.dat

Phase 1: Write all layout variants
  - sequential (identity)
  - bfs (current baseline)
  - neighbor_aware (Exp 1)
  - heavy_edge (Exp 2)

Phase 2: Cold benchmark (all layouts, holdout queries [200..300))
  - Full counter breakdown: recall, p50, p99, phys/q, miss/q, hit/q,
    pf_i/q, pf_c/q, sf/q, unique_pages/q
  - Pool config printed per layout

Phase 3: Warm benchmark (all layouts, warmup [300..320), measure [200..300))

Phase 4: Page-aware scheduling (Exp 3, BFS layout only)
  - B=0 (baseline), B=4, B=8, B=16
  - Additional counter: page_sched_hits/q
  - Both cold and warm

Phase 5: Best reorder + best B combined
  - If Exp 1 or 2 wins, run with page-aware scheduling on top
```

### Holdout protocol (carried from hardened TWPP)

- Queries [0..200): NOT USED (reserved for trace experiments)
- Queries [200..300): benchmark measurement
- Queries [300..320): warmup only
- Ground truth indexed accordingly

### Files changed

| File | Change | Est. LOC |
|------|--------|----------|
| `crates/storage/src/adjacency.rs` | `neighbor_aware_reorder_graph`, `heavy_edge_reorder_graph` | ~80 |
| `crates/storage/src/lib.rs` | Export new functions | 2 |
| `crates/index/src/heap.rs` | `pop_page_preferred` on CandidateHeap | ~25 |
| `crates/index/src/lib.rs` | Export | 1 |
| `crates/engine/src/search.rs` | `disk_graph_search_pipe_v3_pagesched` | ~30 |
| `crates/engine/src/perf.rs` | `page_sched_hits` counter | 3 |
| `crates/engine/src/lib.rs` | Export | 1 |
| `crates/engine/tests/disk_search.rs` | `exp_topology_packing` test | ~250 |

### Implementation order

1. **Experiment 3** (page-aware scheduling): `pop_page_preferred` + search variant + benchmark
2. **Experiment 1** (neighbor-aware BFS): reorder function + add to benchmark
3. **Experiment 2** (heavy-edge): reorder function + add to benchmark
4. Combined: best reorder + best scheduling

---

## Expected outcomes

| Experiment | Expected signal | Rationale |
|------------|----------------|-----------|
| Exp 3 (page sched) | Small (5-15% phys/q reduction) | Exploits existing page co-location; limited by BFS layout quality |
| Exp 1 (neighbor BFS) | Moderate (10-25% reduction) | Directly co-locates neighbors; BFS already partially does this |
| Exp 2 (heavy-edge) | Moderate-to-strong (15-30%) | Prioritizes high-traffic subgraphs; closest to MARGO |
| Exp 1 + Exp 3 | Compound benefit | Better layout + smarter scheduling |

These are rough estimates. The TWPP negative result teaches us that the bar is high:
BFS already captures most structural locality at this scale. Any improvement needs to
exploit a property BFS systematically misses.

### What BFS misses

BFS assigns VIDs in layer order. Within a BFS layer, nodes are assigned in the order
they were discovered (FIFO). Two nodes u, v that are neighbors but discovered from
different BFS parents may be assigned distant VIDs — they end up on different pages
even though they're structurally adjacent.

Experiment 1 addresses this directly. Experiment 2 addresses it via edge importance.
Experiment 3 works around it at search time without changing the layout.
