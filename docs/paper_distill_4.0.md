# Paper Distillation 4.0 — HNSW / Graph ANNS

Four papers mapped to our codebase. Ordered by actionable impact.

---

## Papers

| # | Paper | Key Idea |
|---|-------|----------|
| 1 | **FlatNav** (ICML'25 Workshop) — "The H in HNSW Stands for Hubs" | Flat NSW = HNSW for d≥32. Hub nodes form a natural highway. |
| 2 | **Ada-ef** (SIGMOD'26) — Distribution-Aware Adaptive HNSW Search | Per-query ef via Gaussian FDL theory. No ML, pure stats. |
| 3 | **VSAG** (VLDB) — Ant Group Production Framework | Deterministic access, PRS (inline codes), edge labeling, auto-tune. |
| 4 | **PiPNN** (Google, 2026) — Ultra-Scalable Graph Construction | Search-free build: partition→GEMM→HashPrune. 10x faster than Vamana. |

---

## Tier 1: Validates What We Already Have

### 1a. Flat graph is a sound engineering choice (FlatNav)

Our `NswBuilder` produces a single-layer NSW graph with hub entry_set
(selected by degree, `nsw.rs:420`). FlatNav shows that for d≥32, flat NSW
achieves **comparable** latency and recall to full hierarchical HNSW on their
benchmarks, with 18-39% less memory on 100M scale.

This is strong evidence that hierarchy is **not worth the complexity** for our
target dimensions (d≥96). But "comparable" ≠ "mathematically identical" — our
specific construction heuristic + bounded degree + hub entry_set is not the
same thing FlatNav evaluated (they extracted hnswlib's base layer). The claim
is empirical, not a proof.

**Decision**: Don't implement hierarchical layers — the engineering cost is
high and the evidence says the payoff is negligible at d≥96.

**Caveat**: We are explicitly giving up hierarchy as an IO-reduction lever
(see `io_reduction_research.md:170` — it IS one of only two known paths to
<100 blk/q). Our path to winning IO is per-query stopping + inline codes +
layout, not hierarchy. Acknowledge the tradeoff.

### 1b. SQ is the right graph-traversal distance (VSAG)

VSAG ablation: SQ (Int8/Int4) is better suited than PQ for graph-based
traversal. PQ-Fast Scan wastes SIMD bandwidth on visited nodes and has random
storage patterns incompatible with graph access. SQ directly compresses
dimensions, fully utilizes SIMD.

**Decision**: Our Int8 SQ approach is correct for **CPU-side compute
speedup**. But Int8 is NOT an IO optimization — our own EXP-0 proved this:
at iso-recall, blk_ratio = 1.00 (`io_reduction_research.md:59-67`). Both
precisions need the same ef and therefore the same blocks. Int8's 1.58x
speedup is purely CPU-side and won't survive NVMe IO latency.

PQ is still relevant for **inline codes** in adjacency blocks (Tier 3a) —
small PQ codes for approximate next-hop scoring is a different use case than
primary graph distance.

### 1c. Prefetch scheduling hides latency (VSAG)

VSAG ablation: basic software prefetch = 17% QPS boost; stride prefetch adds
marginal improvement; deterministic access (batch-filter visited before
prefetch) = 70% QPS on GIST1M. Our prefetch_hint + prefetch_worker already
implements this at the IO level (`cache.rs:692`, `search.rs:475-486`).

**Important caveat**: Prefetch hides latency but does NOT reduce blk/q. We
measured this: blk/q stays at 201 regardless of W (`EXP-P1`). The fundamental
IO count is determined by ef and graph structure. Prefetch is a necessary
optimization but not the path to <100 blk/q.

---

## Tier 2: High-Value, Implementable Now

### 2a. Query-Adaptive ef via FDL Distribution (Ada-ef) ★★★

**The most directly applicable insight for reducing blk/q.**

**Theory**: For high-d learned embeddings, the Full Distance List `FDL(q,V)`
follows a Gaussian distribution (via CLT). Parameters computable per-query:

```
mu_CS = dot(q_hat, mean_of_V_hat)          // d multiplies
sigma_CS² = q_hat * Cov_V_hat * q_hat^T    // d² multiplies (or d if diagonal)
FDL_cosine ~ N(1 - mu_CS, sigma_CS²)
```

**Offline** (one-time, <5% of build time):
- Compute column mean of V (d floats)
- Compute covariance matrix of V (d×d floats; for d=768 → 2.4MB)
- Sample 200 vectors as proxy queries → build (score → ef) lookup table

**Online** (per-query, microseconds):
1. During first 2 hops of search (already happening), collect distance samples
2. Compute query score: bin distances into Gaussian quantiles, exponential-weighted sum
3. Lookup ef from table → set as search budget

**Impact**: Converts our fixed blk/q ≈ ef (201 blocks) into adaptive:
- Easy queries (majority): ef=80-120 → 40-60% IO reduction
- Hard queries (long tail): ef=300+ → better recall
- Ada-ef achieves **4x latency reduction** vs static ef at same recall

**Combines with our adaptive stopping**: Ada-ef estimates query difficulty
early → selects per-query (ef, stall_limit, drain_budget) triple. Stall
detector then exits early within that budget. Double savings.

**Key parameters**: m=5 bins, delta=0.001, 200 offline samples,
exponential weights w_i = 100·e^(-i+1), 2-hop distance collection.

**Implementation sketch** (~200 lines):
1. `struct FdlStats { mean: Vec<f32>, cov_diag: Vec<f32> }` (start diagonal-only)
2. `fn estimate_difficulty(query, distances_2hop, stats, table) -> (usize, u32, u32)`
   returns `(ef, stall_limit, drain_budget)` triple
3. Integrate into `disk_graph_search_pipe`: after entry_set + first few
   expansions, call estimate_difficulty. Set ef as the capacity going forward
   (do NOT resize FixedCapacityHeap mid-search — `clear(new_capacity)` would
   lose accumulated results). Instead, **start with a large ef and shrink the
   termination condition**, or start from the estimated ef directly. The
   simplest correct integration: estimate difficulty from entry_set distances
   BEFORE the main expansion loop begins, then pass the chosen ef to the
   heap constructor.

### 2b. Hub Pinning in AdjacencyPool (FlatNav) ★★

**Finding**: Hub nodes (top 1-5% by access frequency) are visited in 40-70%
of the first 5-10% of search steps. They form a "highway" subgraph — queries
route through hubs first, then converge to local neighborhoods.

**Opportunity**: Our AdjacencyPool uses clock eviction. Hot hub blocks may be
evicted under pressure despite being reused across nearly every query.

**Implementation** (must match actual cache state machine):
1. At build time: run 1000 random queries, count per-node access frequency
2. Mark top 1% (or top-N by degree) as hub nodes in metadata
3. Add explicit `hub_pinned: bool` field to cache entries (or a reserved
   `pin_count` value). The clock `referenced` bit alone is NOT sufficient —
   it gets cleared on sweep. True pinning requires the eviction loop to
   **skip entries where `hub_pinned == true`**, same as it already skips
   `LOADING` and `pin_count > 0` entries (`cache.rs:717`).
4. On engine startup: pre-load hub adjacency blocks into the pool

**Expected impact**: With 5% cache (pool_bytes = n/20 * 4096), hub pinning
would ensure the ~1000 most-accessed blocks never miss. Given hubs account
for 40-70% of early-step visits, this converts the highest-frequency IO to
guaranteed cache hits.

**Cost**: ~1% of cache capacity dedicated to hubs. Negligible.

### 2c. Truncated Scalar Quantization — p99 range (VSAG)

**Finding**: Using absolute min/max for SQ range wastes quantization
resolution. On GIST1M, 99% of values < 0.3 but max = 1.0 → 70% of range
unused. Using p99 as the quantization boundary preserves resolution.

**Our situation**: We use uniform scale=1 for pre-normalized cosine vectors
(all components in [-1,1]), quantize as `round(x * 127)`. Per-dim scale
was tested and recall dropped to 0.377 — but that failure might be a
**pipeline inconsistency bug** (query vs vector scale/offset mismatch),
not proof that per-dim is fundamentally wrong.

**Status**: Needs investigation, not a settled question.
- p99 truncation is a different technique from per-dim scale. It clips
  outliers to use more of the int8 range for common values.
- For pre-normalized cosine, components are bounded [-1,1] but the actual
  distribution may be much tighter. Check empirically on Cohere embeddings.
- The per-dim recall failure should be re-examined for implementation bugs
  before writing it off as a design constraint.

---

## Tier 3: Medium-Term Architecture Improvements

### 3a. Inline Compressed Codes (VSAG PRS + PageANN)

VSAG's PRS = PageANN's inline compressed neighbors. Co-locate compressed
neighbor codes with adjacency list in the same block. Enables zero-IO
approximate scoring for next-hop candidates.

Our adjacency block wastes most of its 4KB:
```
Current:  [degree(u16) | padding(6B) | neighbor_vids(u32 × 32)] = 136 bytes used, 3960 wasted
```

**The hard constraint** (`io_reduction_research.md:267-272`): at d=768, int8
codes are 768 bytes/neighbor. With degree=32: 24KB. **Does NOT fit in 4KB.**

Viable inline code options:
- **PQ16×8 = 16 bytes/neighbor** → 32 × 16 = 512B → fits easily with room to spare
- **RaBitQ / 4-bit = ~96 bytes/neighbor** → 32 × 96 = 3072B → fits in 4KB (tight)
- **Top-N only**: store codes for 16 of 32 neighbors → halves cost

**Do NOT write "PQ 64B" as the baseline** — that's 2048B for 32 neighbors,
which fits but leaves no room for edge labels or other metadata. Prefer the
smallest code that gives usable approximate scoring. PQ16×8 (16B) is the
sweet spot: 512B total, leaves 3352B for labels/metadata/future use.

This is the planned **Opt-A**. Depends on implementing a PQ codebook first.

### 3b. Edge Labeling for Parameter Tuning (VSAG)

**Theorems**: Given fixed construction, G^a (degree=a) ⊂ G^b (degree=b)
for a<b. Build once with max degree, annotate each edge with minimal
alpha_e needed to retain it. At search time, filter edges by threshold.

**Practical benefit**: Build a single index with m_max=64. At search time,
use m_s=32 (filter to effective degree 32) or m_s=16 for fast approximate
search. No rebuild needed. 19x tuning time savings.

**Block layout budget conflict**: Edge labels and inline codes both compete
for the 4KB block. Per-edge label = 1 byte (alpha quantized to u8).
32 edges × 1B = 32 bytes. This is negligible — labels and PQ16 codes can
coexist: 136B (vids) + 32B (labels) + 512B (PQ16 codes) = 680B, well within
4KB. But the block layout must be designed holistically — not piecemeal.

### 3c. Search-Free Construction (PiPNN)

**Problem PiPNN solves**: Incremental HNSW/Vamana construction requires
O(n) beam searches, each doing random LLC misses. At 1B scale, this takes
hours-days.

**Solution**: Partition → cache-friendly leaf GEMM → HashPrune → RobustPrune.
10x faster, 10x fewer LLC misses, IPC 1.26 vs 0.44 (Vamana) vs 0.33 (HNSW).

**Key algorithm (HashPrune)**: History-independent edge pruning via
residualized LSH. 8 bytes per slot (4B ID + 2B hash + 2B bf16 distance).
12-bit hash, reservoir capacity = max degree. O(log l) insertion.

**Leaf parameters**: C_max=1024-2048, k=2 bi-directed, GEMM for all-pairs.

**Relevance**: Our NswBuilder works for 100K. At 1M-100M, it becomes the
bottleneck. PiPNN's approach would let us build 1B-scale indexes in <30min
on 192 cores. But this is future work — not needed until we scale beyond
current Cohere 100K testbed.

**RobustPrune as final pass**: Always add a final RobustPrune after any
construction method. +10% QPS at +10% build cost. Low-hanging fruit even
for our current NswBuilder output.

---

## Tier 4: Insights That Do NOT Apply (with caveats)

| Insight | Why Not | Caveat |
|---------|---------|--------|
| HNSW hierarchical layers | Not worth the complexity at d≥96. | But it IS one of two known paths to <100 blk/q (`io_reduction_research.md:170`). We're betting on per-query stopping + inline codes instead. |
| PRS hardware prefetch (VSAG) | CPU cache-line optimization. Our prefetch is IO-level (4KB blocks). | — |
| VSAG's GBDT query classifier | Heavy offline training. Ada-ef achieves same goal with pure statistics, 50x less compute. | — |
| PiPNN's build speed | Not a bottleneck at 100K. | Matters at 1M+. |
| Ada-ef's incremental covariance update | Matters for mutable indexes. We're append-only for now. | — |
| VSAG stride prefetch omega | CPU cache-line tuning. Not applicable to 4KB IO blocks. | — |

---

## Priority Ranking for Implementation

| Priority | Feature | Source | What It Reduces | Effort |
|----------|---------|--------|----------------|--------|
| **P0** | Query-adaptive ef (FDL) | Ada-ef | blk/q (40-60% for easy queries) | ~200 LOC |
| **P1** | Hub pinning in cache | FlatNav | cache miss rate on hot nodes | ~80 LOC |
| **P2** | Inline PQ16 codes (Opt-A) | VSAG+PageANN | blk/q (score before expand) | ~500 LOC + PQ impl |
| **P3** | Edge labeling | VSAG | per-query degree tuning | ~100 LOC |
| **P4** | RobustPrune final pass | PiPNN | graph quality (+10% QPS) | ~50 LOC |
| **P5** | Search-free construction | PiPNN | build time at scale | ~800 LOC |

**The path to <100 blk/q at recall≥0.96:**
- P0 (Ada-ef) reduces blk/q for easy queries: 201 → ~100-120 on median
- P2 (inline PQ16 codes) enables score-before-expand: skip ~50% of expansions
- P1 (hub pinning) converts the most frequent misses to guaranteed hits
- Combined: 3-4x effective IO reduction from current baseline

**What we're explicitly NOT doing**: hierarchy (complex, marginal for d≥96),
PQ as primary graph distance (SQ is better for traversal), any ML-based
query classification (Ada-ef's stats approach is lighter and sufficient).
