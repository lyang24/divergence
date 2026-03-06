# Paper Distillation 4.0 — HNSW / Graph ANNS

Four papers mapped ruthlessly to our codebase. Ordered by actionable impact.

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

### 1a. Flat graph is correct (FlatNav)

Our `NswBuilder` produces a single-layer NSW graph. FlatNav proves this is
**identical** to HNSW for d≥32 (all modern embeddings are d≥96). The hierarchy
adds memory overhead (18-39% on 100M scale) and implementation complexity for
**zero latency benefit**. Our entry_set (hub nodes selected by degree) IS the
hub highway entrance.

**Decision**: Never implement hierarchical layers. Single-layer NSW is the
architecture. This is now proven, not just our bet.

### 1b. SQ > PQ for graph search (VSAG)

VSAG ablation: SQ (Int8/Int4) is strictly better than PQ for graph-based
traversal. PQ-Fast Scan wastes SIMD bandwidth on visited nodes and has random
storage patterns incompatible with graph access. SQ directly compresses
dimensions, fully utilizes SIMD, preserves graph locality.

**Decision**: Our Int8 SQ approach is correct. Do NOT switch to PQ for
graph-level scoring. PQ may still be useful for inline codes in adjacency
blocks (approximate next-hop scoring, not primary distance).

### 1c. Prefetch scheduling is the right approach (VSAG)

VSAG ablation shows basic software prefetch gives 17% QPS boost, but stride
prefetch adds only marginal improvement. The real win is deterministic access
(batch-filter visited before prefetching) — 70% QPS boost on GIST1M. Our
prefetch_hint + prefetch_worker already implements this pattern at the IO level.

---

## Tier 2: High-Value, Implementable Now

### 2a. Query-Adaptive ef via FDL Distribution (Ada-ef) ★★★

**THE most valuable insight across all four papers for IO reduction.**

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

**Combines with our adaptive stopping**: Ada-ef sets initial ef budget,
stall detector exits early within budget. Double savings.

**Key parameters**: m=5 bins, delta=0.001, 200 offline samples,
exponential weights w_i = 100·e^(-i+1), 2-hop distance collection.

**Implementation sketch** (~200 lines):
1. `struct FdlStats { mean: Vec<f32>, cov_diag: Vec<f32> }` (start diagonal-only)
2. `fn estimate_ef(query, distances_2hop, stats, table) -> usize`
3. Integrate into `disk_graph_search_pipe`: after first 2 expansion hops,
   call estimate_ef and resize FixedCapacityHeap if ef < current capacity

### 2b. Hub Pinning in AdjacencyPool (FlatNav) ★★

**Finding**: Hub nodes (top 1-5% by access frequency) are visited in 40-70%
of the first 5-10% of search steps. They form a "highway" subgraph — queries
route through hubs first, then converge to local neighborhoods.

**Opportunity**: Our AdjacencyPool uses clock eviction. Hot hub blocks may be
evicted under pressure despite being reused across nearly every query.

**Implementation**:
1. At build time: run 1000 random queries, count per-node access frequency
2. Mark top 1% (or top-N by degree) as hub nodes in metadata
3. On engine startup: pre-load hub adjacency blocks, set `referenced=true` permanently
4. In clock eviction: skip hub-pinned entries (never evict)

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
(all components in [-1,1]). Per-dim scale failed (recall→0.377). But p99
truncation is different: it clips outliers to use more of the int8 range for
the common values.

**Applicability**: For pre-normalized cosine vectors, component values are
already bounded in [-1,1] and scale=1 maps perfectly. p99 truncation would
only help if the actual distribution is much tighter than [-1,1] (e.g.,
components mostly in [-0.3, 0.3]). **Check empirically on Cohere/Ada-002
embeddings before implementing.**

---

## Tier 3: Medium-Term Architecture Improvements

### 3a. PRS / Inline Compressed Codes (VSAG + PageANN)

VSAG's PRS (Partial Redundant Storage) = PageANN's inline compressed
neighbors. Co-locate compressed neighbor codes with adjacency list in the
same block. Transforms random vector access into sequential access within
a single IO.

Our adjacency block format:
```
[num_neighbors: u16][padding: 6B][neighbor_0: u32]...[zero-pad to 4096]
```
Max neighbors = (4096-8)/4 = 1022. We use m_max=32, so actual payload =
8 + 32*4 = 136 bytes. **3960 bytes wasted per block.**

With PQ codes (e.g., 64 bytes per vector for d=768 with m=64 subspaces):
136 bytes (current) + 32 × 64 bytes (PQ codes) = 2184 bytes. Fits in 4KB.

This enables **zero-IO approximate scoring** for next-hop candidates:
score neighbors using inline PQ codes → only issue IO for promising
candidates. Could reduce blk/q by 50%+ (similar to PageANN's results).

**This is the planned Opt-A.** VSAG's delta (redundancy ratio) gives a
tuning knob: 0=no inline codes (current), 1=all neighbors have codes.
VSAG shows the tradeoff is workload-dependent (helps when IO-bound,
hurts when CPU-bound). For NVMe, we are IO-bound → delta=1 likely optimal.

### 3b. Edge Labeling for Parameter Tuning (VSAG)

**Theorems**: Given fixed construction, G^a (degree=a) ⊂ G^b (degree=b)
for a<b. Build once with max degree, annotate each edge with minimal
alpha_e needed to retain it. At search time, filter edges by threshold.

**Practical benefit**: Build a single index with m_max=64. At search time,
use m_s=32 (filter to effective degree 32) or m_s=16 for fast approximate
search. No rebuild needed. 19x tuning time savings.

**For us**: Build with generous m_max, store alpha labels in adjacency block
headers. At search time, `alpha_s` threshold controls effective degree.
Enables fast per-query degree tuning (easy queries: m_s=16, hard: m_s=48).

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

## Tier 4: Insights That Do NOT Apply

| Insight | Why Not |
|---------|---------|
| HNSW hierarchical layers | Proven unnecessary for d≥32 (FlatNav). We never had them. |
| PRS hardware prefetch (VSAG) | Assumes DRAM-resident vectors. Our vectors ARE in DRAM but adjacency is on disk. PRS transforms L3→L1, not disk→cache. |
| VSAG's GBDT query classifier | Heavy offline training. Ada-ef achieves same goal with pure statistics, 50x less compute. |
| PiPNN's build speed | Not a bottleneck at 100K. Matters at 1M+. |
| Ada-ef's incremental covariance update | Matters for mutable indexes. We're append-only for now. |
| VSAG stride prefetch omega | CPU cache-line level optimization. Our prefetch is IO-level (4KB blocks). |

---

## Priority Ranking for Implementation

| Priority | Feature | Source | IO Reduction | Effort |
|----------|---------|--------|-------------|--------|
| **P0** | Query-adaptive ef (FDL) | Ada-ef | 40-60% for easy queries | ~200 LOC |
| **P1** | Hub pinning in cache | FlatNav | eliminate hot-path misses | ~50 LOC |
| **P2** | Inline PQ codes (Opt-A) | VSAG+PageANN | 50%+ blk/q reduction | ~500 LOC |
| **P3** | Edge labeling | VSAG | per-query degree tuning | ~100 LOC |
| **P4** | RobustPrune final pass | PiPNN | +10% QPS | ~50 LOC |
| **P5** | Search-free construction | PiPNN | build-time only | ~800 LOC |

**P0 + P1 together could reduce blk/q from 201 to ~100-120 for median
queries. P2 (inline codes) would further halve that to ~50-60. Combined:
3-4x IO reduction from current baseline.**
