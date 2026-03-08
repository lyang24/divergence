# Paper Distillation 5.0: Ada-ef, FlatNav, VSAG, DiskANN

**Date**: 2026-03-07
**Sources**: Ada-ef (arXiv 2512.06636), FlatNav (ICML 2025 Workshop), VSAG (PVLDB, Ant Group), DiskANN Rust port (Microsoft)

---

## 1. Ada-ef: Distribution-Aware Adaptive ef

**Paper**: "Distribution-Aware Exploration for Adaptive HNSW Search" (SIGMOD 2026, Chao Zhang & Renée J. Miller)
**Code**: https://github.com/chaozhang-cs/hnsw-ada-ef

### 1.1 Core Idea

Per-query ef estimation using the statistical distribution of distances from query to all dataset vectors (the "FDL" — Full Distance List). Easy queries (many close neighbors) get small ef; hard queries get large ef. No ML model — pure statistics + lookup table.

### 1.2 FDL Distribution

**Definition**: FDL(q, V) = (dist(q, v₁), ..., dist(q, vₙ)) — all n distances from query to dataset.

**Key theorem (5.2)**: Under i.i.d.-across-dimensions assumption, FDL converges to Gaussian as d→∞ (CLT). For cosine distance:

```
FDL_CD(q, V) ~ N(1 - μ_CS, σ²_CS + Δ_CS)         [Eq. 3]

μ_CS = Σᵢ q̂ᵢ · E[v̂ᵢ]                              (dot product of normalized q with mean of normalized V)
σ²_CS = Σᵢ q̂ᵢ² · Var(v̂ᵢ)                           (weighted variance)
Δ_CS = 2·Σᵢ<ⱼ q̂ᵢ·q̂ⱼ·Cov(v̂ᵢ, v̂ⱼ)                  (cross-dimension correlation correction)
```

Compact form: μ = q̂ · M, σ² = q̂ · Σ · q̂ᵀ (quadratic form with covariance matrix).

**L2 NOT derived** — paper states Euclidean case is open due to squared terms. Cosine/IP only.

### 1.3 Two-Stage Online Algorithm (Paper's Version)

**Phase 1 — Distance collection**: Search begins with ef=∞ (unbounded). Collect distances to all visited nodes into list D until |D| = l, where l = size of 2-hop neighborhood from entry point (~M² ≈ 256 for M=16).

**Phase 2 — Estimated ef**: Call ESTIMATE-EF(q, D, r) once, get ef, truncate result set W to ef, continue search normally.

**⚠ Divergence adaptation**: The paper's Phase 1 requires ~M² adjacency reads with ef=∞ before estimation — on cold NVMe this means ~256 IOs of pure overhead before we even know the ef. This is unacceptable for disk-based search. Instead, Divergence will use **entry_set seed distances** (pure DRAM, zero IO): compute distances from query to all entry_set vectors in VectorBank, use those as the sample D. Current entry_set=64; expand to 128-256 for Ada-ef sampling (still pure DRAM, cost = 128-256 distance computations ≈ 0.1ms at d=768). This replaces the 2-hop IO-heavy collection with a zero-IO alternative.

### 1.4 ESTIMATE-EF Algorithm

```
ESTIMATE-EF(q, D, r):
  1. μ = q̂ · M                          // O(d) dot product
  2. σ = sqrt(q̂ · Σ · q̂ᵀ)              // O(d²) quadratic form — dominant cost
  3. Compute m=5 bin thresholds:
     θᵢ = μ_CD + σ · Φ⁻¹(0.001·i)      // inverse normal CDF, 5 lookups
  4. Count distances in D per bin:
     cᵢ = |{d ∈ D : θᵢ₋₁ < d ≤ θᵢ}|
  5. Score = Σᵢ wᵢ · cᵢ / |D|
     where wᵢ = 100 · e^{-(i-1)}        // exponential decay: [100, 36.8, 13.5, 5.0, 1.8]
  6. Look up score (cast to int) in ef-estimation table
  7. Return max(ef_from_table, WAE)      // WAE = weighted average ef (floor)
```

**Interpretation**: High score = many distances in low quantiles = easy query = small ef. Low score = hard query = large ef.

### 1.5 EF-Estimation Table (Offline, Once)

1. Sample 200 vectors from V as proxy queries
2. For each: compute score, run HNSW search at increasing ef values, record (ef, recall) pairs
3. Group by integer score → table: score → [(ef₁, recall₁), (ef₂, recall₂), ...]
4. Online: look up smallest ef where recall ≥ target r

### 1.6 Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| m (bins) | 5 | Quantile bins |
| δ (bin width) | 0.001 | Each bin = 0.1% of distribution |
| l (sample size) | 2-hop (paper) | Paper: ~M² visited distances. **Divergence v0: entry_set size (128–256), pure DRAM** |
| Sampling | 200 | Proxy queries for table construction |
| Decay | Exponential | Critical — uniform weights fail badly |
| ef upper bound | 5000 | Table cap |

### 1.7 Results

- **4x latency reduction** vs DARTH (SOTA learning-based) at same recall
- **50x less offline compute**, **100x less offline memory** vs DARTH/LAET
- Reaches target recall 0.95 on **all 8 datasets** (DARTH/LAET: only 5/8)
- All experiments use **cosine distance**, M=16, efConstruction=500
- 2-hop is sweet spot; 3-hop too expensive with no recall gain

### 1.8 Limitations

- **L2 not supported** — only cosine/IP derived
- **O(d²) per query** for full covariance (see §1.9 for mitigation)
- Gaussian assumption requires d ≥ ~96
- ef estimated once, no mid-search correction
- Cross-modal retrieval (text→image) is hardest case

### 1.9 Mapping to Divergence

**Distance sampling**: Use entry_set seed distances (pure DRAM, zero IO) instead of the paper's 2-hop graph traversal. Current entry_set=64; expand to 128-256 for Ada-ef (still negligible: 256 × d=768 dot products ≈ 0.1ms). The ef estimate drives (ef, stall_limit, drain_budget) jointly:

```
(ef, stall_limit, drain_budget) = f(score)   // not just ef
```

**Covariance computation — v0 is diagonal-only**:

The paper uses the full d×d covariance matrix Σ for the quadratic form q̂·Σ·q̂ᵀ. This is problematic:
- **Storage**: 768² × 4 = 2.36MB — acceptable
- **Offline compute**: O(n·d²). At 100K×768 = 45 billion FLOPs. At 1M×768 = 450B FLOPs. At 10M = impractical without sampling/approximation.
- **Online compute**: O(d²) = 590K FLOPs per query — acceptable (<1ms) but unnecessary if diagonal works.

**v0 implementation**: Use diagonal-only variance (ignore Δ cross-correlation term):
```
σ² = Σᵢ q̂ᵢ² · Var(v̂ᵢ)     // O(d), precompute Var(v̂ᵢ) offline as d floats
```
This drops the Δ correction but the paper's ablation shows exponential bin weighting matters far more than variance accuracy. If v0 diagonal scoring separates easy/hard queries well enough (testable via score-vs-recall correlation on our Cohere data), full covariance is unnecessary.

**Upgrade path** (only if diagonal is insufficient):
1. Low-rank approximation: Σ ≈ U·diag(λ)·Uᵀ with top-r eigenvectors → O(r·d) per query
2. Full Cholesky: Σ = L·Lᵀ, compute ||L·q̂||² → O(d²) but cache-friendly
3. Sampled covariance: compute Σ from √n random vectors instead of all n

---

## 2. FlatNav: Flat NSW = HNSW for d ≥ 32

**Paper**: "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" (ICML 2025 Workshop)
**Code**: ~/repos/flatnav

### 2.1 Central Claim

On high-dimensional datasets (d ≥ 32), flat NSW achieves **comparable recall and latency** to HNSW empirically, with **~38% less peak memory** (no upper layers). Not a formal proof — empirical finding across 13+ datasets.

**Methodology**: Both "extract HNSW base layer" and "build flat NSW from scratch" produce identical results. Tested on 4 BigANN 100M datasets + 9 ANN Benchmark 1M datasets + synthetics.

### 2.2 Hub Highway Hypothesis

**Why flat works**: In high-d, *hubness* (Radovanovic et al., 2010) causes a small subset of nodes to appear in many k-NN lists. These hubs get high degree via preferential attachment during construction, forming a natural "highway" subgraph. Queries visit hubs in the first 5-10% of search, routing rapidly to the correct neighborhood — exactly what HNSW's hierarchy was supposed to do.

**Key nuance**: Cosine/angular distance **reduces hubness** (anti-hub properties). The hub highway is weaker for cosine than L2. But flat NSW still matches HNSW for cosine — the effect just isn't as dramatic.

**Implication for us**: Our Cohere cosine workload likely has weak hub highway. Hub pinning may have less ROI for cosine than for L2. However, with cold hit%≈58%, hubs could still affect early miss distribution. **Decision deferred**: run Ada-ef v0 first, then use the stable bench TSV to measure hub miss contribution before committing.

### 2.3 Code Architecture

- **Single header**: `include/flatnav/index/Index.h` (~930 lines)
- **Memory layout**: Contiguous array, each node = `[data | M links | label]`
- **Fixed-degree M**: unused slots are self-loops (link == own ID)
- **Entry point**: Sample 100 nodes uniformly, pick closest to query. Not hierarchical.
- **Neighbor selection**: HNSW diversity heuristic with M/2 cutoff during construction, M during re-pruning
- **Graph reordering**: Gorder + Reverse Cuthill-McKee for cache locality (relevant to our Opt-A block layout)
- **Purely in-memory** — no disk IO, no io_uring, no caching

### 2.4 What We Can Use

| Finding | Action for Divergence |
|---------|----------------------|
| Flat NSW = HNSW for d≥32 | **Validates our design.** No hierarchy needed. |
| Hub highway weak for cosine | Defer hub pinning — validate with Ada-ef data first |
| 100-sample entry point | Our entry phase is <1% — consistent, no change needed |
| M/2 vs M asymmetry | Check if our NSW does this (construction vs re-prune) |
| Self-loops as empty markers | Consider for our adjacency block format |
| Gorder reordering | Relevant to Opt-A: physically co-locate graph neighbors on disk |

---

## 3. VSAG: Ant Group's Vector Search Engine

**Code**: ~/repos/vsag (PVLDB paper). Observations below are from our local checkout; behavior may differ in other versions.

### 3.1 Architecture

Two index families:
- **HGraph** (modern): Multi-level HNSW-style, composable "datacells" for graph/vectors/quantization
- **Legacy HNSW** (hnswlib fork): Inline data layout (vector + neighbors contiguous)

### 3.2 Key Finding: NO Inline PRS Layout in Modern Path

**HGraph stores graph and vectors in completely separate IO regions.** Three independent datacells:
- `basic_flatten_codes_` — quantized codes (SQ8, PQ, etc.)
- `high_precise_codes_` — full-precision codes for reranking
- `bottom_graph_` — adjacency lists

Access during search = two separate random reads per expansion. The "PRS" (Prefetch-Ready Storage) from the paper is **only in the legacy hnswlib path** for in-memory full-precision vectors.

### 3.3 Prefetch: CPU Cache-Line Only

Three levels of `__builtin_prefetch`:
1. Graph neighbor prefetch (stride=3 ahead in neighbor list)
2. Vector code prefetch (stride ahead in batch-4 distance computation)
3. Visited-set prefetch

**No disk IO prefetch, no io_uring, no async pipeline.** VSAG is fundamentally in-memory.

### 3.4 ELP Optimizer (Auto-tuning)

Grid search over 2-3 prefetch stride parameters via mock queries. Accepts only >2% improvement. This is NOT per-query adaptive ef — it's one-time post-build CPU cache optimization.

### 3.5 Tune() Hot-Swap

Can change quantizer type at runtime (FP32 → SQ8 → RaBitQ) without rebuilding graph. Enabled by clean datacell separation. Interesting but not directly applicable to disk-based search.

### 3.6 Quantization

SQ8 is default for high-dim. Per-vector storage: `[sq_codes(dim bytes) | norm(8B) | sum(4B)]`. Codes stored separately from graph edges — no inline neighbor codes.

### 3.7 What We Can Use

| Finding | Action for Divergence |
|---------|----------------------|
| No inline PRS in modern path | Confirms inline is hard in practice; separate stores is the pragmatic default |
| Prefetch stride tuning | Our W=4 is already tuned; ELP-style auto-tune is a nice-to-have |
| SQ8 as default | Consistent with our Int8 results |
| Runtime quantizer swap | Interesting for future but not priority |

---

## 4. DiskANN: SSD ANN (Rust Rewrite)

**Code**: ~/repos/DiskANN — a Rust rewrite of Microsoft's DiskANN (local repo, not the official C++ release). Architecture maps closely to the NeurIPS 2019 paper but implementation details may diverge from the original C++ codebase.

### 4.1 Vamana / RobustPrune

**Core algorithm** (`occlude_list` in index.rs):

For each candidate k, track `occlude_factor[k]` = max ratio d(q,k)/d(j,k) across all selected neighbors j. Keep candidate only if factor < current_alpha.

**Progressive alpha relaxation**: Start at α=1.0, increment by min(α, 1.2) per pass until reaching target α (default 1.2). First pass is strict greedy, subsequent passes add long-range edges.

**Saturation**: When α > 1.0, after pruning, greedily fill remaining slots to reach target degree. Compensates for aggressive pruning.

**Key difference from our NSW**: Vamana's RobustPrune with α>1 creates **sparser but more navigable** graphs. Our NSW uses the HNSW diversity heuristic (geometrically similar but without the α relaxation pass). This is Opt-D.

**Note**: Rust port does NOT implement true two-pass construction. Single-pass with uniform α=1.2 + final global prune.

### 4.2 On-Disk Layout: Co-located Vector + Adjacency

**This is the critical architectural difference from us.**

Each 4KB sector contains: `| vector (dim × sizeof<T>) | neighbor_count (u32) | neighbors[] | filler |`

Single sector read = adjacency list + full-precision vector. No second IO for vector refinement.

**Constraint**: For 768d float32 (3072B vector), a 4KB sector leaves ~1020B for adjacency = ~254 neighbors max. For dim=384 Half precision, they get 59 max degree (confirmed in tests).

Multi-node packing: if node_len < 4KB, pack multiple nodes per sector. If node_len > 4KB, span multiple sectors with alignment padding.

### 4.3 IO Organization

- **io_uring** with O_DIRECT, registered file descriptors
- **Batch-and-wait**: Collect up to beam_width IOs, submit all, wait for ALL to complete, then process
- **No overlap**: IO and compute are sequential within each batch — no pipelining
- **No sector dedup**: Same sector read twice if two candidates are co-located
- **No eviction cache**: Static BFS-from-medoid HashMap, populated at startup, no eviction
- MAX_IO_CONCURRENCY = 128 (ring size)

### 4.4 PQ Codes: RAM Only

PQ compressed data loaded into RAM at startup. During search: PQ table lookups for approximate distance, full-precision vectors from the same sector read for refinement. PQ codes never read from disk during search.

### 4.5 Parameters

| Parameter | Default | Our Equivalent |
|-----------|---------|---------------|
| pruned_degree (R) | user-set, typ. 64-128 | m_max (32) |
| alpha | 1.2 | N/A (HNSW heuristic, no α) |
| l_build | user-set | ef_construction (200) |
| search_list_size (L) | user-set | ef (200) |
| beam_width | 1 | W (4, but semantically different) |
| DISK_SECTOR_LEN | 4096 | adjacency block size (4096) |
| GRAPH_SLACK_FACTOR | 1.3 | N/A |

### 4.6 What We Can Use

| Finding | Action for Divergence |
|---------|----------------------|
| Co-located layout eliminates refinement IO | **Opt-A priority**: inline compressed codes with adjacency, not full vectors |
| Batch-and-wait IO (no pipeline) | Our prefetch pipeline is strictly better (overlaps IO+compute) |
| No sector dedup | Our AdjacencyPool singleflight is strictly better |
| Static BFS cache | Our clock eviction is strictly better for varying workloads |
| α>1 progressive relaxation | **Opt-D**: implement Vamana-style pruning for fewer hops |
| Saturation (backfill after prune) | Important companion to α>1 pruning |

---

## 5. Synthesis: Priority Actions for Divergence

### P0: Ada-ef (Reduce blk/q — the fundamental bottleneck)

**What**: Per-query (ef, stall_limit, drain_budget) from FDL-based difficulty scoring.

**Implementation sketch** (Divergence-adapted, NOT paper's 2-hop version):
1. Offline: compute per-dimension variance Var(v̂ᵢ) and mean M of normalized dataset — O(n·d), d floats each
2. Offline: sample 200 vectors, build score→(ef, S, D, recall) table via bench runner
3. Online per query (BEFORE main IO loop):
   - Compute distances from q̂ to entry_set vectors (128-256 seeds, pure DRAM, ~0.1ms)
   - Compute μ = q̂·M (O(d)), σ² = Σᵢ q̂ᵢ²·Var(v̂ᵢ) (O(d), diagonal-only v0)
   - Bin seed distances into 5 quantile bins, compute weighted score
   - Look up (ef, S, D) from table
4. Continue search with estimated parameters

**Risk**: Diagonal-only may not separate easy/hard queries as well as full covariance. Testable before committing.

**Expected impact**: 2-4x latency reduction on easy queries (majority). Hard queries get larger ef → better tail recall.

### P1: Inline PQ Codes (Opt-A, from DiskANN co-location insight)

Moved up from P2. See §5 P2 below. Hub pinning deferred.

### Hub Pinning (Deferred, pending Ada-ef data)

FlatNav shows cosine distance has weak hubness, but cold hit%≈58% means hubs could still matter for miss distribution. **Decision**: run Ada-ef v0, then use stable bench TSV to measure hub miss contribution before committing.

### P2: Inline PQ Codes (Opt-A, from DiskANN insight)

DiskANN's co-located layout eliminates refinement IO but limits max degree. A better hybrid: keep adjacency block, but **inline PQ16 codes** (16 bytes/neighbor) for zero-IO approximate scoring of next-hop candidates. At M=32, PQ codes = 512B, fitting easily in 4KB alongside adjacency.

VSAG's modern HGraph does NOT do this (separate stores), confirming it's engineering-hard. But DiskANN proves the benefit of co-location.

### P3: Vamana α>1 (Opt-D, from DiskANN)

Progressive alpha relaxation creates sparser, more navigable graphs → fewer hops → fewer IOs. Combined with saturation (degree backfill), this could reduce blk/q significantly.

**Not urgent**: Ada-ef attacks the same problem (reduce wasted expansions) from the query side. Do Ada-ef first, measure, then decide if graph improvement is still needed.

---

## 6. Key Corrections to Previous Distillations

| Previous claim | Correction |
|----------------|-----------|
| "VSAG PRS = inline codes" (paper_distill_4.0) | **Wrong for modern HGraph.** PRS inline layout only exists in legacy hnswlib path. Modern VSAG uses separate datacells. |
| "Hub pinning = P1 priority" (MEMORY.md) | **Deferred.** FlatNav shows weak cosine hubness, but cold hit%≈58% leaves open question. Validate after Ada-ef v0. |
| "Per-dim SQ failure may be pipeline bug" (MEMORY.md) | FlatNav implements per-dim SQ (min/max uniform). Worth re-investigating but secondary to Ada-ef. |
| Ada-ef rated ★★★ in distill 4.0 | **Confirmed.** Now with full algorithmic details for implementation. |
