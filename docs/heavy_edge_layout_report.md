# Heavy-Edge Layout: End-to-End Validation Report

**Date**: 2026-03-10
**Dataset**: Cohere 100K, dim=768, cosine, k=100
**Hardware**: AWS i4i (NVMe SSD), single core
**Graph**: NSW m_max=32, ef_construction=200

## Summary

Heavy-edge page packing reduces adjacency IO by 20% vs BFS across all pipeline stages,
with **zero recall loss**. The improvement carries end-to-end through SAQ graph-only,
SAQ+FP32 refine, SAQ+FP16 refine, SAQ+int8 refine, and three-stage pipelines.

Heavy-edge is now the default v3 layout (`V3_LAYOUT=heavy_edge`).

---

## 1. Index Build

| Layout     | Pages | Adj Size (MB) | adj_reorder |
|------------|------:|---------------:|-------------|
| BFS        | 1,537 |           6.0  | bfs         |
| heavy_edge | 1,526 |           6.0  | heavy_edge  |

Heavy-edge packs 11 fewer pages (0.7% denser). Both use the same graph — only page
assignment differs.

---

## 2. Graph-Only Traversal (FP32 scoring in DRAM)

### Per-query cold (pool.clear() each query — pure layout signal)

| Layout     | recall | p50 ms | p99 ms | QPS   | mis/q | phy/q | hit%  |
|------------|-------:|-------:|-------:|------:|------:|------:|------:|
| BFS        | 0.963  |   8.0  |  10.5  | 109.5 |  23.5 | 135.3 | 88.3% |
| heavy_edge | 0.963  |   6.9  |   8.8  | 125.8 |  20.2 | 108.1 | 90.0% |
| **delta**  |   —    | **-14%** | **-16%** | **+15%** | **-14%** | **-20%** | **+1.7pp** |

### Warm (10 warmup queries, cross-query cache reuse)

| Layout     | recall | p50 ms | p99 ms | QPS   | mis/q | phy/q | hit%  |
|------------|-------:|-------:|-------:|------:|------:|------:|------:|
| BFS        | 0.963  |   7.1  |   9.2  | 140.1 |  18.9 | 113.8 | 90.6% |
| heavy_edge | 0.963  |   6.3  |   7.8  | 161.7 |  15.5 |  87.1 | 92.3% |
| **delta**  |   —    | **-11%** | **-15%** | **+15%** | **-18%** | **-23%** | **+1.7pp** |

---

## 3. SAQ Graph-Only (Proxy Distance, No Disk Refine)

Same graph traversal but scoring via SAQ (12-segment eqseg16, unpacked).

### Per-query cold

| Layout     | recall | p50 ms | p99 ms | QPS   | mis/q | phy/q | hit%  |
|------------|-------:|-------:|-------:|------:|------:|------:|------:|
| BFS        | 0.913  |   7.9  |   9.7  | 111.1 |  23.6 | 135.9 | 88.2% |
| heavy_edge | 0.913  |   6.9  |   8.8  | 125.5 |  20.4 | 108.3 | 89.9% |
| **delta**  |   —    | **-13%** | **-9%** | **+13%** | **-14%** | **-20%** | **+1.7pp** |

### Warm

| Layout     | recall | p50 ms | p99 ms | QPS   | mis/q | phy/q | hit%  |
|------------|-------:|-------:|-------:|------:|------:|------:|------:|
| BFS        | 0.913  |   7.0  |   9.1  | 142.5 |  19.2 | 113.7 | 90.5% |
| heavy_edge | 0.913  |   6.2  |   8.0  | 159.8 |  16.1 |  88.1 | 92.0% |
| **delta**  |   —    | **-11%** | **-12%** | **+12%** | **-16%** | **-23%** | **+1.5pp** |

---

## 4. SAQ + FP32 Disk Refine (Two-Stage Pipeline)

The headline production pipeline: SAQ proxy traversal → FP32 disk refinement.
All perq-cold unless noted.

### Refine R sweep (perq-cold)

| Layout     | R   | recall | p50 ms | p99 ms | QPS   | adj phy/q | total io/q |
|------------|----:|-------:|-------:|-------:|------:|----------:|-----------:|
| BFS        |  80 | 0.785  |   9.0  |  10.7  |  99.6 |     135.8 |      215.8 |
| heavy_edge |  80 | 0.785  |   7.8  |   9.7  | 112.1 |     108.3 |      188.3 |
| BFS        | 120 | 0.954  |   9.4  |  11.2  |  94.7 |     135.9 |      255.9 |
| heavy_edge | 120 | 0.954  |   8.3  |  10.2  | 106.3 |     108.2 |      228.2 |
| **BFS**    | **160** | **0.963** | **9.9** | **11.8** | **90.8** | **135.9** | **295.9** |
| **heavy_edge** | **160** | **0.963** | **8.6** | **10.7** | **102.2** | **108.3** | **268.3** |
| BFS        | 200 | 0.964  |  10.5  |  12.4  |  86.1 |     135.9 |      335.9 |
| heavy_edge | 200 | 0.964  |   9.2  |  11.2  |  96.7 |     108.3 |      308.3 |

**At iso-recall 0.963 (R=160)**: heavy_edge is **13% faster** (p50 8.6 vs 9.9ms),
**+13% QPS** (102.2 vs 90.8), **20% fewer adj IO** (108.3 vs 135.9), **9% fewer total IO**
(268.3 vs 295.9).

### Warm (R=160)

| Layout     | recall | p50 ms | p99 ms | QPS   | adj phy/q | total io/q |
|------------|-------:|-------:|-------:|------:|----------:|-----------:|
| BFS        | 0.963  |   9.1  |  11.0  | 110.7 |     112.5 |      272.6 |
| heavy_edge | 0.963  |   8.2  |   9.9  | 122.0 |      87.4 |      247.4 |
| **delta**  |   —    | **-10%** | **-10%** | **+10%** | **-22%** | **-9%** |

---

## 5. SAQ + FP16 Disk Refine (perq-cold)

| Layout     | R   | recall | p50 ms | p99 ms | QPS  | adj phy/q | total io/q |
|------------|----:|-------:|-------:|-------:|-----:|----------:|-----------:|
| BFS        | 160 | 0.963  |  11.4  |  13.4  | 80.8 |     135.9 |      295.9 |
| heavy_edge | 160 | 0.963  |  10.2  |  12.2  | 88.1 |     108.3 |      268.3 |
| **delta**  |     |   —    | **-11%** | **-9%** | **+9%** | **-20%** | **-9%** |

---

## 6. SAQ + Int8 Disk Refine (perq-cold)

| Layout     | R   | recall | p50 ms | p99 ms | QPS  | adj phy/q | total io/q |
|------------|----:|-------:|-------:|-------:|-----:|----------:|-----------:|
| BFS        | 160 | 0.929  |  11.1  |  13.0  | 81.7 |     135.9 |      295.9 |
| heavy_edge | 160 | 0.929  |  10.1  |  12.5  | 88.8 |     108.3 |      268.3 |
| **delta**  |     |   —    | **-9%** | **-4%** | **+9%** | **-20%** | **-9%** |

---

## 7. Hub Pinning (SAQ+FP32 R=160, perq-cold)

| Layout     | Pinned | recall | p50 ms | p99 ms | QPS   | mis/q | total io/q |
|------------|-------:|-------:|-------:|-------:|------:|------:|-----------:|
| BFS        |     64 | 0.963  |   9.3  |  11.3  |  96.5 |  21.7 |      282.9 |
| heavy_edge |     64 | 0.963  |   8.4  |  10.1  | 105.3 |  17.4 |      254.2 |
| BFS        |    128 | 0.963  |   9.0  |  11.4  |  98.7 |  20.8 |      277.7 |
| heavy_edge |    128 | 0.963  |   8.2  |   9.9  | 107.1 |  16.4 |      249.4 |

Heavy-edge + pin128 achieves the lowest p50 (8.2ms) and highest QPS (107.1) of
any perq-cold SAQ+refine config.

---

## 8. Three-Stage Pipeline (SAQ → int8 filter → FP32 exact)

| Layout     | T   | recall | p50 ms | p99 ms | QPS  |
|------------|----:|-------:|-------:|-------:|-----:|
| BFS        | 120 | 0.962  |  12.7  |  14.4  | 72.8 |
| heavy_edge | 120 | 0.962  |  11.7  |  14.1  | 77.8 |
| **delta**  |     |   —    | **-8%** | **-2%** | **+7%** |

---

## 9. exp_bench_stable: v3 with heavy_edge (headline bench)

The stable benchmark now uses heavy_edge by default (`V3_LAYOUT=heavy_edge`).
Key results from `exp_bench_stable`:

| Config          | recall | p50 ms | p99 ms | QPS   | mis/q | phy/q | hit%  |
|-----------------|-------:|-------:|-------:|------:|------:|------:|------:|
| v3-warm         | 0.963  |   6.4  |   8.1  | 157.5 |  15.5 |  87.2 | 92.3% |
| v3-warm-S4D16   | 0.960  |   6.1  |   8.3  | 166.1 |  15.4 |  83.2 | 91.6% |
| v3-cold         | 0.963  |   6.5  |   8.4  | 153.7 |  16.1 |  89.5 | 92.0% |
| v3-perq-cold    | 0.963  |   7.1  |   8.9  | 122.4 |  20.2 | 108.2 | 90.0% |
| v3-warm-ef250   | 0.973  |   7.8  |   9.6  | 127.4 |  18.4 | 109.6 | 92.7% |

### v1 vs v3 equal-budget comparison (perq-cold)

| Config   | p50 ms | QPS   | phy/q | hit%  |
|----------|-------:|------:|------:|------:|
| v1-256KB |  10.2  |  87.6 | 213.6 | 82.8% |
| v3-256KB |   7.5  | 117.3 | 123.8 | 88.9% |
| v1-1MB   |  10.6  |  82.7 | 209.2 | 83.3% |
| v3-1MB   |   7.1  | 121.9 | 108.2 | 90.0% |

v3 heavy_edge consistently beats v1 by 1.3-1.5× in QPS at equal cache budget.

---

## 10. Conclusions

1. **Heavy-edge wins everywhere**: 13-15% QPS improvement over BFS in graph-only,
   7-13% in two-stage pipelines. Zero recall change.

2. **The win is purely from fewer page misses**: mis/q drops from ~23.5 to ~20.2
   (perq-cold) and ~18.9 to ~15.5 (warm). Graph structure is identical — only
   page assignment changed.

3. **Carries through end-to-end**: SAQ proxy, FP32 refine, FP16 refine, int8 refine,
   three-stage pipeline, hub pinning — all show the same directional improvement.

4. **Refine IO dilutes the advantage**: adj IO is ~40% of total IO in SAQ+R160
   (108/268 for HE). As refine R increases, adj IO becomes a smaller fraction,
   so layout improvement matters less. At R=200, HE still saves 8%.

5. **Pool size interaction**: At 256 pages (1MB), HE gets 90% hit rate vs BFS 88%.
   The advantage is largest under small pools (the harder test).

6. **Default recommendation**: `V3_LAYOUT=heavy_edge` for all production and
   benchmark use. BFS remains available via `V3_LAYOUT=bfs` for comparison.
   `meta.json` records `adj_reorder` for reproducibility.

---

## Appendix: Implementation

- **Reorder algorithm**: MARGO-style greedy page packing. Seeds sorted by in-degree
  descending. Candidates scored by sum of edge weights to current page members,
  where w(u,v) = indeg(u) + indeg(v). Deterministic tie-breaks (score desc, indeg
  desc, VID asc).

- **Performance**: Stamp array for O(1) candidate dedup (replaces O(k) linear scan).
  Precomputed `rec_sizes[]` avoids repeated `neighbors_fn` calls. u32 round overflow
  guard for graphs >4B fill iterations.

- **V3_LAYOUT knob**: Env var defaults to `heavy_edge`. Writes `adj_reorder` field
  to `meta.json`. Printed in all bench headers for auditability.

- **Files changed**: `crates/storage/src/adjacency.rs` (reorder optimization),
  `crates/engine/tests/disk_search.rs` (dual-layout SAQ bench, V3_LAYOUT knob),
  `crates/storage/src/meta.rs` (adj_reorder field),
  `crates/storage/src/writer.rs` (adj_reorder_label parameter).
