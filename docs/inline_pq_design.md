# Opt-A: Inline PQ Codes — Design Document

## Status: Design (pre-implementation)

## 1. Problem Statement

Current search: each expansion pops a candidate, fetches its 4KB adjacency block (1 NVMe IO), decodes ~24 neighbor IDs, computes exact distances to ALL neighbors (from VectorBank in DRAM), and pushes all non-dominated neighbors into the candidate heap. This means:

- **blk/q ≈ ef + 1** (201 blocks at ef=200) — validated across all experiments
- **54% of expansions are wasted** (add 0 neighbors to beam)
- Prefetch hides latency but doesn't reduce IO count
- Ada-ef can redistribute budget but can't cut the per-expansion cost

The root cause: **every neighbor that passes the `visited` check gets a full distance computation and potentially enters the candidate heap**, generating future expansions. There's no cheap pre-filter.

**Implementation note (current codebase)**: the current adjacency decode returns an owned `Vec<u32>`,
allocating per expansion. Any inline-code design should replace this with a zero-copy decode that
borrows slices from the 4KB buffer to avoid allocator noise on p99.

## 2. Solution: Score-Before-Expand with Inline PQ Codes

Store PQ-compressed codes for each neighbor **inline in the adjacency block**. After fetching a block:

1. Decode neighbor IDs + their PQ codes (zero extra IO)
2. Compute **approximate PQ distances** to all neighbors (cheap: M table lookups per neighbor)
3. **Gate**: only compute exact distance / push the Top-T best neighbors into the candidate heap
4. Skip the rest — they never trigger future expansions

If T < degree, fewer candidates enter the heap → fewer expansions → fewer block fetches → lower blk/q.

### Why This Works for Disk Search

- DiskANN keeps PQ codes **in DRAM** because their vectors are on disk. We keep vectors in DRAM too, but our bottleneck is adjacency block IO, not vector IO.
- PageANN stores inline codes and achieves **46% fewer IOs** at iso-recall (their primary result).
- VSAG decouples graph and codes (separate FlattenDataCell) — simpler but loses the "one IO brings neighbor codes for free" advantage.
- Our approach: follow PageANN's inline philosophy, but simpler — one vector per block (not multi-vector pages).

## 3. Block Layout v2

### Current Layout (v1)
```
Offset  Size   Content
0       2      degree (u16, little-endian)
2       6      padding (zeros)
8       4*m    neighbor_vids (u32 × degree)
8+4*m   ...    zeros to 4096

At m_max=32: header(8) + IDs(128) = 136 bytes used. 3960 bytes WASTED.
```

### Proposed Layout (v2)
```
Offset  Size   Content
0       2      degree (u16)
2       1      version (u8) — 0x00=v1 (no codes), 0x01=v2 (PQ inline)
3       1      code_type (u8) — 0x01=PQ
4       2      num_subquantizers (u16, M) — e.g., 48
6       2      reserved (zeros)
8       4*m    neighbor_vids (u32 × degree)
8+4*m   M*m    neighbor_codes (u8[M] × degree, row-major)
...     ...    zeros to 4096

At m_max=32, M=48: header(8) + IDs(128) + codes(1536) = 1672 bytes. 2424 bytes free.
```

**Constraints**:
- `dim % M == 0` must hold (or we must define how to pad / handle a short tail subspace).
- `degree` is per-node; blocks are fixed-size, so `degree <= m_max` at build time.
- Endianness: all integer fields are little-endian.

### Capacity Analysis (m_max=32)

| PQ Config | Bytes/Neighbor | Total Codes | Block Used | Free |
|-----------|---------------|-------------|------------|------|
| PQ16×8 | 16 | 512 | 648 | 3448 |
| PQ32×8 | 32 | 1024 | 1160 | 2936 |
| PQ48×8 | 48 | 1536 | 1672 | 2424 |
| PQ96×8 | 96 | 3072 | 3208 | 888 |

All fit comfortably. **PQ32–PQ48** are reasonable starting points for 768-dim vectors:
- PQ32: smaller per-expansion compute + smaller LUT (often friendlier to L1D)
- PQ48: better approximation quality (fewer false negatives at the same gate_ratio)

**Note on caches**: most x86_64 L1D caches are 32KB. A PQ48 LUT is `48 × 256 × 4 = 48KB`
and may not fit in L1D; expect it to live partly in L2. If this becomes a hotspot:
- try PQ32 first (32KB LUT fits L1D), or
- store LUT as `f16/i16` and accumulate in `f32`, or
- prefetch LUT lines explicitly (likely unnecessary at degree=32).

### Why Not Int8/SQ8 Inline?

Int8 codes for 768-dim = **768 bytes per neighbor**. At m_max=32: 768 × 32 = 24,576 bytes. Doesn't fit in 4KB. Not even close.

### Version Compatibility

- v1 blocks have byte 2 = 0x00 (was padding). Reader checks version byte.
- v2 reader can transparently handle v1 blocks (no codes → skip gating, use all neighbors).
- No migration needed for existing indices — new indices write v2, old indices remain v1.

## 4. PQ Training Pipeline

### 4a. Codebook Training

```
Input:  flat f32 vectors (N × dim), sampling rate, M subquantizers
Output: codebook (M × 256 × subspace_dim), where subspace_dim = dim / M

Algorithm:
1. Sample min(N, 100K) vectors uniformly (sufficient for k-means convergence)
2. For each subspace m in [0, M):
   a. Extract subspace slice: vectors[:, m*sd : (m+1)*sd]  (sd = dim/M)
   b. Run k-means with k=256, max_iter=25, 3 restarts
   c. Store 256 centroids as codebook[m]
```

Following DiskANN: global codebook (not per-node), trained once offline. Stored in a separate file `pq_codebook.bin`.

**Engineering note**: implementing fast k-means in Rust is non-trivial. For an MVP, it's acceptable
to generate the codebook + codes offline (FAISS/Python) and load them, then replace with native
training later once the on-disk format + search integration are proven.

### 4b. Vector Encoding

Encode **each dataset vector once** and reuse:
1. Precompute `pq_codes_all: Vec<u8>` of shape `N × M` (one code per vector).
2. When writing adjacency block for node `u` with neighbors `[n1..nd]`, write
   `pq_codes_all[n_i]` inline next to `n_i`.

This avoids re-encoding the same neighbor many times across blocks.

### 4c. Lookup Table (per-query, computed once)

```
For query q:
  lut = [[0.0f32; 256]; M]
  for m in 0..M:
    q_sub = q[m*sd .. (m+1)*sd]
    for c in 0..256:
      lut[m][c] = || q_sub - codebook[m][c] ||²  // or appropriate metric
  return lut
```

Cost: M × 256 × sd FLOPs = 48 × 256 × 16 = 196,608 FLOPs. Negligible vs. search.

For cosine distance on normalized vectors: use inner product LUT, then distance = 1 - sum(lut[m][code[m]]).

**Important**: for cosine, FDL + PQ both assume vectors are L2-normalized. If the index stores
raw vectors, normalize for PQ training/encoding and for query LUT generation.

### 4d. Approximate Distance Computation

```
fn pq_distance(lut: &[[f32; 256]; M], code: &[u8; M]) -> f32 {
    let mut dist = 0.0f32;
    for m in 0..M {
        dist += lut[m][code[m] as usize];
    }
    dist
}
```

Cost per neighbor: M additions + M lookups = 48 ops. Compare to exact FP32 distance: 768 multiplies + 768 adds = 1536 ops. **32× cheaper per neighbor.**

## 5. Gating Algorithm

### 5a. Minimum-Risk Version (no recall loss target)

```
Current search loop (simplified):
  pop candidate → fetch block → decode neighbors → distance ALL → push non-dominated

New search loop with gating:
  pop candidate → fetch block → decode neighbors + PQ codes
  → PQ-approximate ALL neighbors (cheap, ~48 ops each)
  → select Top-T by approximate distance (use deterministic tie-break: vid)
  → exact distance only Top-T (from VectorBank)
  → push non-dominated to candidate heap
```

**T selection**: T = min(degree, max(k_min, ceil(degree × gate_ratio)))
- `gate_ratio = 0.5` means keep top 50% of neighbors → 2× fewer candidates entering heap
- `k_min = 4` ensures we never gate too aggressively on low-degree nodes

**Coupling to ef (recommended)**: in our beam search, IO reduction comes from *fewer expansions*.
If T is too small, the candidate heap can starve and terminate early with recall loss.
So treat `gate_ratio` as a function of the per-query budget:
- `ef` small: gate conservatively (higher T)
- `ef` large: gate more aggressively (lower T)

W (prefetch window) does not need to change, but may be re-tuned once gating is in place.

### 5b. Why Gating Reduces blk/q

In our implementation, the main loop stops when the nearest unexpanded candidate is worse than
the current worst of the `nearest` set. Gating can reduce blk/q if it causes the beam to
converge faster (the `nearest` boundary improves quickly) and prevents low-quality neighbors
from entering `candidates` and triggering future expansions.

This is an *empirical* claim: IO reduction must be measured via `blk/q` curves at iso-recall.

### 5c. Interaction with Existing Features

- **Prefetch (W=4)**: Unchanged. Prefetch still looks ahead in candidate heap. Fewer candidates in heap = fewer prefetch targets, but each is higher quality.
- **Adaptive stopping (S/D)**: Still works. May trigger earlier because beam converges faster with better candidates.
- **Ada-ef**: Still works. Per-query ef adjusts budget; gating reduces waste within each budget level.
- **Visited set**: Checked BEFORE PQ distance — don't waste PQ ops on already-visited neighbors.

## 6. Deterministic Access & Prefetch Compatibility

### Current access pattern:
```
1. Pop candidate C from min-heap (deterministic: always smallest distance)
2. Prefetch hint for next W candidates in heap
3. get_or_load(C.vid) — fetch C's adjacency block
4. Decode neighbors, compute distances, push to heap
```

### With inline gating:
```
1. Pop candidate C from min-heap (unchanged — deterministic)
2. Prefetch hint for next W candidates in heap (unchanged)
3. get_or_load(C.vid) — fetch C's adjacency block (unchanged — same IO)
4. Decode neighbors + PQ codes (new: also read codes from block)
5. PQ-approximate all non-visited neighbors (new: cheap computation)
6. Select Top-T by PQ distance (new: gating step)
7. Exact distance only for Top-T (changed: fewer distance computations)
8. Push non-dominated Top-T to heap (changed: fewer pushes)
```

**Access order is deterministic** if:
- PQ distance computation is deterministic, and
- Top-T selection is stable under ties (define a deterministic tie-break: `(pq_dist, vid)`).

Gating affects which neighbors enter the heap, but candidates are still popped in exact-distance order.

**Prefetch remains valid**: hints are issued for candidates already in the heap. Gating reduces future heap insertions but doesn't affect already-queued candidates.

## 7. Evaluation Plan

### 7a. Metrics (all from existing SearchPerfContext + new counters)

**New counters to add to SearchPerfContext:**
- `pq_candidates_scored` — total neighbors scored with PQ
- `pq_candidates_gated` — neighbors that passed the gate (went to exact scoring)
- `pq_candidates_filtered` — neighbors filtered out by gate
- `gate_ratio_effective` — actual gated/scored ratio per query

**Existing metrics to compare (iso-recall):**
- `blk/q` — PRIMARY: must decrease (target: 201 → <150)
- `mis/q` — should decrease proportionally
- `p50ms`, `p99ms` — latency (should improve with fewer IOs)
- `wasted_expansions` — should decrease (higher-quality candidates)
- `waste%` — should decrease
- `qps` — should increase

### 7b. Experiment Design

**EXP-PQ-GATE**: Cohere 100K, 768d, k=100, ef=200, W=4

| Config | gate_ratio | Expected blk/q | Notes |
|--------|-----------|----------------|-------|
| baseline | 1.0 (no gating) | ~201 | Control |
| gate-75 | 0.75 | ~170 | Conservative |
| gate-50 | 0.50 | ~140 | Moderate |
| gate-33 | 0.33 | ~120 | Aggressive |
| gate-25 | 0.25 | ~110 | Very aggressive |

For each: report recall, blk/q, mis/q, p50, p99, waste%, pq_candidates_filtered%.

**Sweep gate_ratio to find the Pareto frontier**: plot recall vs blk/q for different gate ratios. The sweet spot is where recall starts dropping faster than blk/q decreases.

### 7c. Diagnostic: PQ Approximation Quality

Before the full search experiment, validate PQ approximation quality standalone:
- Sample 1000 query-neighbor pairs
- Compute both exact cosine distance and PQ approximate distance
- Report correlation (Spearman rank), MSE, and "Top-T overlap" (what fraction of the exact Top-T are also in the PQ Top-T)
- Target: Top-T overlap > 80% at gate_ratio=0.5

## 8. Build-Side Changes

### 8a. New Files

| File | Content |
|------|---------|
| `crates/core/src/pq.rs` (or `crates/core/src/quantization/pq.rs`) | PQ codebook training (k-means), encoding, LUT generation, distance |
| `crates/storage/src/pq_store.rs` | On-disk codebook load + file format (small, simple) |
| `crates/storage/src/meta.rs` | Add PQ config + adjacency block versioning fields |

### 8b. Modified Files

| File | Change |
|------|--------|
| `crates/storage/src/adjacency.rs` | `encode_adj_block_v2()` / `decode_adj_block_v2()` with PQ codes |
| `crates/storage/src/writer.rs` | `IndexWriter::write()` accepts PQ codebook + encoded neighbors |
| `crates/engine/src/search.rs` | Gating logic in `disk_graph_search_pipe` after neighbor decode |
| `crates/engine/src/perf.rs` | New PQ counters |
| `crates/engine/src/engine.rs` (or index loader) | Load PQ codebook at startup and pass handle into search |
| `crates/engine/tests/disk_search.rs` | New `exp_pq_gate` benchmark |

### 8c. Index Build Flow (Updated)

```
1. Build NSW graph (unchanged)
2. Train PQ codebook on dataset vectors (NEW)
3. Encode all dataset vectors once: pq_codes_all = pq_encode_all(vectors)  (N × M bytes)
4. For each vector u with neighbors [n1, n2, ...]:
   a. Gather codes: codes[i] = pq_codes_all[n_i]
   b. Write v2 adjacency block: [header][neighbor_ids][neighbor_codes]
4. Write codebook to `pq_codebook.bin`
5. Write meta.json with adjacency layout + PQ config (see below)
```

**Meta fields to add (minimum)**:
- `adj_layout_version: u32` (1=v1 IDs only, 2=v2 IDs+codes)
- `pq: { enabled: bool, m: u16, k: u16 (=256), subspace_dim: u16, metric: "l2"|"ip", codebook_file: "pq_codebook.bin" }`

**pq_codebook.bin format (minimum)**:
- Magic + version
- `m`, `k`, `subspace_dim`, `metric_id`
- Centroid table (little-endian f32): `m × k × subspace_dim`

### 8d. Search Flow (Updated)

```
1. Load PQ codebook into DRAM at startup (~48 × 256 × 16 × 4 = 768KB)
2. Per query:
   a. Compute PQ lookup table: lut[m][c] = dist(q_sub_m, codebook[m][c])
   b. Search loop (modified):
      - Pop candidate, fetch block, decode neighbors + codes
      - For each non-visited neighbor: pq_dist = sum(lut[m][code[m]])
      - Sort neighbors by pq_dist, take Top-T
      - Exact distance only for Top-T, push non-dominated
```

## 9. Implementation Order

1. **PQ crate**: codebook training, encode, LUT, distance (pure computation, no IO)
2. **Block v2**: encode/decode with inline codes (storage only, no search change)
3. **Writer v2**: write v2 blocks during index build
4. **Search gating**: modify `disk_graph_search_pipe` to use PQ LUT + gate
5. **Benchmark**: `exp_pq_gate` with gate_ratio sweep
6. **Validate**: PQ approximation quality diagnostic

Steps 1-2 can be developed and unit-tested without touching the search path. Step 4 is the critical integration point.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PQ approximation too poor for 768d cosine | Gate filters good neighbors → recall drops | Start with PQ48 (16d subspaces). If insufficient, try PQ96 (8d). Validate with Top-T overlap diagnostic before search integration. |
| Gate ratio too aggressive | Recall loss > 1% | Default gate_ratio=0.75 (conservative). Sweep to find sweet spot. |
| PQ distance computation adds latency | p50 increases despite fewer IOs | Net CPU cost depends on (M, degree, T) and cache behavior. Treat as a measurement problem: use existing `cmp_ms`/`dst_ms` breakdown to confirm PQ is not the new bottleneck. |
| Block format change breaks existing indices | Can't read old indices | Version byte (offset 2) — v1 reader skips codes, v2 reader uses them. Fully backward compatible. |
| LUT / codebook cache behavior | PQ LUT lookups may become a hotspot | L1D is often 32KB; PQ32 LUT fits, PQ48 LUT may spill into L2. Start with PQ32–PQ48 and validate `cmp_ms`/`dst_ms` counters before optimizing. |

**Layering note**: IoDriver should remain “read adjacency blocks” only. Prefer loading the PQ codebook via `divergence_storage` (regular buffered IO) and pass a handle into the search path.
