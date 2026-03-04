# IO Reduction Research: Cutting blocks/query from 201 to <100

Distilled from 6 papers + 2 experiments. Goal: understand how to make
int8+refine's compute advantage survive under real IO conditions.

---

## Current State (Cohere 100K, dim=768, k=100)

```
mode               r@k    p50_ms   blk/q(cold)  blk/q(warm)  compute speedup
fp32 ef=200       0.963    2.85      201            134          1.00x
fp32 ef=180       0.957    2.62      181            124          1.00x (iso-recall baseline)
int8+ref R=2k     0.962    1.66      202            134          1.58x (iso-recall)
```

**Correction (2026-03-03):** Earlier "134 blocks/query" was cache-miss count
averaged across 100 queries (warm cache). Actual expansion count per query is
**201 = ef+1** (every expansion reads one block). Under cold cache or dataset
larger than DRAM, every expansion is a miss → blocks/query = ef+1.

Problem: int8 saves compute but visits the **same number of blocks**.
Under NVMe IO (80us/miss), cold-cache p50 jumps from 1.66ms to 17.8ms.
The 201 misses dominate; the 1.58x compute win evaporates.

---

## EXP-0 Results: blocks/query Decomposition (2026-03-03)

```
mode            r@k  blk/q  useful  waste  best_md  best_99  1st_tk  entry%
fp32 ef=200   0.963   201     92     109      6       139      1.3     0.6
fp32 ef=180   0.957   181     84      97      6       139      1.3     0.7
fp32 ef=140   0.940   141     68      74      6        93      1.3     0.9
i8+ref R=2k   0.962   202     94     108      6       171      0.9     0.4
```

**Key findings:**
1. **Entry phase is <1%** — first top-k result enters beam at expansion 1.3 (median 1).
   The "approach phase wastes blocks" hypothesis is **WRONG** for this graph+dataset.
2. **54% of expansions are wasted** — add zero neighbors to beam (all visited or dominated).
   This waste is in the CONVERGENCE phase, not the approach phase.
3. **blk/q = ef + 1** — expansion count is directly tied to beam width parameter.
4. **Best result found early** — rank-1 result appears at median expansion 6.

---

## EXP-C Results: Iso-Recall blocks/query Curve (2026-03-03)

```
recall   fp32_ef  fp32_blk  i8r_blk  i8r_ef  blk_ratio
0.913      100      102       -         -       -        (i8 can't reach 0.913)
0.948      160      161      162       160     1.005
0.957      180      181      182       180     1.002
0.962      200      201      202       200     1.002
0.977      280      281      282       280     1.002
```

**FAIL:** At iso-recall, blk_ratio ≈ 1.00 everywhere. Int8+refine provides
**zero structural IO advantage**. Both modes need the same ef (= same blocks)
to achieve the same recall.

At low recall (<0.94), int8 actually needs MORE blocks (1.17x at recall=0.932).
Quantization errors require a wider beam to compensate.

**Conclusion:** Int8's 1.58x compute speedup is real but will NOT survive IO.
blocks/query reduction must come from graph-level or algorithmic changes.

---

## EXP-W Results: Convergence Budgeting (2026-03-03)

Hard-cutoff early stopping sweep: stop search at max_expansions instead of ef.

```
max_exp    r@k    blk/q   waste%   blk_saved
50       0.800     50      6.0%     75%
100      0.911    100     26.0%     50%
130      0.934    130     37.1%     35%
150      0.944    150     42.9%     25%
175      0.955    175     48.9%     13%
200      0.962    200     53.9%     baseline
```

**Key findings:**
1. Recall scales smoothly with expansions — no cliff, no knee.
2. Diminishing returns: first 100 expansions get recall=0.911; next 100 add only 0.051.
3. Waste increases monotonically: 6% at 50 exp, 54% at 200 exp. Late expansions
   are increasingly unproductive.
4. Hard cutoff at 150 saves 25% blocks but loses 0.018 recall. Not quite iso-recall.

**Implication:** A per-query adaptive stopping rule could do better than a global
cutoff. Easy queries converge fast (best found at expansion 6) while hard queries
need more. A stall detector (e.g., "stop when top-k hasn't changed for W expansions")
would save blocks on easy queries without hurting hard ones.

---

## EXP-T Results: Top-t Neighbor Gating (2026-03-03)

Limit each expansion to enqueue only the top-t closest neighbors (instead of all 32).

```
max_nbr    r@k    blk/q   useful   waste
2        0.664     201     155      46
4        0.835     201     115      86
8        0.939     201      97     104
12       0.958     201      94     107
16       0.961     201      93     108
all(32)  0.963     201      92     109
```

**FAIL: blk/q = 201 for ALL values of t.** Neighbor gating does not reduce block
count. The beam search termination condition (`candidate.dist > furthest.dist`) keeps
expanding as long as good candidates exist. With selective gating, fewer but better
candidates are enqueued — enough to sustain 201 expansions.

**Why it fails:** Gating filters what gets enqueued, not what gets expanded. The
expansion count is locked at ef+1 by the beam width. Gating changes recall quality
(by filtering candidates) but not search volume.

**Side effect:** Gating reduces waste (46 vs 109 at t=2) and increases useful
expansions (155 vs 92). Each expansion is more productive, but the total count
doesn't change.

---

## Fundamental Constraint (2026-03-03)

**blk/q = ef + 1 is a hard property of beam search.** Neither int8 quantization,
neighbor gating, nor any technique that doesn't change ef or the termination
condition can reduce it. The ONLY levers are:

1. **Reduce ef** — directly reduces blocks, but trades recall (smooth curve)
2. **Per-query adaptive stopping** — stop early when converged; saves blocks on
   easy queries without hurting hard ones
3. **Graph hierarchy (HNSW)** — upper layers reduce the effective search space,
   allowing lower ef at iso-recall

---

## 1. blocks/query Lower Bound Model

From Design Space paper (Li et al., PVLDB 2025) and VeloANN:

```
page_reads ≈ (R̄ × H) / (OR(G) × n_p)
```

Where:
- R̄ = average out-degree (our m_max=32)
- H = hop count / search path length
- OR(G) = overlap ratio (fraction of neighbors co-located on same page)
- n_p = useful records per page

**Our numbers:**
- R̄=32, H≈4-5 hops (NSW flat), n_p=1 (768d×4B=3KB, one vec per 4KB page)
- OR(G) ≈ 0 (VID-ordered layout, zero locality)
- Theoretical: 32 × 5 / (0.06 × 1) ≈ 2667 (but ef=200 caps actual expansion)

**Practical bound:** DiskANN on SIFT100M (128d, 24-degree) needs ~85 page reads
at 90% recall@10 (PageANN) vs 187 baseline. VeloANN achieves ~45 IOs at 10%
buffer ratio on SIFT1M. Scaling to 100K at 768d, our 134 is in range but
improvable.

**Target (revised):** 60-80 blocks/query at recall ≥0.96. BUT: experiments show
blk/q = ef+1 is a hard property of beam search. At recall=0.96, ef≈200 is
required. Reaching <100 blocks/query at recall ≥0.96 requires either:
(a) per-query convergence detection to stop early on easy queries, or
(b) graph hierarchy (HNSW layers) to reduce the effective search space.

---

## 2. Technique 0: blocks/query Decomposition (instrument first)

Before optimizing, we need to understand WHERE blocks are spent. Instrument
`disk_graph_search` to emit per-query counters:

```
blocks_entry_phase     — expansions before any top-k candidate enters the result set
blocks_convergence     — expansions after first top-k candidate until beam exhausts
blocks_refine          — vector fetches during int8→fp32 refinement (R parameter)
unique_blocks          — distinct block IDs actually read (dedup via VisitedSet)
repeated_blocks        — expansion attempts that hit already-visited nodes
blocks_before_best_improves — expansions before the best (rank-1) result first appears
```

**Why this must come first:**
- "Approach phase wastes blocks" is a hypothesis, not a measurement
- DynamicWidth's ROI depends on how many blocks are actually spent in approach
- MemGraph's ROI depends on how many hops happen before reaching the query region
- Without decomposition, we're guessing which technique matters most

**RESULT (2026-03-03):** Hypothesis was WRONG. Entry phase is <1%. Waste is
54% but distributed across convergence, not concentrated in approach. See
EXP-0 results above.

**Implementation:** Add `SearchDiagnostics` struct to `search.rs`, populated during
beam search. Return alongside results. Zero overhead when not compiled with
`#[cfg(feature = "diagnostics")]` or always-on for now (counters are just increments).

**Deliverable:** Table like:

```
mode              blk/q  entry  conv  refine  unique  repeated  best_appears_at
fp32 ef=200        134    ??     ??     0       ??       ??         ??
fp32 ef=180        124    ??     ??     0       ??       ??         ??
int8+ref R=2k      134    ??     ??    ??       ??       ??         ??
```

This table tells us exactly where to cut.

---

## 3. Four Techniques to Reduce Page Reads (ordered by ROI)

### Technique A: DynamicWidth (highest ROI, zero memory cost)

**Source:** Design Space paper Section 4.3.1

Start beam search with small ef during "approach phase" (far from query),
grow ef as convergence is detected. Early hops are navigation — expanding
many neighbors wastes IO on distant nodes.

**Insertion point:** `search.rs` beam loop, before candidate expansion.

```
// Pseudocode
if phase == Approach && improvement_rate < threshold {
    effective_ef = ef_min  // e.g., ef/4
} else {
    effective_ef = ef      // full width for convergence
}
```

**Expected impact:** 25% page-read reduction (Design Space Figure 22).
Our 134 → ~100 blocks/query.

**Why it works for us:** Our NSW flat graph has no hierarchy. Early hops
from entry points are pure navigation through hub nodes — most expanded
neighbors are discarded. DynamicWidth avoids expanding the full ef=200
beam during this phase.

### Technique B: Neighbor-PQ-in-AdjBlock (free next-hop scoring per IO)

**Source:** DistributedANN Section 4.2, PageANN Section 4.2 (adapted)

At build time, copy each neighbor's PQ/int8 code INTO the adjacency block.
When we read a 4KB adjacency block for node v, we get v's neighbor list AND
their cheap distance codes — enabling next-hop scoring without additional IO.

**Why this replaces PageSearch:** PageSearch requires PageShuffle (NP-hard
page assignment) to co-locate related records. That's deferred indefinitely.
Neighbor-PQ-in-AdjBlock achieves the same "free computation per IO" goal
with zero layout changes — just a build-time copy.

**Block layout change:**

```
Current 4KB block:  [degree(u16) | neighbor_vids(u32 × 32) | padding]
                     = 2 + 128 = 130 bytes used, 3966 bytes wasted

Proposed 4KB block: [degree(u16) | neighbor_vids(u32 × 32) | neighbor_codes(768B × 32)]
                     = 2 + 128 + 24576 = needs ~24KB → does NOT fit in 4KB
```

**Problem:** At dim=768, even int8 codes are 768 bytes per neighbor. With
degree=32, that's 24KB — far exceeds 4KB block size. Options:
- (a) Increase block size to 32KB (wastes IO bandwidth on low-degree nodes)
- (b) Use PQ codes instead of int8 (e.g., PQ16×8 = 16 bytes/neighbor → 512B for 32 neighbors → fits easily)
- (c) Truncate: store codes for top-16 neighbors only (512B int8 × 16 = 12KB, still too big with int8)

**Decision:** Viable only with PQ codes (option b). With PQ16×8, 32 neighbors
cost 512 bytes — fits in one 4KB block with the neighbor list. But we don't
have PQ yet. **Defer until PQ codebook is implemented.** Revisit as part of
the compression roadmap (post-MVP Opt-A).

**Expected impact (when viable):** Eliminates ~50% of convergence-phase IOs
(score neighbors before deciding to expand them). Design Space reports 28%
page-read reduction from the PageSearch concept.

### Technique C: In-Memory Navigation Graph (MemGraph)

**Source:** Design Space Section 4.1.3, DistributedANN head index

Build a small in-memory navigation graph over ~0.1% of vectors (100 vectors
for 100K dataset, ~1000 for 1M). Use it as first-stage search to find a
high-quality entry point close to the query, eliminating wasteful "approach
phase" hops on disk.

**Insertion point:** Before `disk_graph_search`, add in-memory pre-search.

```
// 1. Search small in-memory graph for best entry point
let coarse_results = mem_graph.search(query, 1, 32);
let entry_point = coarse_results[0].id;
// 2. Start disk search from this entry point instead of random hubs
disk_graph_search(query, &[entry_point], k, ef, ...).await;
```

**Expected impact:** 32-54% page-read reduction (Design Space Figure 22,
largest single-factor gain). Our 134 → ~70-90 blocks/query.

**Memory cost:** ~30-50 MB for 100K dataset. For 1M: ~300 MB. Acceptable.

**Why this is the most impactful:** Our current entry_set is 64 random hub
nodes. Many initial hops are wasted navigating from these hubs toward the
query region. A good entry point starts the search closer to the true
nearest neighbors, cutting the approach phase.

**Measurable indicators** (from Technique 0 instrumentation):
- `blocks_entry_phase` should drop significantly (this is where MemGraph cuts)
- `blocks_before_best_improves` should decrease (better entry → find good
  results earlier)
- `blocks_convergence` should stay roughly the same (convergence work is
  inherent to the graph structure, not the entry point)

The actual waste is `blocks_entry_phase` — measure it first with Technique 0
before predicting the gain. Design Space's 32-54% figure is averaged across
datasets and may not transfer directly to Cohere 768d.

**EXP-0 RESULT (2026-03-03):** Entry phase = 0.6% of total expansions.
First top-k result enters beam at median expansion 1. MemGraph would save
~1.3 expansions out of 201 = **<1% improvement**. The 32-54% gains from
Design Space do not transfer to our NSW flat graph on Cohere 768d.
MemGraph is NOT worth implementing for this workload.

---

### Technique D: Prefetching Coroutine (overlaps IO with compute)

**Source:** VeloANN Section 4.3

Don't block on a single IO — issue multiple adjacency reads concurrently
and process results as they complete. VeloANN reports 2.15x throughput
improvement from this alone (their biggest single-factor gain).

**Insertion point:** `search.rs` beam loop — instead of sequential
`pool.get_or_load(vid).await` for each candidate, batch multiple candidates
and issue reads in parallel via io_uring SQE batching.

**Expected impact:** Does NOT reduce blocks/query — reduces wall-clock time
per query by overlapping IO latency. Complementary to Techniques A/C which
reduce block count.

**Prerequisite:** AdjacencyPool must exist first (cache dedup prevents
duplicate in-flight reads for the same block).

---

## 4. graph-replicated 4KB Block Decision Table

From PageANN and VeloANN: should we co-locate vectors + adjacency in one block?

### Current layout (separate files)
```
adjacency.dat: [vid → 4KB block with neighbor list]
vectors.dat:   [vid → flat f32 array, memory-mapped]
```

### Co-located layout (PageANN-style)
```
pages.dat: [page → representative vec + neighbor list + PQ codes of neighbors]
```

| Factor | Current | Co-located | Verdict |
|--------|---------|------------|---------|
| IOs per hop | 1 (adj only, vecs in DRAM) | 1 (everything in page) | Tie if vecs fit in DRAM |
| IOs per hop (vecs on disk) | 2 (adj + vec) | 1 | Co-located wins |
| Vecs per page (768d, fp32) | N/A | 1 (3KB vec + 1KB adj) | Too few — PageANN advantage vanishes |
| Vecs per page (768d, int8) | N/A | 3-4 (768B vec) | Viable with compression |
| Write amplification | 1x | 2-3x (PQ codes duplicated) | Current wins |
| Build complexity | Simple | NP-hard page assignment | Current wins |
| Space overhead | 1x | 1.3-2x | Current wins |
| Update cost | O(1) per vector | Rebuild pages | Current wins |

**Decision:** NOT worth it at 768d with fp32 vectors. Only 1 vector fits per
4KB page, so PageANN's advantage (multiple useful vecs per read) vanishes.
Co-location becomes viable only with compression (int8: 4 vecs/page, or
RaBitQ: 8+ vecs/page).

**Revisit when:** We add compressed on-disk vectors (ExtRaBitQ/PQ) AND the
dataset exceeds DRAM capacity for vectors.

---

## 5. Four Micro-Experiments (ordered by priority)

### EXP-0: blocks/query Decomposition (run FIRST)

**Setup:** Instrument `disk_graph_search` to emit `SearchDiagnostics`.
Run on Cohere 100K with fp32 ef=180, fp32 ef=200, int8+ref R=2k ef=200.

**Measure:** Full decomposition table (see Technique 0 above).

**Pass if:** We can fill in every `??` in the deliverable table. No
performance target — this is pure measurement.

**Why first:** Every subsequent experiment's pass/fail criteria depend on
knowing the decomposition. Without it, we're optimizing blind.

### EXP-C: Iso-Recall blocks/query Curve

**Setup:** Sweep ef from 100 to 300 (step 20) for both FP32 and int8+refine
R=2k. For each (mode, ef) pair, record recall@100 and blocks/query.

**Measure:** For each recall level achievable by both modes, compare
blocks/query. Primary output: Pareto frontier of recall vs blocks/query.

**Primary KPI:** At iso-recall (e.g., both modes at recall ≈ 0.96), compare
blocks/query. This is the STRUCTURAL test — can int8's cheap traversal
achieve the same recall with fewer block reads?

**Pass if:** There exists a recall level ≥ 0.96 where
`blocks_i8 ≤ 0.85 × blocks_fp32`. That is, int8+refine uses ≥15% fewer
blocks at the same recall. If this fails, int8's advantage is purely
compute and will shrink under IO.

### EXP-B: Entry Point Quality (approach phase reduction)

**Setup:** Compare current 64-hub entry set vs:
  (a) single centroid of all vectors
  (b) small in-memory NSW over 1000 sampled vectors, search for 1 best entry

**Measure:** blocks/query (total AND decomposed: entry vs convergence phase
from EXP-0 instrumentation), recall@100, p50/p99 compute latency.

**Primary KPI:** `blocks_entry_phase` reduction. This is the SPECIFIC phase
that MemGraph targets. Total blocks/query may also drop, but the mechanism
must be measurable in the entry phase.

**Pass if:** Total blocks/query drops ≥30% with option (b) AND recall ≥ 0.95
AND p99 does not increase by more than 20% (no tail blowup).

### EXP-A: DynamicWidth (iso-recall + iso-refine-budget)

**Setup:** Modify `disk_graph_search` to use ef_min=ef/4 for the first
N expansions (N=3,4,5), then switch to full ef. Run on both FP32 and
int8+refine R=2k.

**Measure:** blocks/query, recall@100, p50/p99 compute latency. For int8,
hold R=2k constant (iso-refine-budget) — DynamicWidth must not compensate
by letting refine do more work.

**Primary KPI:** blocks/query at iso-recall. Find the DynamicWidth config
that matches baseline recall (within 0.01), then compare blocks/query.

**Pass if:** blocks/query drops ≥15% at iso-recall (within 0.01 of baseline)
AND iso-refine-budget (same R for int8 mode).

---

## 6. Key Insights from Each Paper

### Design Space (Li et al., PVLDB 2025) — OctopusANN
- Winning combo: MemGraph + PageShuffle + PageSearch + DynamicWidth
- MemGraph alone: 54% QPS improvement, 32.5% page-read reduction
- DynamicWidth alone: 12.5% QPS, 25.2% page-read reduction
- Pipeline search HURTS at high concurrency (Finding 5) — don't do it
- Track IO utilization: U_io = N_eff / N_read

### VeloANN (Zhao et al., PVLDB 2026)
- Record-level buffer pool >> page-level (our AdjacencyPool is correct)
- Compression is the IO multiplier: ExtRaBitQ gets ~4.5x compression,
  multiple records per page, enabling co-placement
- Prefetching coroutine: biggest single throughput jump (2.15x)
- Cache-aware beam search: opportunistic in-memory pivoting when
  candidates happen to be cached
- Batch size formula: B = ceil(α × I / T) for coroutine count tuning
- 10x smaller disk footprint than DiskANN via compression

### PageANN (2025)
- Page-node concept: lift graph granularity from node to page
- 46% IO reduction on SIFT100M (85 vs 187 page reads)
- Key trick: embed PQ codes of neighbors IN the page → zero extra IO for
  next-hop scoring
- NOT worth it at high dimensionality (768d: 1 vec per 4KB page)
- Viable only with vector compression

### DistributedANN (Microsoft, 2025)
- Single unified graph >> partitioned sub-graphs (7.5x fewer IOs)
- Duplicate PQ codes into adjacency blocks (our design matches)
- IO budget as first-class query parameter (expose max_ios alongside ef)
- Head index eliminates early-hop IOs

### ANN-Cache (2025)
- Caches QUERY RESULTS, not blocks (different layer than our pool)
- 4.2-7.25x reduction in disk reads via result caching
- Workload skew is massive: 90% queries hit 10% vectors (Last.fm)
- Cache size sweet spot: ~40% of working set
- For our AdjacencyPool: TinyLFU/CLOCK with frequency awareness is correct

### IISWC Measurement (2025)
- Existing systems (Milvus+DiskANN) use only 8.9% of NVMe bandwidth
- 99.99% of IOs are 4KB random reads
- CPU is the bottleneck, not SSD — serial dependency chain is the killer
- Superlinear throughput scaling only at small scale (fits in page cache)
- Key metric to beat: sustain >2 GiB/s of useful 4KB random reads

---

## 7. Priority Order (updated 2026-03-03 after all 4 experiments)

### Completed experiments:
- **EXP-0** ✓ — Entry phase is <1%. 54% of expansions are wasted (convergence, not approach).
- **EXP-C** ✓ FAIL — int8 provides zero IO advantage. blk_ratio = 1.00 at iso-recall.
- **EXP-W** ✓ — Hard cutoff saves 25% blocks at 0.018 recall loss. Smooth curve, no cliff.
  Per-query adaptive stopping is the viable path (easy queries converge fast, hard ones need more).
- **EXP-T** ✓ FAIL — Neighbor gating does NOT reduce blocks. blk/q = 201 for all values of t.
  Gating controls enqueue quality, not expansion count.

### What the data tells us:
- **blk/q = ef + 1** — this is the fundamental constraint. Reducing blocks = reducing ef
  or changing the termination condition.
- **Int8 cannot reduce ef** at iso-recall — quantization errors require the same beam width.
- **MemGraph won't help** — entry phase is <1%, not the 30-50% expected.
- **DynamicWidth targets the wrong phase** — approach phase is trivial, waste is in convergence.
- **Neighbor gating is a no-op for blocks** — fewer enqueued neighbors ≠ fewer expansions.
- **Convergence budgeting works** — diminishing returns are real (first 100 exp → 0.911 recall,
  next 100 → +0.051). Per-query adaptive stopping can exploit this without a global cutoff.

### Revised priorities (what CAN actually reduce blocks or latency):
1. **AdjacencyPool (cache)** — warm cache already reduces effective misses from 201 to
   134 per query. This is the foundation for everything else: prefetching, dedup, and
   latency hiding all require a cache layer. Plan exists, implement next.
2. **Prefetching coroutine** — doesn't reduce blocks but overlaps IO with compute.
   VeloANN's 2.15x throughput gain. Requires AdjacencyPool (dedup prevents duplicate
   in-flight reads). Implement immediately after cache.
3. **Per-query adaptive stopping** — EXP-W proved the opportunity: 54% of expansions are
   wasted, and recall scales smoothly. A stall detector (e.g., "stop when top-k hasn't
   changed for W expansions") saves blocks on easy queries without hurting hard ones.
   Implement as a search option after cache + prefetch are stable.
4. **Graph quality (HNSW hierarchy)** — the only structural way to reduce ef at iso-recall.
   NSW flat has no hierarchy, so the beam explores exhaustively. HNSW layers would let
   lower ef achieve the same recall. Bigger project, defer to graph builder phase.

### Do NOT pursue:
- MemGraph (entry phase <1%, ROI near zero)
- DynamicWidth approach-phase variant (approach phase trivial)
- Neighbor gating / top-t filtering (doesn't reduce blocks — EXP-T proved this)
- PageShuffle (NP-hard build complexity)
- Co-located layout (768d too fat)
- Pipeline search (hurts at high concurrency)
- Query-result caching (wrong layer for now)

### Defer to compression roadmap:
- Neighbor-PQ-in-AdjBlock (needs PQ codebook first)
- Co-located layout (needs int8/RaBitQ on-disk vectors)
