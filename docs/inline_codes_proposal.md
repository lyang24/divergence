# EXP-SAQ-GATE: SAQ Neighbor Gating to Reduce blk/q

**Date**: 2026-03-31
**Status**: PROPOSAL
**Prior art**: PageANN (46% IO reduction via score-before-expand), VeloANN (cache-aware beam search)

---

## 1. Goal

Reduce blk/q from ~201 to <140 at iso-recall (0.963) by gating which neighbors enter the candidate heap during SAQ graph traversal.

**Current bottleneck**: 54% of expansions are wasted (add 0 neighbors to beam). Every non-visited neighbor gets SAQ-scored and, if non-dominated, pushed into the candidate heap — generating a future page read. There's no limit on how many neighbors get pushed per expansion.

**Key insight**: SAQ codes are already in DRAM. We already compute SAQ distances for ALL neighbors. But we push ALL non-dominated neighbors — even marginal ones that will never improve top-k. By gating to only the **Top-T best SAQ-scored neighbors** per expansion, we reduce candidates → fewer expansions → fewer page reads.

**Why not inline PQ?** Our v3 pipeline already has SAQ codes in DRAM via `SaqVectorBank`. SAQ scoring is zero-IO. The gating can use the SAQ distance directly — no need for a separate PQ codebook, no storage format changes, no rebuild. Pure search-side optimization.

---

## 2. Implementation

### Single change: gate in `disk_graph_search_pipe_v3_inner`

Add `gate_ratio: f32` and `gate_min: usize` parameters. After SAQ-scoring all non-visited neighbors of an expansion, use `select_nth_unstable_by` (O(n) partial sort) to keep only the top-T, where T = max(gate_min, ceil(degree × gate_ratio)).

**Critical correctness constraint**: gated-out neighbors must NOT be marked visited. They may be reachable via a different, better path later. Only mark visited when committing to push.

**No storage changes. No new files. No PQ codebook. No block format change.**

### Files changed

| File | Change |
|------|--------|
| `crates/engine/src/search.rs` | Add gate_ratio/gate_min to v3_inner, gating logic, new `_gated` wrapper functions |
| `crates/engine/src/lib.rs` | Export new functions |
| `crates/engine/tests/disk_search.rs` | `exp_saq_gating` test |

See `docs/impl_saq_gating.md` for full implementation specification.

---

## 3. Expected Key Results

**EXP-SAQ-GATE**: Cohere 100K, dim=768, k=100, ef=200, W=4, 5% cache, perq-cold

| gate_ratio | Expected recall | Expected blk/q | Expected p50 |
|-----------|----------------|-----------------|--------------|
| 1.0 (baseline) | 0.963 | ~201 | ~9.1ms |
| 0.75 | 0.963 | ~170 (-15%) | ~8.0ms |
| 0.50 | 0.960-0.963 | ~140 (-30%) | ~7.0ms |
| 0.33 | 0.955-0.960 | ~120 (-40%) | ~6.0ms |
| 0.25 | 0.950-0.955 | ~110 (-45%) | ~5.5ms |

With FP16 refine (R=160), the blk/q savings translate directly to p50 savings since refine_ms is constant (~1.7ms).

---

## 4. Results to Verify (Go/No-Go)

**Go criteria** (all must pass):
1. gate_ratio=1.0 produces **identical** recall and blk/q to current code (baseline equivalence)
2. blk/q reduction ≥ 20% at recall ≥ 0.960
3. p50 latency improves (IO savings not eaten by partial-sort overhead)
4. waste% decreases (fewer wasted expansions)

**No-Go criteria** (any one kills the experiment):
1. blk/q reduction < 10% at recall ≥ 0.960 (gating doesn't reduce expansions — beam structure forces same path)
2. recall drops > 2% at gate_ratio=0.50 (SAQ proxy too noisy for gating)
3. p50 increases despite blk/q decrease (partial-sort overhead dominates)

### Diagnostic metrics to report

| Metric | What it tells us |
|--------|-----------------|
| blk/q | Primary: IO reduction |
| recall | Primary: accuracy preservation |
| p50, p99 | Primary: latency |
| gate_scored/q | Neighbors SAQ-scored per query (before gating) |
| gate_passed/q | Neighbors that passed gate |
| gate_filtered% | 1 - passed/scored — effective filter rate |
| waste% | Wasted expansions (should decrease) |
| dst_ms | Time in SAQ distance per query |

---

## 5. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Gating starves candidate heap → early termination → recall cliff | Medium | gate_min=4 floor; start conservative at 0.75 |
| SAQ distance ranking too noisy for gating (σ ≈ 0.5) | Medium | If SAQ is too noisy, this becomes the empirical proof that inline PQ (finer granularity) is needed |
| Partial-sort overhead per expansion | Low | select_nth_unstable is O(n) for n=32; trivial vs IO cost |
| Beam converges to different (worse) region | Low | Monitor recall curve shape; if recall drops discontinuously, beam topology is changing |

---

## 6. If SAQ Gating Fails

If SAQ proxy distances are too noisy for effective gating (σ ≈ 0.5 relative error), this proves we need a **higher-quality** inline proxy. That's when inline PQ codes (from `inline_pq_design.md`) become the next step:
- PQ32 (32 bytes/neighbor) fits in v3 pages
- PQ distance is a different error profile (systematic quantization vs SAQ's rescaled approximation)
- PQ gating + SAQ scoring would be a two-level proxy: PQ gates → SAQ scores → FP16 refines

But try SAQ gating first — it's zero-cost to implement and validates whether gating helps at all before investing in PQ infrastructure.

---

## 7. Relationship to PageANN

PageANN's 46% IO reduction comes from score-before-expand with inline compressed codes. But PageANN uses a page-node graph (multiple vectors per node, merged edges) — a fundamentally different structure.

Our approach tests the **gating hypothesis** in isolation: does limiting which neighbors enter the beam reduce IO without killing recall? If yes, we get the benefit with zero infrastructure cost. If no, the problem isn't gating — it's that beam search requires exploring a minimum set of paths regardless of scoring quality.
