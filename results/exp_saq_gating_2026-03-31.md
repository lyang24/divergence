# EXP-SAQ-GATE Results — NO-GO

**Date**: 2026-03-31
**Instance**: i4i.xlarge (NVMe), EC2 us-east-1
**Dataset**: Cohere 100K, dim=768, k=100, ef=200, W=4, 5% cache, perq-cold
**SAQ**: 1 segment, 4-bit, rotation=true, packed 396 B/vec (7.8x DRAM savings vs FP32)

## Verdict: NO-GO

blk/q did NOT decrease. Hits No-Go criterion #1: blk/q reduction < 10% at recall >= 0.960.

## SAQ Gating + FP16 Refine R=160 (heavy_edge layout)

| gate_ratio | recall | blk/q | p50ms | p99ms | QPS  | exp/q | waste% | gate_scored/q | gate_filtered% | dst_ms | ref_ms | ref_KB/q |
|-----------|--------|-------|-------|-------|------|-------|--------|---------------|----------------|--------|--------|----------|
| 1.00      | 0.963  | 201.1 | 10.4  | 12.5  | 87.0 | 201.1 | 53.9%  | 0.0           | 0.0%           | 1.51   | 3.46   | 240      |
| 0.75      | 0.963  | 201.1 | 10.0  | 12.1  | 90.1 | 201.1 | 53.9%  | 3263.9        | 22.6%          | 1.18   | 3.45   | 240      |
| 0.50      | 0.963  | 201.1 | 9.9   | 11.9  | 90.6 | 201.1 | 53.5%  | 3440.4        | 48.0%          | 1.21   | 3.44   | 240      |
| 0.33      | 0.962  | 201.1 | 10.0  | 12.0  | 90.2 | 201.1 | 51.9%  | 3631.4        | 63.7%          | 1.19   | 3.44   | 240      |
| 0.25      | 0.960  | 201.1 | 10.0  | 12.0  | 90.1 | 201.1 | 49.9%  | 3760.1        | 71.2%          | 1.24   | 3.43   | 240      |

## SAQ Gating + FP16 Refine R=160 (bfs layout)

| gate_ratio | recall | blk/q | p50ms | p99ms | QPS  | exp/q | waste% | gate_scored/q | gate_filtered% | dst_ms | ref_ms | ref_KB/q |
|-----------|--------|-------|-------|-------|------|-------|--------|---------------|----------------|--------|--------|----------|
| 1.00      | 0.963  | 201.1 | 11.2  | 12.9  | 80.6 | 201.1 | 53.9%  | 0.0           | 0.0%           | 1.50   | 3.43   | 240      |
| 0.75      | 0.963  | 201.1 | 10.8  | 12.2  | 83.8 | 201.1 | 53.9%  | 3263.9        | 22.6%          | 1.20   | 3.43   | 240      |
| 0.50      | 0.963  | 201.1 | 10.9  | 12.5  | 82.9 | 201.1 | 53.5%  | 3440.4        | 48.0%          | 1.22   | 3.44   | 240      |
| 0.33      | 0.962  | 201.1 | 11.0  | 12.5  | 83.2 | 201.1 | 51.9%  | 3631.4        | 63.7%          | 1.24   | 3.42   | 240      |
| 0.25      | 0.960  | 201.1 | 11.0  | 25.7  | 77.4 | 201.1 | 49.9%  | 3760.1        | 71.2%          | 1.26   | 3.55   | 240      |

## Analysis

**What happened**: Gating filters up to 71.2% of SAQ-scored neighbors, but exp/q and blk/q
remain exactly 201.1 across all gate_ratios. The search expands the same number of nodes
regardless of how aggressively we gate.

**Why**: The gated-out neighbors were being pushed to the candidate heap in the ungated path,
but they never won as the next expansion candidate. The beam's greedy best-first ordering
means the winning candidates at each step are always in the top-T by SAQ distance. Filtering
the bottom 75% of neighbors has zero effect on the expansion sequence because those neighbors
were never competitive.

**Minor effects observed**:
- dst_ms drops ~20% (1.51 -> 1.18ms) because fewer neighbors are pushed/compared in the heap
- waste% drops slightly (53.9% -> 49.9%) confirming some gated neighbors would have been
  wasted expansions, but they never displaced the actual winners
- p50 improves ~0.4ms from reduced heap overhead, not from IO reduction
- p99 spikes at gate_ratio=0.25 on bfs layout (25.7ms), suggesting occasional pathological
  behavior when gating is too aggressive

**Conclusion**: SAQ neighbor gating cannot reduce IO because beam search on this graph
already follows the optimal path as ranked by SAQ distance. The "wasted" expansions are
wasted because they find no better neighbors, not because they were reached via bad
neighbors. Reducing the candidate set doesn't change which nodes get expanded.

This invalidates the core hypothesis from inline_codes_proposal.md: "By gating to only the
Top-T best SAQ-scored neighbors per expansion, we reduce candidates -> fewer expansions ->
fewer page reads." The first step (reduce candidates) works, but it doesn't lead to fewer
expansions because the expansion order is unchanged.

## Baseline Reference (heavy_edge, perq-cold, no gating)

| label           | recall | blk/q | p50ms | p99ms | QPS   |
|-----------------|--------|-------|-------|-------|-------|
| FP32-cosine     | 0.963  | 201.1 | 7.3   | 9.2   | 119.6 |
| SAQ-pack4       | 0.901  | 201.1 | 6.8   | 8.9   | 126.5 |
| SAQ+ref-R160    | 0.963  | 201.1 | 8.6   | 10.5  | 102.6 |
| f16+ref-R160    | 0.963  | 201.1 | 10.2  | 12.3  | 87.5  |
| f16+R160-pin128 | 0.963  | 201.1 | 9.8   | 11.7  | 91.3  |
