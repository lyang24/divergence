# DiskANN vs Divergence: Fair Comparison Methodology

## Dataset

- **Cohere 100K**: 100,000 vectors, 768 dimensions, cosine metric
- **Queries**: 1,000 test queries
- **Ground truth**: Brute-force top-100 (exact), computed once, shared by both systems
- **Normalization**: All vectors L2-normalized. Verified: `||x||` min/max/mean within 1% of 1.0
- **Format**: DiskANN `.bin` (8B header `[u32 npoints][u32 ndims]` + row-major f32)
- **Integrity**: SHA256 checksums of raw payloads stored in `checksums.sha256`.
  Divergence reads headerless files; DiskANN reads headered files.
  Both contain identical payloads (verified by matching SHA256).

## Metric Equivalence

Since all vectors are unit-normalized:
- L2 distance: `||q - x||² = 2 - 2·dot(q, x)`
- Cosine distance: `1 - dot(q, x)`

L2² and cosine distance produce identical rankings on the unit sphere.
DiskANN uses `--dist_fn l2`. Divergence uses `MetricType::Cosine` (inner product on normalized vectors).

**Verification**: The export script checks norm statistics and aborts if vectors are not
unit-normalized (unless `--normalize` is passed).

## Architectural Differences (must be reported, not hidden)

| Aspect | DiskANN | Divergence |
|--------|---------|------------|
| Graph | Vamana (alpha>1 pruning, 2-pass) | NSW (flat, no hierarchy) |
| Disk layout | 4KB sector: adjacency + co-located vector | v3: page-packed adjacency (4KB), vectors in DRAM |
| IO model | O_DIRECT + io_uring (batch-and-wait) | O_DIRECT + io_uring (prefetch pipeline, W=4) |
| Vector scoring | From disk (co-located) or PQ | From DRAM (VectorBank) |
| Cache | OS page cache (disabled for fair test) | AdjacencyPool (set-associative, explicit) |

Key implication: Divergence pays 0 IO for vector scoring (DRAM), DiskANN pays 0 extra IO
because vectors are co-located in the same 4KB sector. Different tradeoffs, same result
for adjacency-dominated workloads.

## Controlled Variables

- Same dataset + ground truth (SHA256 verified)
- Same hardware: EC2 i4i instance (NVMe)
- Single-thread search (1 core) for both
- Cold cache: `echo 3 > /proc/sys/vm/drop_caches` before each sweep point
- O_DIRECT: both systems use O_DIRECT (no OS page cache interference)
- No in-memory caching for DiskANN: `--num_nodes_to_cache 0`
- Divergence: per-query cold (`pool.clear()` between queries) for intra-query-only comparison

## DiskANN Build Parameters

Using `build_disk_index` (disk-index mode, NOT async-index-build):
- `R`: max graph degree (sweep: 32, 64)
- `L`: build search list size (200)
- `--PQ_disk_bytes 0`: full-precision vectors on disk (no PQ, eliminates compression variable)
- `--search_DRAM_budget`: memory budget for search (in GB)

Note: `alpha` (Vamana pruning parameter) is NOT a parameter of `build_disk_index`.
It is fixed internally. Do not confuse with `async-index-build` which exposes alpha.

## Divergence Build Parameters

- `m_max=32`: max graph degree (matches DiskANN R=32)
- `ef_construction=200`: build search list size
- v3 BFS reorder: physical page layout optimization
  - Logical VIDs are unchanged (BFS reorder is physical-only in Divergence v3)
  - Recall computation uses original IDs, no mapping needed
  - This is a Divergence-specific property, not a general claim about BFS reorder

## Search Parameter Sweep (iso-recall comparison)

**DiskANN**: sweep `search_list` (L) = {100, 150, 200, 250, 300, 400}
**Divergence**: sweep `ef` = {100, 150, 200, 250, 300, 400}

These are NOT equivalent parameters — DiskANN's L and Divergence's ef have different
semantics. The correct comparison is iso-recall:

1. For each system, find the parameter setting that achieves recall@100 >= 0.96
2. Compare QPS, p50, p99, and physical IO at that recall level
3. Plot recall-vs-QPS and recall-vs-IO Pareto curves for both systems

## IO Measurement

### Divergence
- Internal: `SearchPerfContext.phys_reads` (miss loads + prefetch loads + bypasses)
- OS-level: `/proc/diskstats` read_ios delta before/after search run
- Both should agree within ~5% (small overhead from metadata reads)

### DiskANN
- OS-level only: `/proc/diskstats` read_ios delta before/after search run
- DiskANN uses O_DIRECT, so OS page cache does not absorb reads
- `read_ios_per_query = delta_read_ios / total_queries`

### Validation
- Disable readahead if possible: `blockdev --setra 0 /dev/nvmeXnY` (optional, needs root)
- For O_DIRECT workloads, readahead has minimal effect
- Ensure no other IO activity on the NVMe device during benchmark

## Memory Budget (must be reported)

| Metric | DiskANN | Divergence |
|--------|---------|------------|
| Index size on disk | Reported | Reported |
| Runtime RSS | `/proc/<pid>/status VmRSS` | `/proc/self/status VmRSS` |
| Cache config | `--num_nodes_to_cache` | `pool_bytes` (AdjacencyPool) |
| Pinned pages | N/A | `pin_pages` count |
| Vector memory | On disk (co-located) | Full DRAM (VectorBank) |

Divergence's "vectors in DRAM" is a significant memory advantage for scoring
but means its total memory footprint is higher. This must be quantified:
`vector_memory = n * dim * 4 bytes` = 100K * 768 * 4 = 294 MB for Cohere 100K.

## Expected Outcome

This is NOT about proving one system is "better". The comparison should reveal:

1. **Graph quality**: Vamana (DiskANN) vs NSW (Divergence) — does alpha>1 pruning
   produce sparser graphs that need fewer hops?
2. **IO efficiency**: DiskANN's co-located layout vs Divergence's page-packed adjacency
   + DRAM vectors — which does fewer physical reads per query?
3. **Prefetch effectiveness**: DiskANN's batch-and-wait vs Divergence's pipeline prefetch
   — which hides IO latency better?
4. **Memory tradeoff**: Divergence uses ~294MB more DRAM for vectors. Is the IO savings
   worth the memory cost?
