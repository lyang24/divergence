# Paper + Code Distillation 3.1 — IO Orchestration for Out-of-Core ANN

Context: Divergence has a validated prefetch pipeline (W=4, 60% latency reduction on NVMe).
Next: understand how others solve IO orchestration, queue depth, and layout.

---

## 1. PipeANN (OSDI'25) — Intra-Query IO Pipeline

**Source**: `~/repos/PipeANN/src/search/pipe_search.cpp`

### IO Pipeline Architecture
- Two-phase loop: `send_best_read_req()` → `poll_all()` → `calc_best_node()`
- Non-blocking `io_uring_peek_batch_cqe()` reaps up to 256 CQEs per poll
- Single SQE submit per request (`io_uring_submit()` after each add) — NO batch accumulation
- **SQPOLL enabled** for pipe_search mode (`IORING_SETUP_SQPOLL`) — eliminates per-submit syscall
- Per-thread `thread_local io_uring *ring` (no sharing across threads)

### Queue Depth Control
- Static: `on_flight_ios.size() < cur_beam_width`
- Initial W=4, ramps up by +1 when waste_ratio < 10%
- Max W = user `beam_width` (typically 32)
- Only increases, never decreases
- MAX_IO_DEPTH = 128 per thread

### Dynamic W
- Trigger: after 5+ completed batches, measure `waste_ratio = out_of_range / total_completions`
- If waste < 10%: `cur_beam_width = min(beam_width, cur_beam_width + 1)`
- Hardcoded threshold, no decrease path
- Takes ~(W-4) iterations to reach target

### Inter-Query (CoroSearch mode)
- Up to 8 concurrent queries per thread, round-robin execution
- Each query has independent state (`CoroDataOne`)
- Same io_uring ring shared across queries
- No inter-query IO batching — queries issue IO independently

### Key Numbers
- SIFT100M, recall@10=0.99: <1ms p50, <4ms p99
- 1B SPACEV: ~2ms p50, 20K QPS
- "1.14-2.02x of in-memory index" latency

### What to Steal
- **SQPOLL**: eliminates syscall per submit — check if monoio exposes this
- **Waste-ratio W tuning**: simple, proven. Our stall detector aligns with this
- **peek_batch_cqe for non-blocking reap**: batch completion polling is strictly better than one-at-a-time

### What Doesn't Apply
- Their W only increases (no decrease) — we need bidirectional adaptation
- CoroSearch (B=8 per thread) — we proved B>1 explodes p99

---

## 2. DiskANN (NeurIPS'19) — Vamana + SSD BeamSearch

**Source**: `~/repos/DiskANN/diskann-disk/src/` (Rust rewrite)

### IO Architecture
- io_uring via `IoUring` Rust crate, O_DIRECT, registered FD
- Batched submissions: MAX_IO_CONCURRENCY = 128, `submit_and_wait` per batch
- **Synchronous batch model**: issue W reads → wait ALL complete → process → next round
- No IO-compute overlap within a round (overlap comes from NVMe internal parallelism)

### Disk Layout
- Fixed-size records: `[full_vector | #neighbors | neighbor_ids | padding]`
- Offset = `id * record_size` — no in-memory offset table
- Co-locate vector + adjacency in same sector — **piggyback vector for free**
- At 128d: vector (512B) + neighbors (512B) fits in 1 sector
- At 768d: vector (3072B) doesn't fit alongside neighbors in 4KB

### Caching
- Static BFS cache: nodes within C=3-4 hops of medoid, loaded at startup
- No eviction (fixed capacity HashMap)
- Exponential growth limits practical C to 3-4

### Queue Depth
- W=4-8 is optimal. Explicitly warns: "W>=16 wastes compute + SSD bandwidth"
- **SSD load factor 30-40%** for low latency — don't saturate the device queue
- Multi-thread scaling by adding threads, not by increasing per-thread QD

### Key Numbers
- SIFT1B: >5000 QPS at 95%+ recall, <3ms mean latency
- Vamana: 2-3x fewer hops than HNSW/NSG (alpha>1 long-range edges)
- Average degree ~92-113, R=128 max

### What to Steal
- **Piggyback vector in adjacency block**: at 768d Int8, vector=768B, fits in 4KB alongside ~50 neighbor IDs. Free reranking.
- **W=4-8 confirmed independently**: our W=4 is right in the sweet spot
- **30-40% device load for low latency**: validates our semaphore-based IO budgeting
- **Vamana alpha>1 two-pass construction**: for Opt-D, long-range edges reduce diameter
- **BFS-from-medoid cache warming**: principled hot-set selection for AdjacencyPool

### What Doesn't Apply
- Synchronous W-batch: our async coroutine model is strictly more flexible
- PQ as primary distance: Int8 at 768d gives better approximation
- RAID-0 dual SSD: we target single NVMe

---

## 3. PageANN (2509.25487v2) — Page-Node Graph

### Core Idea
- Graph nodes are full SSD **pages** (4KB/8KB), not individual vectors
- Each page contains multiple co-located vectors + neighbor IDs + **compressed neighbor vectors inline**
- One graph hop = exactly one page read, every byte is useful
- Topology-guided packing: cluster vectors by graph neighborhood

### IO Architecture
- Linux AIO (`io_submit/io_getevents`), not io_uring
- **IO batch size b=5**: collect 5 page IDs, deduplicate, submit as single batch
- Page-level dedup: skip pages already visited or scheduled
- IO-compute pipeline: process completed reads while waiting for next batch

### Key Numbers (SIFT100M, recall@10=0.9, 30% memory)
- PageANN: 2749 QPS, 5.78ms latency, 85 mean IOs
- DiskANN: 1100 QPS, 14.45ms latency, 187 mean IOs
- **2.5x throughput, 2.5x lower latency, 46% fewer IOs** vs DiskANN
- At SIFT1B: 1.9-3.8x throughput, 48-71% latency improvement

### What to Steal
- **Inline compressed neighbors**: store SQ/PQ codes for neighbors inside adjacency block. Next-hop distance estimation requires ZERO additional IO. Highest-value idea.
- **Batch IO with page-level dedup**: collect W candidates, dedup by block ID, submit as single io_uring batch
- **Topology-guided page packing**: cluster co-visited vectors in same block at index build time

### What Doesn't Apply
- Page-node graph abstraction: requires complete index redesign
- LSH routing: we have graph entry points
- Linux AIO: we use io_uring

---

## 4. CXL-ANNS (ATC'23) — CXL Memory Disaggregation

### Core Idea
- Places entire graph + embeddings in CXL Type 3 memory (TBs of DRAM via PCIe)
- CXL memory: 3.9x slower than local DRAM, but load/store accessible
- DSA at each CXL endpoint computes sub-distances in parallel

### What to Steal (concepts only, no CXL hardware)
- **82.3% next-node prediction from candidate array**: our CandidateHeap is an excellent predictor of next visits. Validates our prefetch lookahead approach.
- **Urgent/deferrable subtask split**: node selection (urgent, critical path) vs kNN heap maintenance (deferrable, overlap with IO wait)
- **Hop-count cache warming**: nodes at hop 1-3 from medoid dominate visit counts. BFS-based warm set is principled.
- **Distance = 81.8% of latency**: when data is in cache, compute dominates. Int8 speedup attacks the right thing.

### What Doesn't Apply
- CXL hardware, near-data processing, vector sharding — all hardware-specific

---

## Cross-Paper Synthesis: Top 5 Actionable Items

### 1. Inline Compressed Neighbors (PageANN + DiskANN)
Store Int8 codes for top-N neighbors inside each 4KB adjacency block.
Budget: 768B (self-vector Int8) + 50×4B (neighbor IDs) + 3×768B (top-3 neighbor codes) = 3336B < 4096B.
Result: next-hop distance estimation with ZERO extra IO.

### 2. Batch IO Submission with Dedup (PageANN)
Collect all W prefetch candidates, dedup against visited + inflight, submit remaining as a single io_uring SQE batch. Reduces submission overhead.

### 3. SQPOLL Mode (PipeANN)
Enable `IORING_SETUP_SQPOLL` to eliminate per-submit syscall overhead.
Check if monoio exposes this flag. Could reduce per-IO overhead by ~1μs.

### 4. Waste-Ratio Adaptive W (PipeANN)
Simple rule: if waste_ratio < 10%, increase W by 1. Start at W=4, max at W=8.
Our 54% waste rate from EXP-0 suggests W should ramp quickly.

### 5. BFS-from-Medoid Cache Warming (DiskANN + CXL-ANNS)
Pre-warm AdjacencyPool with nodes within 3 hops of medoid/entry points.
These cover the most-visited nodes across all queries.

---

## What NOT to Do

- **B>1 intra-core concurrency**: proven harmful (PipeANN's CoroSearch, our EXP-BW)
- **W>=16**: both DiskANN and PipeANN warn against over-fetching
- **Synchronous W-batch**: our async model is strictly better than DiskANN's "wait all W"
- **Page-node graph redesign**: too invasive; inline compressed neighbors captures 80% of the value
- **PQ for 768d**: Int8 is better at our dimensionality (768B vs 32B, but 768d PQ reconstruction is expensive)
