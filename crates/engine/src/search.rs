//! Async beam search over on-disk adjacency graph.
//!
//! Vectors stay in DRAM. Adjacency blocks are read from NVMe via io_uring,
//! with per-core caching via AdjacencyPool. Each `.await` on cache miss is
//! a coroutine suspension point — monoio services other queries while
//! the NVMe read is in flight.

use std::collections::HashMap;
use std::time::Instant;

use divergence_core::distance::VectorBank;
use divergence_core::quantization::pq::PqDistanceTable;
use divergence_core::VectorId;
use divergence_index::{CandidateHeap, FixedCapacityHeap, ScoredId};
use divergence_storage::{decode_adj_block, decode_adj_block_view, page_record_vid, AdjIndexEntry};

use crate::cache::AdjacencyPool;
use crate::io::IoDriver;
use crate::perf::{PerfLevel, SearchPerfContext};

// ---------------------------------------------------------------------------
// TraceRecorder — records co-expansion pairs for TWPP layout optimization
// ---------------------------------------------------------------------------

/// Collects co-expansion edge weights during search for TWPP page packing.
///
/// Records (parent, child) pairs when both VIDs are expanded in the same query.
/// Parent = the VID whose expansion pushed child into the candidate heap.
/// Co-located parents and children on the same page saves one IO per co-expansion.
///
/// Weights are stored with symmetric keys: `(min(u,v), max(u,v))` to avoid
/// double-counting. Per-query, the recorder also tracks which VIDs were expanded
/// (for seeding the packer with high-traffic hubs).
pub struct TraceRecorder {
    /// Co-expansion pair weights: (min(u,v), max(u,v)) → count.
    pub co_expand: HashMap<(u32, u32), u32>,
    /// Node expansion counts: vid → total expansions across all queries.
    pub node_counts: HashMap<u32, u32>,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self {
            co_expand: HashMap::new(),
            node_counts: HashMap::new(),
        }
    }

    /// Pre-allocate for expected number of unique VIDs and edges.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            co_expand: HashMap::with_capacity(edges),
            node_counts: HashMap::with_capacity(nodes),
        }
    }

    #[inline]
    pub fn record_expansion(&mut self, vid: u32) {
        *self.node_counts.entry(vid).or_insert(0) += 1;
    }

    #[inline]
    pub fn record_co_expand(&mut self, parent: u32, child: u32) {
        let key = if parent <= child { (parent, child) } else { (child, parent) };
        *self.co_expand.entry(key).or_insert(0) += 1;
    }
}

/// Async beam search on disk-resident adjacency graph.
///
/// - `query`: the query vector
/// - `entry_set`: hub entry point VIDs
/// - `k`: number of results to return
/// - `ef`: beam width (ef >= k)
/// - `pool`: per-core adjacency block cache
/// - `io`: async IO driver for adjacency reads (used on cache miss)
/// - `bank`: DRAM-resident vector storage + distance function (FP32, FP16, etc.)
/// - `perf`: per-query performance counters (caller manages lifecycle via SearchGuard)
/// - `level`: profiling level (CountOnly = counters; EnableTime = + wall-clock timers)
pub async fn disk_graph_search(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    let timing = level >= PerfLevel::EnableTime;

    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    // Bounded visited set: capacity tied to max expansions.
    // For MVP, use a simple Vec<bool> sized to num_vectors.
    // TODO: replace with bounded open-addressing table for large N.
    let num_vectors = bank.num_vectors();
    let mut visited = vec![false; num_vectors];

    // Track when each VID first entered the beam (expansion number, 1-indexed).
    // 0 = seed entry (from entry_set, no expansion needed).
    let mut entered_at = vec![u32::MAX; num_vectors];

    // Snapshot cache stats before search for per-query delta
    let cache_before = pool.stats();

    // Seed from entry set (DRAM only, no IO)
    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, vid);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
            entered_at[vid] = 0; // seed entry
        }
    }

    // Beam search: expand candidates by reading adjacency blocks from disk
    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        perf.expansions += 1;
        perf.blocks_read += 1;
        let expansion_num = perf.expansions;

        // ← ONLY yield point: get adjacency block (cache hit = sync, miss = async IO)
        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                perf.wasted_expansions += 1;
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        // Compute phase: decode + distance + heap (all sync, DRAM only)
        let compute_start = if timing { Some(Instant::now()) } else { None };

        let neighbors = decode_adj_block(guard.data());
        let mut added_this_expansion = 0u32;

        for nbr_raw in neighbors {
            let nbr_idx = nbr_raw as usize;
            if nbr_idx >= num_vectors || visited[nbr_idx] {
                continue;
            }
            visited[nbr_idx] = true;

            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, nbr_idx);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;

            let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
            if !dominated {
                let scored = ScoredId {
                    distance: d,
                    id: VectorId(nbr_raw),
                };
                candidates.push(scored);
                nearest.push(scored);
                added_this_expansion += 1;
                if entered_at[nbr_idx] == u32::MAX {
                    entered_at[nbr_idx] = expansion_num as u32;
                }
            }
        }

        if added_this_expansion > 0 {
            perf.useful_expansions += 1;
        } else {
            perf.wasted_expansions += 1;
        }

        if let Some(start) = compute_start {
            perf.compute_ns += start.elapsed().as_nanos() as u64;
        }

        drop(guard); // unpin after processing neighbors
    }

    // Compute cache stats delta for this query
    let cache_after = pool.stats();
    perf.blocks_hit = cache_after.hits - cache_before.hits;
    perf.blocks_miss = (cache_after.misses - cache_before.misses)
        + (cache_after.bypasses - cache_before.bypasses);
    perf.phys_reads = cache_after.phys_reads - cache_before.phys_reads;
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);

    // Compute entry-phase diagnostics from the final result set
    if !results.is_empty() {
        let best_vid = results[0].id.0 as usize;
        perf.best_result_at_expansion = if best_vid < num_vectors {
            entered_at[best_vid] as u64
        } else {
            0
        };

        let mut first_topk = u64::MAX;
        for r in &results {
            let vid = r.id.0 as usize;
            if vid < num_vectors && (entered_at[vid] as u64) < first_topk {
                first_topk = entered_at[vid] as u64;
            }
        }
        perf.first_topk_at_expansion = if first_topk == u64::MAX { 0 } else { first_topk };
    }

    results
}

/// Experimental beam search with early-stop and neighbor gating knobs.
///
/// - `max_expansions`: stop after N expansions (0 = use ef as limit, normal behavior)
/// - `max_neighbors`: per expansion, only enqueue top-N closest neighbors (0 = all)
///
/// Used for EXP-W (convergence budgeting) and EXP-T (neighbor gating) experiments.
pub async fn disk_graph_search_exp(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    max_expansions: usize,
    max_neighbors: usize,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    let timing = level >= PerfLevel::EnableTime;

    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    let num_vectors = bank.num_vectors();
    let mut visited = vec![false; num_vectors];
    let mut entered_at = vec![u32::MAX; num_vectors];

    let cache_before = pool.stats();

    // Seed from entry set
    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, vid);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
            entered_at[vid] = 0;
        }
    }

    let expansion_limit = if max_expansions > 0 { max_expansions } else { usize::MAX };

    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        if perf.expansions as usize >= expansion_limit {
            break;
        }

        perf.expansions += 1;
        perf.blocks_read += 1;
        let expansion_num = perf.expansions;

        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                perf.wasted_expansions += 1;
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        let compute_start = if timing { Some(Instant::now()) } else { None };
        let neighbors = decode_adj_block(guard.data());

        // Score all unvisited neighbors
        let mut scored_neighbors: Vec<ScoredId> = Vec::new();
        for nbr_raw in neighbors {
            let nbr_idx = nbr_raw as usize;
            if nbr_idx >= num_vectors || visited[nbr_idx] {
                continue;
            }
            visited[nbr_idx] = true;

            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, nbr_idx);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;

            let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
            if !dominated {
                scored_neighbors.push(ScoredId {
                    distance: d,
                    id: VectorId(nbr_raw),
                });
            }
        }

        // Neighbor gating: if max_neighbors > 0, only keep the top-t closest
        if max_neighbors > 0 && scored_neighbors.len() > max_neighbors {
            scored_neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            scored_neighbors.truncate(max_neighbors);
        }

        let mut added_this_expansion = 0u32;
        for scored in scored_neighbors {
            candidates.push(scored);
            nearest.push(scored);
            added_this_expansion += 1;
            let nbr_idx = scored.id.0 as usize;
            if nbr_idx < num_vectors && entered_at[nbr_idx] == u32::MAX {
                entered_at[nbr_idx] = expansion_num as u32;
            }
        }

        if added_this_expansion > 0 {
            perf.useful_expansions += 1;
        } else {
            perf.wasted_expansions += 1;
        }

        if let Some(start) = compute_start {
            perf.compute_ns += start.elapsed().as_nanos() as u64;
        }

        drop(guard);
    }

    let cache_after = pool.stats();
    perf.blocks_hit = cache_after.hits - cache_before.hits;
    perf.blocks_miss = (cache_after.misses - cache_before.misses)
        + (cache_after.bypasses - cache_before.bypasses);
    perf.phys_reads = cache_after.phys_reads - cache_before.phys_reads;
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);

    if !results.is_empty() {
        let best_vid = results[0].id.0 as usize;
        perf.best_result_at_expansion = if best_vid < num_vectors {
            entered_at[best_vid] as u64
        } else {
            0
        };
        let mut first_topk = u64::MAX;
        for r in &results {
            let vid = r.id.0 as usize;
            if vid < num_vectors && (entered_at[vid] as u64) < first_topk {
                first_topk = entered_at[vid] as u64;
            }
        }
        perf.first_topk_at_expansion = if first_topk == u64::MAX { 0 } else { first_topk };
    }

    results
}

/// Two-stage search: cheap distance for graph traversal, exact distance for top-R refinement.
///
/// 1. Run beam search with `cheap_bank` (e.g. FP16) → collect top-ef candidates
/// 2. Re-score top `refine_r` candidates with `exact_bank` (e.g. FP32)
/// 3. Return top-k from refined scores
///
/// `refine_r` should be > k (e.g. k * 4). If the graph search returns fewer than
/// `refine_r` candidates, all are refined.
pub async fn disk_graph_search_refine(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    exact_bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    // Stage 1: graph traversal with cheap distance
    let mut candidates = disk_graph_search(
        query, entry_set, refine_r, ef, pool, io, cheap_bank, perf, level,
    )
    .await;

    // Stage 2: refine top-R with exact distance
    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    for c in &mut candidates {
        let d = exact_bank.distance(query, c.id.0 as usize);
        c.distance = d;
        perf.refine_count += 1;
    }

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    // Sort by refined distance, return top-k
    candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    candidates.truncate(k);
    candidates
}

/// Async beam search with intra-query prefetch pipeline and adaptive stopping.
///
/// Clone of `disk_graph_search` with two additions:
/// 1. **Prefetch**: before each `get_or_load`, peeks ahead into the candidate
///    heap and issues `prefetch_hint()` for the next `prefetch_window` nearest
///    candidates. `prefetch_window=0` disables prefetching.
/// 2. **Adaptive stopping**: when the top-k boundary stops improving for
///    `stall_limit` consecutive expansions, enters DRAIN mode (allows
///    `drain_budget` more expansions). If no improvement during DRAIN, stops
///    early. `stall_limit=0` disables adaptive stopping (current behavior).
///
/// Does NOT spawn or stop the prefetch worker — caller manages worker lifecycle.
pub async fn disk_graph_search_pipe(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    let timing = level >= PerfLevel::EnableTime;

    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    let num_vectors = bank.num_vectors();
    let mut visited = vec![false; num_vectors];
    let mut entered_at = vec![u32::MAX; num_vectors];

    // Snapshot cache stats before search for per-query delta
    let cache_before = pool.stats();

    // Seed from entry set (DRAM only, no IO)
    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, vid);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
            entered_at[vid] = 0;
        }
    }

    // Lookahead buffer for prefetch (stack-allocated, max 8)
    let mut lookahead = [ScoredId::default(); 8];

    // Adaptive stopping state
    let mut consecutive_stalls: u64 = 0;
    let mut prev_furthest: f32 = f32::MAX;
    let mut drain_remaining: u32 = 0;

    // Beam search: expand candidates by reading adjacency blocks from disk
    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        perf.expansions += 1;
        perf.blocks_read += 1;
        let expansion_num = perf.expansions;

        // Issue prefetch hints for upcoming candidates
        if prefetch_window > 0 {
            let w = prefetch_window.min(8);
            let count = candidates.peek_nearest(&mut lookahead[..w]);
            for i in 0..count {
                let vid = lookahead[i].id.0;
                if !pool.is_resident(vid) {
                    pool.prefetch_hint(vid);
                    perf.prefetch_issued += 1;
                }
            }
        }

        // Sample inflight IO depth before awaiting (proves pipeline depth)
        let inflight = (io.adj_capacity() - io.available_adj_permits()) as u64;
        perf.inflight_sum += inflight;
        perf.inflight_samples += 1;
        if inflight > perf.inflight_max {
            perf.inflight_max = inflight;
        }

        // Sample global inflight depth (cross-core device QD)
        if let Some(g) = io.global_inflight() {
            perf.global_inflight_sum += g as u64;
            perf.global_inflight_samples += 1;
            if g as u64 > perf.global_inflight_max {
                perf.global_inflight_max = g as u64;
            }
        }

        // ← yield point: get adjacency block (cache hit = sync, miss = async IO)
        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                perf.wasted_expansions += 1;
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        // Compute phase: decode + distance + heap (all sync, DRAM only)
        let compute_start = if timing { Some(Instant::now()) } else { None };

        let neighbors = decode_adj_block(guard.data());
        let mut added_this_expansion = 0u32;

        for nbr_raw in neighbors {
            let nbr_idx = nbr_raw as usize;
            if nbr_idx >= num_vectors || visited[nbr_idx] {
                continue;
            }
            visited[nbr_idx] = true;

            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, nbr_idx);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;

            let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
            if !dominated {
                let scored = ScoredId {
                    distance: d,
                    id: VectorId(nbr_raw),
                };
                candidates.push(scored);
                nearest.push(scored);
                added_this_expansion += 1;
                if entered_at[nbr_idx] == u32::MAX {
                    entered_at[nbr_idx] = expansion_num as u32;
                }
            }
        }

        if added_this_expansion > 0 {
            perf.useful_expansions += 1;
        } else {
            perf.wasted_expansions += 1;
        }

        if let Some(start) = compute_start {
            perf.compute_ns += start.elapsed().as_nanos() as u64;
        }

        drop(guard); // unpin after processing neighbors

        // Adaptive stopping: track top-k boundary convergence
        if stall_limit > 0 && nearest.len() >= ef {
            let cur = nearest.furthest().unwrap().distance;
            if cur < prev_furthest {
                // Improvement — reset stall counter and exit drain
                consecutive_stalls = 0;
                prev_furthest = cur;
                drain_remaining = 0;
            } else {
                consecutive_stalls += 1;
            }

            if drain_remaining > 0 {
                drain_remaining -= 1;
                if drain_remaining == 0 {
                    // DRAIN exhausted with no improvement → stop
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            } else if consecutive_stalls >= stall_limit as u64 {
                // Enter DRAIN mode
                drain_remaining = drain_budget;
                if drain_budget == 0 {
                    // No drain budget → stop immediately
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            }
        }
    }

    // Record final stall state even if not stopped early
    if !perf.stopped_early {
        perf.consecutive_stalls_at_end = consecutive_stalls;
    }

    // Compute cache stats delta for this query
    let cache_after = pool.stats();
    perf.blocks_hit = cache_after.hits - cache_before.hits;
    perf.blocks_miss = (cache_after.misses - cache_before.misses)
        + (cache_after.bypasses - cache_before.bypasses);
    perf.phys_reads = cache_after.phys_reads - cache_before.phys_reads;
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;
    perf.prefetch_consumed = cache_after.prefetch_hits - cache_before.prefetch_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);

    // Compute entry-phase diagnostics from the final result set
    if !results.is_empty() {
        let best_vid = results[0].id.0 as usize;
        perf.best_result_at_expansion = if best_vid < num_vectors {
            entered_at[best_vid] as u64
        } else {
            0
        };

        let mut first_topk = u64::MAX;
        for r in &results {
            let vid = r.id.0 as usize;
            if vid < num_vectors && (entered_at[vid] as u64) < first_topk {
                first_topk = entered_at[vid] as u64;
            }
        }
        perf.first_topk_at_expansion = if first_topk == u64::MAX { 0 } else { first_topk };
    }

    results
}

/// v3 page-packed adjacency search.
///
/// Same beam search algorithm as `disk_graph_search_pipe`, but adjacency is stored
/// as multiple variable-length records packed into 4KB pages. `adj_index` maps
/// `vid -> (page_id, offset, degree)` so the search can:
/// 1. load a page by `page_id` via AdjacencyPool + IoDriver
/// 2. read neighbors from the packed record (no heap allocation)
///
/// Important: `AdjacencyPool` is reused unchanged. The cache key becomes `page_id`
/// (not `vid`). Pool hit/miss counters therefore measure page reuse directly.
pub async fn disk_graph_search_pipe_v3(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    disk_graph_search_pipe_v3_inner(
        query, entry_set, k, ef, prefetch_window, stall_limit, drain_budget,
        pool, io, bank, adj_index, perf, level, None, 0, 1.0, 0,
    ).await
}

/// Traced variant of v3 search — records co-expansion pairs for TWPP.
/// The trace is observational only: search behavior is identical to the untraced version.
pub async fn disk_graph_search_pipe_v3_traced(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    trace: &mut TraceRecorder,
) -> Vec<ScoredId> {
    disk_graph_search_pipe_v3_inner(
        query, entry_set, k, ef, prefetch_window, stall_limit, drain_budget,
        pool, io, bank, adj_index, perf, level, Some(trace), 0, 1.0, 0,
    ).await
}

/// Page-aware scheduling variant of v3 search.
///
/// Among the top `page_sched_b` candidates in the heap, prefers the one whose page
/// is already resident in cache. This reduces physical reads by expanding cache-hot
/// candidates first, at the cost of a bounded deviation from pure distance order.
///
/// `page_sched_b = 0` disables scheduling (same as `disk_graph_search_pipe_v3`).
pub async fn disk_graph_search_pipe_v3_pagesched(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    page_sched_b: usize,
) -> Vec<ScoredId> {
    disk_graph_search_pipe_v3_inner(
        query, entry_set, k, ef, prefetch_window, stall_limit, drain_budget,
        pool, io, bank, adj_index, perf, level, None, page_sched_b, 1.0, 0,
    ).await
}

/// SAQ-gated variant of v3 search.
///
/// After SAQ-scoring all unvisited neighbors per expansion, keeps only the
/// top-T (by SAQ distance) before pushing to the candidate heap.
/// T = max(gate_min, ceil(degree × gate_ratio)).
///
/// Gated-out neighbors are NOT marked visited — they remain reachable via
/// alternate paths in later expansions.
///
/// `gate_ratio = 1.0` is equivalent to the ungated baseline.
pub async fn disk_graph_search_pipe_v3_gated(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    gate_ratio: f32,
    gate_min: usize,
) -> Vec<ScoredId> {
    disk_graph_search_pipe_v3_inner(
        query, entry_set, k, ef, prefetch_window, stall_limit, drain_budget,
        pool, io, bank, adj_index, perf, level, None, 0,
        gate_ratio, gate_min,
    ).await
}

async fn disk_graph_search_pipe_v3_inner(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    mut trace: Option<&mut TraceRecorder>,
    page_sched_b: usize,
    gate_ratio: f32,
    gate_min: usize,
) -> Vec<ScoredId> {
    let gating_enabled = gate_ratio < 1.0;
    let timing = level >= PerfLevel::EnableTime;

    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    let num_vectors = bank.num_vectors();
    assert_eq!(
        adj_index.len(),
        num_vectors,
        "adj_index len mismatch: {} vs num_vectors {}",
        adj_index.len(),
        num_vectors
    );

    let mut visited = vec![false; num_vectors];
    let mut entered_at = vec![u32::MAX; num_vectors];
    // parent_of[vid] = the VID whose expansion pushed this vid into the heap.
    // Used by trace to record co-expansion pairs.
    let mut parent_of = if trace.is_some() {
        vec![u32::MAX; num_vectors]
    } else {
        Vec::new() // zero alloc when not tracing
    };

    let cache_before = pool.stats();

    // Seed from entry set (DRAM only, no IO)
    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, vid);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
            entered_at[vid] = 0;
        }
    }

    let mut lookahead = [ScoredId::default(); 8];

    let mut consecutive_stalls: u64 = 0;
    let mut prev_furthest: f32 = f32::MAX;
    let mut drain_remaining: u32 = 0;

    loop {
        // Page-aware scheduling: among top-B candidates, prefer one whose page is resident.
        let (candidate, was_preferred) = if page_sched_b > 1 {
            match candidates.pop_preferred(page_sched_b, |vid| {
                let idx = vid as usize;
                if idx < num_vectors {
                    pool.is_resident(adj_index[idx].page_id)
                } else {
                    false
                }
            }) {
                Some(pair) => pair,
                None => break,
            }
        } else {
            match candidates.pop() {
                Some(c) => (c, false),
                None => break,
            }
        };

        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        let vid = candidate.id.0 as usize;
        if vid >= num_vectors {
            continue;
        }
        let entry = adj_index[vid];
        let page_id = entry.page_id;

        perf.expansions += 1;
        perf.blocks_read += 1;
        if was_preferred {
            perf.page_sched_hits += 1;
        }
        let expansion_num = perf.expansions;

        // Trace: record expansion + co-expansion with parent
        if let Some(ref mut tr) = trace {
            tr.record_expansion(vid as u32);
            let parent = parent_of[vid];
            if parent != u32::MAX {
                tr.record_co_expand(parent, vid as u32);
            }
        }

        // Prefetch upcoming pages by looking at upcoming candidate VIDs.
        if prefetch_window > 0 {
            let w = prefetch_window.min(8);
            let count = candidates.peek_nearest(&mut lookahead[..w]);
            for i in 0..count {
                let cand_vid = lookahead[i].id.0 as usize;
                if cand_vid >= num_vectors {
                    continue;
                }
                let pid = adj_index[cand_vid].page_id;
                if !pool.is_resident(pid) {
                    pool.prefetch_hint(pid);
                    perf.prefetch_issued += 1;
                }
            }
        }

        let inflight = (io.adj_capacity() - io.available_adj_permits()) as u64;
        perf.inflight_sum += inflight;
        perf.inflight_samples += 1;
        if inflight > perf.inflight_max {
            perf.inflight_max = inflight;
        }

        if let Some(g) = io.global_inflight() {
            perf.global_inflight_sum += g as u64;
            perf.global_inflight_samples += 1;
            if g as u64 > perf.global_inflight_max {
                perf.global_inflight_max = g as u64;
            }
        }

        // Load the page (key = page_id).
        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(page_id, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                perf.wasted_expansions += 1;
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        let compute_start = if timing { Some(Instant::now()) } else { None };

        let page = guard.data();
        debug_assert!(
            (entry.offset as usize) + 2 + (entry.degree as usize) * 4 <= page.len(),
            "packed record out of bounds: page_id={} offset={} degree={}",
            entry.page_id,
            entry.offset,
            entry.degree
        );

        let mut added_this_expansion = 0u32;

        if gating_enabled {
            // SAQ-gated path: score all unvisited neighbors, keep top-T.
            // Critical: do NOT mark visited until after gating decision.
            let mut scratch: Vec<(f32, u32)> = Vec::new();

            let dist_start = if timing { Some(Instant::now()) } else { None };
            for i in 0..(entry.degree as usize) {
                let nbr_raw = page_record_vid(page, &entry, i);
                let nbr_idx = nbr_raw as usize;
                if nbr_idx >= num_vectors || visited[nbr_idx] {
                    continue;
                }
                let d = bank.distance(query, nbr_idx);
                scratch.push((d, nbr_raw));
            }
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }

            let scored_count = scratch.len();
            perf.distance_computes += scored_count as u64;
            perf.pq_candidates_scored += scored_count as u64;

            // Gate to top-T by SAQ distance (ascending = best first)
            let t = (scored_count as f32 * gate_ratio).ceil() as usize;
            let t = t.max(gate_min).min(scored_count);

            if t < scored_count {
                scratch.select_nth_unstable_by(t, |a, b| a.0.total_cmp(&b.0));
                perf.pq_candidates_filtered += (scored_count - t) as u64;
            }
            perf.pq_candidates_passed += t.min(scored_count) as u64;

            // Push only the top-T neighbors
            for &(d, nbr_raw) in scratch.iter().take(t) {
                let nbr_idx = nbr_raw as usize;
                visited[nbr_idx] = true;

                let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId {
                        distance: d,
                        id: VectorId(nbr_raw),
                    };
                    candidates.push(scored);
                    nearest.push(scored);
                    added_this_expansion += 1;
                    if entered_at[nbr_idx] == u32::MAX {
                        entered_at[nbr_idx] = expansion_num as u32;
                    }
                    if !parent_of.is_empty() && parent_of[nbr_idx] == u32::MAX {
                        parent_of[nbr_idx] = vid as u32;
                    }
                }
            }
        } else {
            // Standard path: no gating, all neighbors scored and pushed.
            for i in 0..(entry.degree as usize) {
                let nbr_raw = page_record_vid(page, &entry, i);
                let nbr_idx = nbr_raw as usize;
                if nbr_idx >= num_vectors || visited[nbr_idx] {
                    continue;
                }
                visited[nbr_idx] = true;

                let dist_start = if timing { Some(Instant::now()) } else { None };
                let d = bank.distance(query, nbr_idx);
                if let Some(s) = dist_start {
                    perf.dist_ns += s.elapsed().as_nanos() as u64;
                }
                perf.distance_computes += 1;

                let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId {
                        distance: d,
                        id: VectorId(nbr_raw),
                    };
                    candidates.push(scored);
                    nearest.push(scored);
                    added_this_expansion += 1;
                    if entered_at[nbr_idx] == u32::MAX {
                        entered_at[nbr_idx] = expansion_num as u32;
                    }
                    if !parent_of.is_empty() && parent_of[nbr_idx] == u32::MAX {
                        parent_of[nbr_idx] = vid as u32;
                    }
                }
            }
        }

        if added_this_expansion > 0 {
            perf.useful_expansions += 1;
        } else {
            perf.wasted_expansions += 1;
        }

        if let Some(start) = compute_start {
            perf.compute_ns += start.elapsed().as_nanos() as u64;
        }

        drop(guard);

        // Adaptive stopping logic unchanged.
        if stall_limit > 0 && nearest.len() >= ef {
            let cur = nearest.furthest().unwrap().distance;
            if cur < prev_furthest {
                consecutive_stalls = 0;
                prev_furthest = cur;
                drain_remaining = 0;
            } else {
                consecutive_stalls += 1;
            }

            if drain_remaining > 0 {
                drain_remaining -= 1;
                if drain_remaining == 0 {
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            } else if consecutive_stalls >= stall_limit as u64 {
                drain_remaining = drain_budget;
                if drain_budget == 0 {
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            }
        }
    }

    if !perf.stopped_early {
        perf.consecutive_stalls_at_end = consecutive_stalls;
    }

    let cache_after = pool.stats();
    perf.blocks_hit = cache_after.hits - cache_before.hits;
    perf.blocks_miss = (cache_after.misses - cache_before.misses)
        + (cache_after.bypasses - cache_before.bypasses);
    perf.phys_reads = cache_after.phys_reads - cache_before.phys_reads;
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;
    perf.prefetch_consumed = cache_after.prefetch_hits - cache_before.prefetch_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);

    if !results.is_empty() {
        let best_vid = results[0].id.0 as usize;
        perf.best_result_at_expansion = if best_vid < num_vectors {
            entered_at[best_vid] as u64
        } else {
            0
        };

        let mut first_topk = u64::MAX;
        for r in &results {
            let rvid = r.id.0 as usize;
            if rvid < num_vectors && (entered_at[rvid] as u64) < first_topk {
                first_topk = entered_at[rvid] as u64;
            }
        }
        perf.first_topk_at_expansion = if first_topk == u64::MAX { 0 } else { first_topk };
    }

    results
}

/// Async beam search with PQ-based neighbor gating.
///
/// Extension of `disk_graph_search_pipe` that uses inline PQ codes from v2
/// adjacency blocks to pre-filter neighbors before exact distance computation.
///
/// - `pq_dt`: per-query PQ distance lookup table. If None, standard path (no gating).
/// - `gate_ratio`: fraction of neighbors to keep after PQ scoring (0.0 to 1.0).
/// - `gate_min`: minimum neighbors to keep per expansion. Typically 4.
pub async fn disk_graph_search_pq(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    pq_dt: Option<&PqDistanceTable>,
    gate_ratio: f32,
    gate_min: usize,
) -> Vec<ScoredId> {
    let timing = level >= PerfLevel::EnableTime;
    let gating_enabled = pq_dt.is_some() && gate_ratio < 1.0;

    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    let num_vectors = bank.num_vectors();
    let mut visited = vec![false; num_vectors];
    let mut entered_at = vec![u32::MAX; num_vectors];

    let cache_before = pool.stats();

    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let dist_start = if timing { Some(Instant::now()) } else { None };
            let d = bank.distance(query, vid);
            if let Some(s) = dist_start {
                perf.dist_ns += s.elapsed().as_nanos() as u64;
            }
            perf.distance_computes += 1;
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
            entered_at[vid] = 0;
        }
    }

    let mut lookahead = [ScoredId::default(); 8];
    let mut consecutive_stalls: u64 = 0;
    let mut prev_furthest: f32 = f32::MAX;
    let mut drain_remaining: u32 = 0;
    // (pq_score, neighbor_vid, block_idx). `pq_score`: smaller is better.
    let mut pq_scratch: Vec<(f32, u32, usize)> = Vec::new();

    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        perf.expansions += 1;
        perf.blocks_read += 1;
        let expansion_num = perf.expansions;

        if prefetch_window > 0 {
            let w = prefetch_window.min(8);
            let count = candidates.peek_nearest(&mut lookahead[..w]);
            for i in 0..count {
                let vid = lookahead[i].id.0;
                if !pool.is_resident(vid) {
                    pool.prefetch_hint(vid);
                    perf.prefetch_issued += 1;
                }
            }
        }

        let inflight = (io.adj_capacity() - io.available_adj_permits()) as u64;
        perf.inflight_sum += inflight;
        perf.inflight_samples += 1;
        if inflight > perf.inflight_max {
            perf.inflight_max = inflight;
        }
        if let Some(g) = io.global_inflight() {
            perf.global_inflight_sum += g as u64;
            perf.global_inflight_samples += 1;
            if g as u64 > perf.global_inflight_max {
                perf.global_inflight_max = g as u64;
            }
        }

        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                perf.wasted_expansions += 1;
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        let compute_start = if timing { Some(Instant::now()) } else { None };
        let view = decode_adj_block_view(guard.data());
        let mut added_this_expansion = 0u32;

        if gating_enabled && view.has_codes() {
            let dt = pq_dt.unwrap();

            // PQ-score all unvisited neighbors
            pq_scratch.clear();
            pq_scratch.reserve(view.degree);
            for i in 0..view.degree {
                let nbr_raw = view.neighbor_vid(i);
                let nbr_idx = nbr_raw as usize;
                if nbr_idx >= num_vectors || visited[nbr_idx] {
                    continue;
                }
                let raw = dt.approximate_distance(view.neighbor_code(i));
                // If the distance table was built in inner-product mode, `raw` is a dot product
                // where larger is better. Flip the sign so "smaller is better" ordering holds.
                let pq_score = if dt.use_inner_product() { -raw } else { raw };
                pq_scratch.push((pq_score, nbr_raw, i));
                perf.pq_candidates_scored += 1;
            }

            // Select Top-T (deterministic tie-break by block index)
            let t = (pq_scratch.len() as f32 * gate_ratio).ceil() as usize;
            let t = t.max(gate_min).min(pq_scratch.len());

            if t < pq_scratch.len() {
                // Deterministic tie-break: (pq_score, vid, block_idx).
                pq_scratch.sort_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                });
                perf.pq_candidates_filtered += (pq_scratch.len() - t) as u64;
            }
            perf.pq_candidates_passed += t as u64;

            // Exact distance for Top-T only
            for &(_, nbr_raw, _block_idx) in pq_scratch.iter().take(t) {
                let nbr_idx = nbr_raw as usize;
                if visited[nbr_idx] {
                    continue;
                }
                visited[nbr_idx] = true;

                let dist_start = if timing { Some(Instant::now()) } else { None };
                let d = bank.distance(query, nbr_idx);
                if let Some(s) = dist_start {
                    perf.dist_ns += s.elapsed().as_nanos() as u64;
                }
                perf.distance_computes += 1;

                let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId { distance: d, id: VectorId(nbr_raw) };
                    candidates.push(scored);
                    nearest.push(scored);
                    added_this_expansion += 1;
                    if entered_at[nbr_idx] == u32::MAX {
                        entered_at[nbr_idx] = expansion_num as u32;
                    }
                }
            }
        } else {
            // Standard path (no gating / v1 blocks)
            for i in 0..view.degree {
                let nbr_raw = view.neighbor_vid(i);
                let nbr_idx = nbr_raw as usize;
                if nbr_idx >= num_vectors || visited[nbr_idx] {
                    continue;
                }
                visited[nbr_idx] = true;

                let dist_start = if timing { Some(Instant::now()) } else { None };
                let d = bank.distance(query, nbr_idx);
                if let Some(s) = dist_start {
                    perf.dist_ns += s.elapsed().as_nanos() as u64;
                }
                perf.distance_computes += 1;

                let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId { distance: d, id: VectorId(nbr_raw) };
                    candidates.push(scored);
                    nearest.push(scored);
                    added_this_expansion += 1;
                    if entered_at[nbr_idx] == u32::MAX {
                        entered_at[nbr_idx] = expansion_num as u32;
                    }
                }
            }
        }

        if added_this_expansion > 0 {
            perf.useful_expansions += 1;
        } else {
            perf.wasted_expansions += 1;
        }

        if let Some(start) = compute_start {
            perf.compute_ns += start.elapsed().as_nanos() as u64;
        }

        drop(guard);

        if stall_limit > 0 && nearest.len() >= ef {
            let cur = nearest.furthest().unwrap().distance;
            if cur < prev_furthest {
                consecutive_stalls = 0;
                prev_furthest = cur;
                drain_remaining = 0;
            } else {
                consecutive_stalls += 1;
            }

            if drain_remaining > 0 {
                drain_remaining -= 1;
                if drain_remaining == 0 {
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            } else if consecutive_stalls >= stall_limit as u64 {
                drain_remaining = drain_budget;
                if drain_budget == 0 {
                    perf.stopped_early = true;
                    perf.consecutive_stalls_at_end = consecutive_stalls;
                    perf.expansions_at_stop = perf.expansions;
                    break;
                }
            }
        }
    }

    if !perf.stopped_early {
        perf.consecutive_stalls_at_end = consecutive_stalls;
    }

    let cache_after = pool.stats();
    perf.blocks_hit = cache_after.hits - cache_before.hits;
    perf.blocks_miss = (cache_after.misses - cache_before.misses)
        + (cache_after.bypasses - cache_before.bypasses);
    perf.phys_reads = cache_after.phys_reads - cache_before.phys_reads;
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;
    perf.prefetch_consumed = cache_after.prefetch_hits - cache_before.prefetch_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);

    if !results.is_empty() {
        let best_vid = results[0].id.0 as usize;
        perf.best_result_at_expansion = if best_vid < num_vectors {
            entered_at[best_vid] as u64
        } else {
            0
        };
        let mut first_topk = u64::MAX;
        for r in &results {
            let vid = r.id.0 as usize;
            if vid < num_vectors && (entered_at[vid] as u64) < first_topk {
                first_topk = entered_at[vid] as u64;
            }
        }
        perf.first_topk_at_expansion = if first_topk == u64::MAX { 0 } else { first_topk };
    }

    results
}

/// Two-stage v4 search: PQ proxy traversal (v3 pages) + parallel FP32 disk refine.
///
/// 1. Run `disk_graph_search_pipe_v3` with `cheap_bank` (PQ codes in DRAM) → collect top-R
/// 2. Read FP32 vectors from disk in parallel (monoio::spawn batches, bounded inflight)
/// 3. Re-score with exact cosine distance, return top-k
///
/// This is the fair-comparison architecture: PQ codes (~10MB) + adjacency in DRAM,
/// FP32 vectors on disk — same DRAM budget as DiskANN.
pub async fn disk_graph_search_pipe_v3_refine(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    vec_reader: &std::rc::Rc<crate::io::VectorReader>,
    refine_inflight: usize,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    use std::rc::Rc;

    // Stage 1: PQ graph traversal via v3 page-packed adjacency
    let candidates = disk_graph_search_pipe_v3(
        query, entry_set, refine_r, ef, prefetch_window,
        stall_limit, drain_budget, pool, io, cheap_bank, adj_index, perf, level,
    )
    .await;

    // Stage 2: parallel FP32 vector reads + exact cosine rescore.
    // Each spawned task returns its result; no shared mutable state.
    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    // Precompute query norm (constant across all R candidates)
    let norm_a: f32 = query.iter().map(|&x| x * x).sum();

    // Copy query to owned Vec so spawned tasks can capture it ('static).
    let query_rc = Rc::new(query.to_vec());

    // Parallel batch reads bounded by refine_inflight.
    // No `Vec<f32>` per-candidate allocation: read_cosine_distance computes directly on the IO buffer.
    let mut refined = Vec::with_capacity(candidates.len());

    for batch in candidates.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(vec_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok(dist) => ScoredId { distance: dist, id: c.id },
                    Err(_) => c,
                }
            });
            handles.push(handle);
        }
        // Await all tasks in this batch — io_uring processes them concurrently
        for h in handles {
            refined.push(h.await);
        }
    }

    perf.refine_count = refined.len() as u64;

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    // Sort by refined distance, return top-k
    refined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    refined.truncate(k);
    refined
}

/// Two-stage v4 search with int8-quantized refine vectors on disk.
///
/// Same as `disk_graph_search_pipe_v3_refine` but reads int8 vectors (768 bytes/vec
/// instead of 3072 bytes/vec for dim=768), giving ~3× fewer IO bytes per refine read.
/// Int8 encoding: round(f32 * 127), cosine distance computed directly on int8 codes.
/// Tracks actual IO bytes submitted (via read_at return) for honest accounting.
pub async fn disk_graph_search_pipe_v3_refine_int8(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    vec_reader: &std::rc::Rc<crate::io::Int8VectorReader>,
    refine_inflight: usize,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    use std::rc::Rc;

    // Stage 1: proxy graph traversal
    let candidates = disk_graph_search_pipe_v3(
        query, entry_set, refine_r, ef, prefetch_window,
        stall_limit, drain_budget, pool, io, cheap_bank, adj_index, perf, level,
    )
    .await;

    // Stage 2: parallel int8 vector reads + cosine rescore
    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    let norm_a: f32 = query.iter().map(|&x| x * x).sum();
    let query_rc = Rc::new(query.to_vec());
    let mut refined = Vec::with_capacity(candidates.len());
    let mut total_refine_bytes = 0u64;

    for batch in candidates.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(vec_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok((dist, bytes_read)) => (ScoredId { distance: dist, id: c.id }, bytes_read),
                    Err(_) => (c, 0),
                }
            });
            handles.push(handle);
        }
        for h in handles {
            let (scored, bytes) = h.await;
            refined.push(scored);
            total_refine_bytes += bytes as u64;
        }
    }

    perf.refine_count = refined.len() as u64;
    perf.refine_bytes = total_refine_bytes;

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    refined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    refined.truncate(k);
    refined
}

/// Two-stage search with FP16 refine vectors on disk.
///
/// Same structure as `disk_graph_search_pipe_v3_refine` but reads FP16 vectors
/// (dim×2 bytes/vec = 1536 bytes for dim=768) giving 2× fewer IO bytes per refine.
/// FP16 has ~3 decimal digits of precision — much better than int8 (~2.1).
/// Tracks actual IO bytes submitted for honest accounting.
pub async fn disk_graph_search_pipe_v3_refine_fp16(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    fp16_reader: &std::rc::Rc<crate::io::Fp16VectorReader>,
    refine_inflight: usize,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    use std::rc::Rc;

    // Stage 1: proxy graph traversal
    let candidates = disk_graph_search_pipe_v3(
        query, entry_set, refine_r, ef, prefetch_window,
        stall_limit, drain_budget, pool, io, cheap_bank, adj_index, perf, level,
    )
    .await;

    // Stage 2: parallel FP16 vector reads + cosine rescore
    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    let norm_a: f32 = query.iter().map(|&x| x * x).sum();
    let query_rc = Rc::new(query.to_vec());
    let mut refined = Vec::with_capacity(candidates.len());
    let mut total_refine_bytes = 0u64;

    for batch in candidates.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(fp16_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok((dist, bytes_read)) => (ScoredId { distance: dist, id: c.id }, bytes_read),
                    Err(_) => (c, 0),
                }
            });
            handles.push(handle);
        }
        for h in handles {
            let (scored, bytes) = h.await;
            refined.push(scored);
            total_refine_bytes += bytes as u64;
        }
    }

    perf.refine_count = refined.len() as u64;
    perf.refine_bytes = total_refine_bytes;

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    refined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    refined.truncate(k);
    refined
}

/// SAQ-gated FP16 refine: gated graph traversal → FP16 disk refine.
///
/// Same as `disk_graph_search_pipe_v3_refine_fp16` but with SAQ gating in stage 1.
/// `gate_ratio` and `gate_min` control the gating behavior (see `disk_graph_search_pipe_v3_gated`).
pub async fn disk_graph_search_pipe_v3_refine_fp16_gated(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    fp16_reader: &std::rc::Rc<crate::io::Fp16VectorReader>,
    refine_inflight: usize,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
    gate_ratio: f32,
    gate_min: usize,
) -> Vec<ScoredId> {
    use std::rc::Rc;

    // Stage 1: SAQ-gated graph traversal
    let candidates = disk_graph_search_pipe_v3_gated(
        query, entry_set, refine_r, ef, prefetch_window,
        stall_limit, drain_budget, pool, io, cheap_bank, adj_index, perf, level,
        gate_ratio, gate_min,
    )
    .await;

    // Stage 2: parallel FP16 vector reads + cosine rescore
    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    let norm_a: f32 = query.iter().map(|&x| x * x).sum();
    let query_rc = Rc::new(query.to_vec());
    let mut refined = Vec::with_capacity(candidates.len());
    let mut total_refine_bytes = 0u64;

    for batch in candidates.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(fp16_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok((dist, bytes_read)) => (ScoredId { distance: dist, id: c.id }, bytes_read),
                    Err(_) => (c, 0),
                }
            });
            handles.push(handle);
        }
        for h in handles {
            let (scored, bytes) = h.await;
            refined.push(scored);
            total_refine_bytes += bytes as u64;
        }
    }

    perf.refine_count = refined.len() as u64;
    perf.refine_bytes = total_refine_bytes;

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    refined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    refined.truncate(k);
    refined
}

/// Three-stage pipeline: SAQ proxy → int8 filter → FP32 exact refine.
///
/// Stage 1: SAQ graph traversal (DRAM) → top-R candidates
/// Stage 2: Int8 refine (disk, 1KB/vec) → top-`int8_topk` candidates (cheap filter)
/// Stage 3: FP32 refine (disk, 3KB/vec) → top-k exact re-ranking
///
/// This recovers FP32 recall while reading most refine vectors as int8 (3× cheaper).
/// Expected IO at R=160, int8_topk=40: 160KB int8 + 120KB FP32 = 280KB vs 480KB FP32-only.
pub async fn disk_graph_search_pipe_v3_refine_3stage(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    refine_r: usize,
    int8_topk: usize,
    prefetch_window: usize,
    stall_limit: u32,
    drain_budget: u32,
    pool: &AdjacencyPool,
    io: &IoDriver,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    int8_reader: &std::rc::Rc<crate::io::Int8VectorReader>,
    fp32_reader: &std::rc::Rc<crate::io::VectorReader>,
    refine_inflight: usize,
    perf: &mut SearchPerfContext,
    level: PerfLevel,
) -> Vec<ScoredId> {
    use std::rc::Rc;

    // Stage 1: SAQ proxy graph traversal
    let candidates = disk_graph_search_pipe_v3(
        query, entry_set, refine_r, ef, prefetch_window,
        stall_limit, drain_budget, pool, io, cheap_bank, adj_index, perf, level,
    )
    .await;

    let timing = level >= PerfLevel::EnableTime;
    let refine_start = if timing { Some(Instant::now()) } else { None };

    let norm_a: f32 = query.iter().map(|&x| x * x).sum();
    let query_rc = Rc::new(query.to_vec());

    // Stage 2: Int8 filter — read all R candidates as int8, keep top int8_topk
    let mut int8_scored = Vec::with_capacity(candidates.len());
    let mut total_refine_bytes = 0u64;

    for batch in candidates.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(int8_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok((dist, bytes)) => (ScoredId { distance: dist, id: c.id }, bytes),
                    Err(_) => (c, 0),
                }
            });
            handles.push(handle);
        }
        for h in handles {
            let (scored, bytes) = h.await;
            int8_scored.push(scored);
            total_refine_bytes += bytes as u64;
        }
    }

    // Sort by int8 distance, keep top int8_topk for FP32 re-ranking
    int8_scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    int8_scored.truncate(int8_topk);

    // Stage 3: FP32 exact refine on the int8-filtered candidates
    let mut fp32_refined = Vec::with_capacity(int8_scored.len());
    let fp32_bytes_per_vec = fp32_reader.dim() * 4;

    for batch in int8_scored.chunks(refine_inflight) {
        let mut handles = Vec::with_capacity(batch.len());
        for &c in batch {
            let vid = c.id.0;
            let vr = Rc::clone(fp32_reader);
            let q = Rc::clone(&query_rc);
            let handle = monoio::spawn(async move {
                match vr.read_cosine_distance(vid, &q, norm_a).await {
                    Ok(dist) => ScoredId { distance: dist, id: c.id },
                    Err(_) => c,
                }
            });
            handles.push(handle);
        }
        for h in handles {
            fp32_refined.push(h.await);
        }
    }

    perf.refine_count = (candidates.len() + int8_topk) as u64;
    // Honest bytes: int8 measured + FP32 estimated from vec dimensions
    perf.refine_bytes = total_refine_bytes
        + (int8_topk as u64 * ((fp32_bytes_per_vec as u64 + 511) & !511));

    if let Some(start) = refine_start {
        perf.refine_ns += start.elapsed().as_nanos() as u64;
    }

    fp32_refined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    fp32_refined.truncate(k);
    fp32_refined
}
