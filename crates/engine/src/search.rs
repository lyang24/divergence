//! Async beam search over on-disk adjacency graph.
//!
//! Vectors stay in DRAM. Adjacency blocks are read from NVMe via io_uring,
//! with per-core caching via AdjacencyPool. Each `.await` on cache miss is
//! a coroutine suspension point — monoio services other queries while
//! the NVMe read is in flight.

use std::time::Instant;

use divergence_core::distance::VectorBank;
use divergence_core::VectorId;
use divergence_index::{CandidateHeap, FixedCapacityHeap, ScoredId};
use divergence_storage::decode_adj_block;

use crate::cache::AdjacencyPool;
use crate::io::IoDriver;
use crate::perf::{PerfLevel, SearchPerfContext};

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

        // ← ONLY yield point: get adjacency block (cache hit = sync, miss = async IO)
        let io_start = if timing { Some(Instant::now()) } else { None };
        let guard = match pool.get_or_load(candidate.id.0, io).await {
            Ok(g) => g,
            Err(_) => {
                if let Some(start) = io_start {
                    perf.io_wait_ns += start.elapsed().as_nanos() as u64;
                }
                continue;
            }
        };
        if let Some(start) = io_start {
            perf.io_wait_ns += start.elapsed().as_nanos() as u64;
        }

        // Compute phase: decode + distance + heap (all sync, DRAM only)
        let compute_start = if timing { Some(Instant::now()) } else { None };

        let neighbors = decode_adj_block(guard.data());

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
            }
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
    perf.singleflight_waits = cache_after.dedup_hits - cache_before.dedup_hits;

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);
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
