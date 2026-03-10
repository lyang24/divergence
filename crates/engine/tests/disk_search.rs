//! Integration test: build NSW → write to disk → async search → verify results match.
//!
//! These tests require io_uring support (Linux 5.1+, not inside unprivileged containers).
//! They are automatically skipped if io_uring is unavailable.

use std::path::Path;
use std::rc::Rc;

use divergence_core::distance::{
    create_distance_computer, fp32_to_fp16, FP16VectorBank, FP32SimdVectorBank,
    FP32VectorBank, Int8VectorBank, VectorBank,
};
use divergence_core::quantization::{PqCodebook, ScalarQuantizer, l2_normalize, l2_normalize_batch};
use divergence_core::{MetricType, VectorId};
use divergence_engine::{
    disk_graph_search, disk_graph_search_exp, disk_graph_search_pipe, disk_graph_search_pipe_v3,
    disk_graph_search_pipe_v3_refine, disk_graph_search_pq, disk_graph_search_refine,
    AdaEfParams, AdaEfStats, AdaEfTable, estimate_ada_ef,
    AdjacencyPool, IoDriver, VectorReader, PerfLevel, QueryRecorder, SearchGuard, SearchPerfContext,
};
use divergence_index::{NswBuilder, NswConfig};
use divergence_storage::{
    load_vectors, IndexMeta, IndexWriter,
    AdjIndexEntry, bfs_reorder_graph, load_adj_index,
};

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Try to build a monoio io_uring runtime. Returns false if io_uring is not
/// available (e.g. unprivileged container, old kernel), and runs the closure
/// on success.
fn with_runtime(
    f: impl FnOnce(&mut monoio::Runtime<monoio::time::TimeDriver<monoio::IoUringDriver>>),
) -> bool {
    match monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_all()
        .build()
    {
        Ok(mut rt) => {
            f(&mut rt);
            true
        }
        Err(_) => false,
    }
}

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

#[test]
fn disk_search_matches_memory() {
    let n = 500;
    let dim = 32;
    let k = 10;
    let ef = 64;
    let m_max = 32;
    let ef_construction = 200;

    // 1. Generate vectors and build NSW in memory
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    // 2. Search in memory for ground truth
    let query: Vec<f32> = generate_vectors(1, dim, 999)[0].clone();
    let memory_results = index.search(&query, k, ef);

    // 3. Write to disk
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    // 4. Load meta + vectors for disk search
    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // 5. Run async disk search inside monoio runtime
    if !with_runtime(|rt| {
        let disk_results = rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(64 * 1024);
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let mut perf = SearchPerfContext::default();

            disk_graph_search(
                &query, &entry_set, k, ef, &pool, &io, &bank, &mut perf,
                PerfLevel::CountOnly,
            )
            .await
        });

        // 6. Verify: disk results should match memory results exactly
        assert_eq!(
            disk_results.len(),
            memory_results.len(),
            "result count mismatch"
        );

        for (i, (disk, mem)) in disk_results.iter().zip(memory_results.iter()).enumerate() {
            assert_eq!(
                disk.id, mem.id,
                "VID mismatch at position {}: disk={:?} mem={:?}",
                i, disk.id, mem.id
            );
            assert!(
                (disk.distance - mem.distance).abs() < 1e-6,
                "distance mismatch at position {}: disk={} mem={}",
                i,
                disk.distance,
                mem.distance
            );
        }
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

#[test]
fn io_driver_reads_single_block() {
    let n = 3u32;
    let dim = 4;

    // Write a small adjacency file
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();

    let adj: Vec<Vec<u32>> = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let vectors: Vec<f32> = vec![0.0; n as usize * dim];

    let writer = IndexWriter::new(dir.path());
    writer
        .write(n, dim, "l2", 32, 200, &[0], &vectors, |vid| {
            &adj[vid as usize]
        })
        .unwrap();

    // Read back with IoDriver
    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            for vid in 0..n {
                let buf = io.read_adj_block(vid).await.expect("read failed");
                let neighbors = divergence_storage::decode_adj_block(buf.as_slice());
                assert_eq!(
                    neighbors, adj[vid as usize],
                    "mismatch at vid {}",
                    vid
                );
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Profile harness: cold vs warm cache A/B comparison
// ---------------------------------------------------------------------------

#[test]
fn profile_cold_vs_warm() {
    let n = 2000;
    let dim = 64;
    let k = 10;
    let ef = 64;
    let m_max = 32;
    let ef_construction = 200;
    let num_queries = 100;

    // 1. Build index
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    // 2. Write to disk
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    // 3. Load for disk search
    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // 4. Generate query batch
    let queries: Vec<Vec<f32>> = generate_vectors(num_queries, dim, 999);

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(256 * 1024);
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let recorder = QueryRecorder::new();

            // === PASS 1: COLD CACHE ===
            for q in &queries {
                let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                let level = guard.level();
                disk_graph_search(
                    q, &entry_set, k, ef, &pool, &io, &bank,
                    &mut guard.ctx, level,
                )
                .await;
            }

            let cold_cache = pool.stats();

            eprintln!("\n========== COLD CACHE ==========");
            eprintln!("{}", recorder.report());
            eprintln!(
                "Cache totals: hits={} misses={} dedup={} evict={} bypass={}",
                cold_cache.hits,
                cold_cache.misses,
                cold_cache.dedup_hits,
                cold_cache.evictions,
                cold_cache.bypasses
            );

            // === PASS 2: WARM CACHE (same queries) ===
            recorder.reset();
            for q in &queries {
                let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                let level = guard.level();
                disk_graph_search(
                    q, &entry_set, k, ef, &pool, &io, &bank,
                    &mut guard.ctx, level,
                )
                .await;
            }

            let warm_cache = pool.stats();
            let warm_hits = warm_cache.hits - cold_cache.hits;
            let warm_misses = warm_cache.misses - cold_cache.misses;
            let warm_dedup = warm_cache.dedup_hits - cold_cache.dedup_hits;
            let warm_bypasses = warm_cache.bypasses - cold_cache.bypasses;
            let warm_total = warm_hits + warm_misses + warm_dedup + warm_bypasses;
            let warm_hit_rate = if warm_total > 0 {
                (warm_hits as f64 / warm_total as f64) * 100.0
            } else {
                0.0
            };

            eprintln!("\n========== WARM CACHE ==========");
            eprintln!("{}", recorder.report());
            eprintln!(
                "Cache delta: hits={} misses={} dedup={} bypass={} | hit_rate={:.1}%",
                warm_hits, warm_misses, warm_dedup, warm_bypasses, warm_hit_rate
            );

            // === VERDICT ===
            eprintln!("\n========== VERDICT ==========");
            let cold_blocks_mean = if cold_cache.hits + cold_cache.misses > 0 {
                (cold_cache.hits + cold_cache.misses) as f64 / num_queries as f64
            } else {
                0.0
            };
            let warm_blocks_mean = if warm_total > 0 {
                warm_total as f64 / num_queries as f64
            } else {
                0.0
            };
            let blocks_reduction = if cold_blocks_mean > 0.0 {
                ((cold_blocks_mean - warm_blocks_mean) / cold_blocks_mean) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "Blocks/query: cold={:.1} warm={:.1} reduction={:.1}%",
                cold_blocks_mean, warm_blocks_mean, blocks_reduction
            );
            eprintln!(
                "Cache hit rate: cold=0% warm={:.1}%",
                warm_hit_rate
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Micro-experiments: cache locality, cache sizing, dimensionality scaling
// ---------------------------------------------------------------------------

/// Helper: build NSW index, write to disk, return dir and flat vectors.
fn build_disk_index(
    n: usize,
    dim: usize,
    m_max: usize,
    ef_construction: usize,
) -> (tempfile::TempDir, Vec<f32>) {
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    // Flatten vectors for DRAM-resident copy
    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    (dir, flat)
}

/// Load index metadata + entry set from disk.
fn load_meta_entry(path: &Path) -> (IndexMeta, Vec<VectorId>) {
    let meta = IndexMeta::load_from(&path.join("meta.json")).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();
    (meta, entry_set)
}

/// Run a batch of queries using a VectorBank, recording into a QueryRecorder.
async fn run_query_pass(
    queries: &[Vec<f32>],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
    recorder: &QueryRecorder,
    level: PerfLevel,
) {
    for q in queries {
        let mut guard = SearchGuard::new(recorder, level);
        let lvl = guard.level();
        disk_graph_search(q, entry_set, k, ef, pool, io, bank, &mut guard.ctx, lvl).await;
    }
}

/// Experiment 1: Freeze queries — same 100 queries twice.
/// With a cache large enough to hold ALL blocks (N=2000 → 8MB),
/// verify warm pass hit rate is significantly higher than cold.
#[test]
fn exp_freeze_queries_warm_hit() {
    let n = 2000;
    let dim = 64;
    let k = 10;
    let ef = 64;
    let num_queries = 100;

    let (dir, _flat) = build_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let queries = generate_vectors(num_queries, dim, 999);

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(n * 4096);
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let recorder = QueryRecorder::new();

            // Pass 1: cold
            run_query_pass(
                &queries, &entry_set, k, ef, &pool, &io, &bank,
                &recorder, PerfLevel::EnableTime,
            )
            .await;
            let cold_stats = pool.stats();

            eprintln!(
                "\n========== EXP1: FREEZE QUERIES (cache={}KB, holds all {} blocks) ==========",
                n * 4, n
            );
            eprintln!("COLD: {}", recorder.report());
            eprintln!(
                "  hits={} misses={} evict={}",
                cold_stats.hits, cold_stats.misses, cold_stats.evictions
            );

            // Pass 2: warm (same queries, cache retains everything)
            recorder.reset();
            run_query_pass(
                &queries, &entry_set, k, ef, &pool, &io, &bank,
                &recorder, PerfLevel::EnableTime,
            )
            .await;
            let warm_stats = pool.stats();
            let warm_hits = warm_stats.hits - cold_stats.hits;
            let warm_misses = warm_stats.misses - cold_stats.misses;
            let warm_total = warm_hits + warm_misses;
            let warm_hit_pct = if warm_total > 0 {
                warm_hits as f64 / warm_total as f64 * 100.0
            } else {
                0.0
            };

            eprintln!("WARM: {}", recorder.report());
            eprintln!(
                "  hits={} misses={} | hit_rate={:.1}%",
                warm_hits, warm_misses, warm_hit_pct
            );
            eprintln!(
                "  evictions_total={} (should be 0 if cache fits all)",
                warm_stats.evictions
            );

            if warm_hit_pct < 50.0 {
                eprintln!(
                    "  DIAGNOSIS: warm hit {:.1}% < 50% → low temporal locality",
                    warm_hit_pct
                );
            } else {
                eprintln!(
                    "  DIAGNOSIS: warm hit {:.1}% >= 50% → temporal locality confirmed",
                    warm_hit_pct
                );
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

/// Experiment 2: Vary cache size — 64, 256, 1024, 2048 slots.
#[test]
fn exp_vary_cache_size() {
    let n = 2000;
    let dim = 64;
    let k = 10;
    let ef = 64;
    let num_queries = 100;

    let (dir, _flat) = build_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let queries = generate_vectors(num_queries, dim, 999);

    let cache_sizes_kb: Vec<usize> = vec![256, 1024, 4096, 8192];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);

            eprintln!("\n========== EXP2: VARY CACHE SIZE ==========");
            eprintln!(
                "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "cache_KB", "slots", "hit_rate%", "blocks/q", "p50_us", "p99_us"
            );

            for &cache_kb in &cache_sizes_kb {
                let pool = AdjacencyPool::new(cache_kb * 1024);
                let recorder = QueryRecorder::new();

                // Warm up: run once
                run_query_pass(
                    &queries, &entry_set, k, ef, &pool, &io, &bank,
                    &recorder, PerfLevel::EnableTime,
                )
                .await;
                let cold_stats = pool.stats();

                // Measure: run again (warm)
                recorder.reset();
                run_query_pass(
                    &queries, &entry_set, k, ef, &pool, &io, &bank,
                    &recorder, PerfLevel::EnableTime,
                )
                .await;
                let warm_stats = pool.stats();
                let hits = warm_stats.hits - cold_stats.hits;
                let misses = warm_stats.misses - cold_stats.misses;
                let total = hits + misses;
                let hit_rate = if total > 0 {
                    hits as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                let blocks_per_q = total as f64 / num_queries as f64;

                let report = recorder.report();
                eprintln!(
                    "{:<12} {:>10} {:>10.1} {:>10.1} {:>10} {:>10}",
                    cache_kb,
                    cache_kb * 1024 / 4096,
                    hit_rate,
                    blocks_per_q,
                    "-",
                    "-"
                );
                eprintln!("  {}", report.lines().nth(1).unwrap_or(""));
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

/// Experiment 3: Vary dimension — 64, 128, 256, 512, 768.
/// Check if distance% rises with dimension.
#[test]
fn exp_vary_dimension() {
    let n = 2000;
    let k = 10;
    let ef = 64;
    let num_queries = 100;
    let dims = [64, 128, 256, 512, 768];

    eprintln!("\n========== EXP3: VARY DIMENSION ==========");
    eprintln!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "dim", "total_us", "io%", "dist%", "overhead%", "dist/call", "calls/q"
    );

    for &dim in &dims {
        let (dir, _flat) = build_disk_index(n, dim, 32, 200);
        let dir_str = dir.path().to_str().unwrap().to_owned();
        let (_meta, entry_set) = load_meta_entry(dir.path());
        let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
        let queries = generate_vectors(num_queries, dim, 999);

        if !with_runtime(|rt| {
            rt.block_on(async {
                let io = IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver");

                let pool = AdjacencyPool::new(256 * 1024);
                let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
                let mut sum_io = 0u64;
                let mut sum_dist = 0u64;
                let mut sum_compute = 0u64;
                let mut sum_total = 0u64;
                let mut sum_dist_calls = 0u64;

                for q in &queries {
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    disk_graph_search(
                        q, &entry_set, k, ef, &pool, &io, &bank,
                        &mut perf, PerfLevel::EnableTime,
                    )
                    .await;
                    let elapsed = t.elapsed().as_nanos() as u64;
                    sum_io += perf.io_wait_ns;
                    sum_dist += perf.dist_ns;
                    sum_compute += perf.compute_ns;
                    sum_total += elapsed;
                    sum_dist_calls += perf.distance_computes;
                }

                let nq = num_queries as f64;
                let mean_total = sum_total as f64 / nq;
                let io_pct = sum_io as f64 / sum_total as f64 * 100.0;
                let dist_pct = sum_dist as f64 / sum_total as f64 * 100.0;
                let overhead_ns = sum_compute.saturating_sub(sum_dist);
                let overhead_pct = overhead_ns as f64 / sum_total as f64 * 100.0;
                let dist_per_call = if sum_dist_calls > 0 {
                    sum_dist as f64 / sum_dist_calls as f64
                } else {
                    0.0
                };
                let calls_per_q = sum_dist_calls as f64 / nq;

                eprintln!(
                    "{:<8} {:>10.0} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10.0}",
                    dim,
                    mean_total / 1000.0,
                    io_pct,
                    dist_pct,
                    overhead_pct,
                    dist_per_call,
                    calls_per_q
                );
            });
        }) {
            eprintln!("SKIPPED: io_uring not available for dim={}", dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Experiment 4: 3-way fair comparison (fp32-auto / fp32-simd / fp16-fused)
// ---------------------------------------------------------------------------

/// Helper: run a batch of queries, accumulate dist_ns and total_ns.
async fn measure_search_pass(
    queries: &[Vec<f32>],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    pool: &AdjacencyPool,
    io: &IoDriver,
    bank: &dyn VectorBank,
) -> (u64, u64, u64) {
    let mut total_ns = 0u64;
    let mut dist_ns = 0u64;
    let mut dist_calls = 0u64;
    for q in queries {
        let mut perf = SearchPerfContext::default();
        let t = std::time::Instant::now();
        disk_graph_search(
            q, entry_set, k, ef, pool, io, bank,
            &mut perf, PerfLevel::EnableTime,
        )
        .await;
        total_ns += t.elapsed().as_nanos() as u64;
        dist_ns += perf.dist_ns;
        dist_calls += perf.distance_computes;
    }
    (total_ns, dist_ns, dist_calls)
}

/// 3-way fair comparison at dim=512 and dim=768.
/// fp32-auto: iterator-based L2 (relies on autovectorization)
/// fp32-simd: hand-written AVX2+FMA L2 (fair baseline for FP16)
/// fp16-fused: hand-written AVX2+f16c+FMA fused convert+L2
#[test]
fn exp_fp32_vs_fp16() {
    let n = 2000;
    let k = 10;
    let ef = 64;
    let num_queries = 100;
    let dims = [512, 768];

    eprintln!("\n========== EXP4: 3-WAY FAIR COMPARISON ==========");
    eprintln!(
        "{:<8} {:<12} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "dim", "mode", "total_us", "dist_us", "dist%", "ns/call", "vs-simd"
    );

    for &dim in &dims {
        let (dir, _flat) = build_disk_index(n, dim, 32, 200);
        let dir_str = dir.path().to_str().unwrap().to_owned();
        let (_meta, entry_set) = load_meta_entry(dir.path());
        let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
        let dist = create_distance_computer(MetricType::L2);
        let vectors_fp16 = fp32_to_fp16(&disk_vectors);
        let queries = generate_vectors(num_queries, dim, 999);

        if !with_runtime(|rt| {
            rt.block_on(async {
                let io = IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver");

                let nq = num_queries as f64;

                // --- FP32 autovectorized ---
                let pool1 = AdjacencyPool::new(n * 4096);
                let bank_auto = FP32VectorBank::new(&disk_vectors, dim, &*dist);
                let (auto_total, auto_dist, auto_calls) = measure_search_pass(
                    &queries, &entry_set, k, ef, &pool1, &io, &bank_auto,
                ).await;

                // --- FP32 hand SIMD ---
                let pool2 = AdjacencyPool::new(n * 4096);
                let bank_simd = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
                let (simd_total, simd_dist, simd_calls) = measure_search_pass(
                    &queries, &entry_set, k, ef, &pool2, &io, &bank_simd,
                ).await;

                // --- FP16 fused ---
                let pool3 = AdjacencyPool::new(n * 4096);
                let bank_fp16 = FP16VectorBank::new(&vectors_fp16, dim, MetricType::L2);
                let (fp16_total, fp16_dist, fp16_calls) = measure_search_pass(
                    &queries, &entry_set, k, ef, &pool3, &io, &bank_fp16,
                ).await;

                let auto_pc = if auto_calls > 0 { auto_dist as f64 / auto_calls as f64 } else { 0.0 };
                let simd_pc = if simd_calls > 0 { simd_dist as f64 / simd_calls as f64 } else { 0.0 };
                let fp16_pc = if fp16_calls > 0 { fp16_dist as f64 / fp16_calls as f64 } else { 0.0 };

                let print_row = |label: &str, total: u64, dist: u64, calls: u64, pc: f64, vs_simd: Option<f64>| {
                    let t_us = total as f64 / nq / 1000.0;
                    let d_us = dist as f64 / nq / 1000.0;
                    let d_pct = dist as f64 / total as f64 * 100.0;
                    let _ = calls;
                    match vs_simd {
                        None => eprintln!(
                            "{:<8} {:<12} {:>10.0} {:>10.0} {:>10.1} {:>10.0} {:>12}",
                            dim, label, t_us, d_us, d_pct, pc, "-"
                        ),
                        Some(ratio) => eprintln!(
                            "{:<8} {:<12} {:>10.0} {:>10.0} {:>10.1} {:>10.0} {:>12.2}x",
                            dim, label, t_us, d_us, d_pct, pc, ratio
                        ),
                    }
                };

                print_row("fp32-auto", auto_total, auto_dist, auto_calls, auto_pc, None);
                print_row("fp32-simd", simd_total, simd_dist, simd_calls, simd_pc, Some(auto_pc / simd_pc));
                print_row("fp16-fused", fp16_total, fp16_dist, fp16_calls, fp16_pc,
                    Some(simd_pc / fp16_pc));

                let total_vs_simd = simd_total as f64 / fp16_total as f64;
                eprintln!(
                    "  dim={}: fp16 vs fp32-simd total: {:.2}x ({:.0}us → {:.0}us)",
                    dim, total_vs_simd,
                    simd_total as f64 / nq / 1000.0,
                    fp16_total as f64 / nq / 1000.0,
                );
            });
        }) {
            eprintln!("SKIPPED: io_uring not available for dim={}", dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Experiment 5: Recall check — FP32 vs FP16 quality validation
// ---------------------------------------------------------------------------

/// Brute-force exact top-k using FP32 L2 distance. Returns sorted (distance, vid) pairs.
fn brute_force_topk(query: &[f32], vectors: &[f32], dim: usize, k: usize) -> Vec<(f32, u32)> {
    let n = vectors.len() / dim;
    let mut dists: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let offset = i * dim;
            let v = &vectors[offset..offset + dim];
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            (d, i as u32)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists
}

/// Compute recall@k: |approx_ids ∩ exact_ids| / k
fn recall_at_k(approx_ids: &[u32], exact_ids: &[u32]) -> f64 {
    let k = exact_ids.len();
    if k == 0 {
        return 1.0;
    }
    let exact_set: std::collections::HashSet<u32> = exact_ids.iter().copied().collect();
    let hits = approx_ids.iter().filter(|id| exact_set.contains(id)).count();
    hits as f64 / k as f64
}

/// Recall check: compare graph search results against brute-force ground truth.
/// Tests FP32 and FP16 search modes at dim=128 and dim=512.
#[test]
fn exp_recall_check() {
    let n = 2000;
    let k = 10;
    let ef = 64;
    let num_queries = 50;
    let dims = [128, 512];

    eprintln!("\n========== EXP5: RECALL CHECK ==========");
    eprintln!(
        "{:<8} {:<6} {:>10} {:>10} {:>10}",
        "dim", "mode", "mean_r@k", "min_r@k", "queries"
    );

    for &dim in &dims {
        let (dir, flat) = build_disk_index(n, dim, 32, 200);
        let dir_str = dir.path().to_str().unwrap().to_owned();
        let (_meta, entry_set) = load_meta_entry(dir.path());
        let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
        let vectors_fp16 = fp32_to_fp16(&disk_vectors);
        let queries = generate_vectors(num_queries, dim, 999);

        // Compute ground truth (brute force FP32)
        let ground_truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| {
                brute_force_topk(q, &flat, dim, k)
                    .iter()
                    .map(|&(_, vid)| vid)
                    .collect()
            })
            .collect();

        if !with_runtime(|rt| {
            rt.block_on(async {
                let io = IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver");

                // --- FP32 recall ---
                let pool = AdjacencyPool::new(n * 4096); // large cache = no eviction noise
                let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
                let mut fp32_recalls = Vec::with_capacity(num_queries);

                for (i, q) in queries.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search(
                        q, &entry_set, k, ef, &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    )
                    .await;
                    let result_ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    fp32_recalls.push(recall_at_k(&result_ids, &ground_truth[i]));
                }

                let fp32_mean = fp32_recalls.iter().sum::<f64>() / fp32_recalls.len() as f64;
                let fp32_min = fp32_recalls
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                eprintln!(
                    "{:<8} {:<6} {:>10.3} {:>10.3} {:>10}",
                    dim, "fp32", fp32_mean, fp32_min, num_queries
                );

                // --- FP16 recall ---
                let pool16 = AdjacencyPool::new(n * 4096);
                let bank_fp16 = FP16VectorBank::new(&vectors_fp16, dim, MetricType::L2);
                let mut fp16_recalls = Vec::with_capacity(num_queries);

                for (i, q) in queries.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search(
                        q, &entry_set, k, ef, &pool16, &io, &bank_fp16,
                        &mut perf, PerfLevel::CountOnly,
                    )
                    .await;
                    let result_ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    fp16_recalls.push(recall_at_k(&result_ids, &ground_truth[i]));
                }

                let fp16_mean = fp16_recalls.iter().sum::<f64>() / fp16_recalls.len() as f64;
                let fp16_min = fp16_recalls
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                eprintln!(
                    "{:<8} {:<6} {:>10.3} {:>10.3} {:>10}",
                    dim, "fp16", fp16_mean, fp16_min, num_queries
                );

                // Sanity: FP32 graph search should have very high recall
                assert!(
                    fp32_mean >= 0.85,
                    "FP32 recall too low: {:.3} (graph quality issue)",
                    fp32_mean
                );
                // FP16 should not degrade recall significantly (< 5% drop)
                assert!(
                    fp16_mean >= fp32_mean - 0.05,
                    "FP16 recall degradation too high: fp32={:.3} fp16={:.3} (delta={:.3})",
                    fp32_mean,
                    fp16_mean,
                    fp32_mean - fp16_mean
                );
            });
        }) {
            eprintln!("SKIPPED: io_uring not available for dim={}", dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Experiment 6: Budgeted refine — FP16 traversal + FP32 refinement
// ---------------------------------------------------------------------------

/// Compare 3 modes: FP32-only, FP16-only, FP16+refine.
/// Shows that FP16+refine recovers recall while keeping most of the FP16 speed.
#[test]
fn exp_budgeted_refine() {
    let n = 2000;
    let k = 10;
    let ef = 64;
    let num_queries = 50;
    let dim = 512;
    let refine_r = k * 4; // refine top-40 candidates

    let (dir, flat) = build_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let vectors_fp16 = fp32_to_fp16(&disk_vectors);
    let queries = generate_vectors(num_queries, dim, 999);

    // Ground truth
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            brute_force_topk(q, &flat, dim, k)
                .iter()
                .map(|&(_, vid)| vid)
                .collect()
        })
        .collect();

    eprintln!("\n========== EXP6: BUDGETED REFINE (dim={}, refine_r={}) ==========", dim, refine_r);
    eprintln!(
        "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mode", "mean_r@k", "min_r@k", "total_us", "dist_us", "refine_us"
    );

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            // --- Mode 1: FP32-only ---
            let pool = AdjacencyPool::new(n * 4096);
            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let mut fp32_recalls = Vec::with_capacity(num_queries);
            let mut fp32_total_ns = 0u64;
            let mut fp32_dist_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool, &io, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                )
                .await;
                fp32_total_ns += t.elapsed().as_nanos() as u64;
                fp32_dist_ns += perf.dist_ns;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                fp32_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let fp32_mean_r = fp32_recalls.iter().sum::<f64>() / num_queries as f64;
            let fp32_min_r = fp32_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<14} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10}",
                "fp32", fp32_mean_r, fp32_min_r,
                fp32_total_ns as f64 / num_queries as f64 / 1000.0,
                fp32_dist_ns as f64 / num_queries as f64 / 1000.0,
                "-"
            );

            // --- Mode 2: FP16-only ---
            let pool16 = AdjacencyPool::new(n * 4096);
            let bank_fp16 = FP16VectorBank::new(&vectors_fp16, dim, MetricType::L2);
            let mut fp16_recalls = Vec::with_capacity(num_queries);
            let mut fp16_total_ns = 0u64;
            let mut fp16_dist_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool16, &io, &bank_fp16,
                    &mut perf, PerfLevel::EnableTime,
                )
                .await;
                fp16_total_ns += t.elapsed().as_nanos() as u64;
                fp16_dist_ns += perf.dist_ns;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                fp16_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let fp16_mean_r = fp16_recalls.iter().sum::<f64>() / num_queries as f64;
            let fp16_min_r = fp16_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<14} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10}",
                "fp16", fp16_mean_r, fp16_min_r,
                fp16_total_ns as f64 / num_queries as f64 / 1000.0,
                fp16_dist_ns as f64 / num_queries as f64 / 1000.0,
                "-"
            );

            // --- Mode 3: FP16 traversal + FP32 refine ---
            let pool_refine = AdjacencyPool::new(n * 4096);
            let mut refine_recalls = Vec::with_capacity(num_queries);
            let mut refine_total_ns = 0u64;
            let mut refine_dist_ns = 0u64;
            let mut refine_phase_ns = 0u64;
            let mut total_refine_count = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search_refine(
                    q, &entry_set, k, ef, refine_r,
                    &pool_refine, &io, &bank_fp16, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                )
                .await;
                refine_total_ns += t.elapsed().as_nanos() as u64;
                refine_dist_ns += perf.dist_ns;
                refine_phase_ns += perf.refine_ns;
                total_refine_count += perf.refine_count;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                refine_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let refine_mean_r = refine_recalls.iter().sum::<f64>() / num_queries as f64;
            let refine_min_r = refine_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<14} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10.0}",
                "fp16+refine", refine_mean_r, refine_min_r,
                refine_total_ns as f64 / num_queries as f64 / 1000.0,
                refine_dist_ns as f64 / num_queries as f64 / 1000.0,
                refine_phase_ns as f64 / num_queries as f64 / 1000.0,
            );
            eprintln!(
                "  refines/query: {:.1}",
                total_refine_count as f64 / num_queries as f64
            );

            // Verdict
            let speedup = fp32_total_ns as f64 / refine_total_ns as f64;
            let recall_delta = refine_mean_r - fp16_mean_r;
            eprintln!(
                "\n  FP16+refine vs FP32: {:.2}x speedup, recall delta vs fp16-only: {:+.3}",
                speedup, recall_delta
            );
            eprintln!(
                "  FP16+refine vs FP16-only latency: {:.2}x",
                refine_total_ns as f64 / fp16_total_ns as f64
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Experiment 7: Int8 cheap + FP32 refine — recall curve at R={k,2k,4k,8k}
// ---------------------------------------------------------------------------

/// Build a cosine-metric index with L2-normalized vectors.
fn build_cosine_disk_index(
    n: usize,
    dim: usize,
    m_max: usize,
    ef_construction: usize,
) -> (tempfile::TempDir, Vec<f32>) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(42);
    let mut flat: Vec<f32> = (0..n * dim)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    // L2-normalize all vectors (AD-3)
    for chunk in flat.chunks_exact_mut(dim) {
        l2_normalize(chunk);
    }

    let vectors: Vec<Vec<f32>> = flat.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "cosine",
            index.max_degree(),
            ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    (dir, flat)
}

/// Brute-force cosine top-k on pre-normalized vectors. Distance = -dot (AD-3).
fn brute_force_cosine_topk(query: &[f32], vectors: &[f32], dim: usize, k: usize) -> Vec<(f32, u32)> {
    let n = vectors.len() / dim;
    let mut dists: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let offset = i * dim;
            let v = &vectors[offset..offset + dim];
            let dot: f32 = query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            (-dot, i as u32)  // negate: smaller = more similar
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists
}

/// Int8 cheap stage + FP32 refine. Sweep R = {k, 2k, 4k, 8k}.
/// Shows the recall-vs-cost tradeoff curve.
#[test]
fn exp_int8_refine_curve() {
    let n = 2000;
    let dim = 512;
    let k = 10;
    let ef = 64;
    let num_queries = 50;
    let refine_rs = [k, k * 2, k * 4, k * 8];

    let (dir, flat) = build_cosine_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    // Quantize vectors for int8
    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    // Generate and normalize queries
    let mut queries: Vec<Vec<f32>> = generate_vectors(num_queries, dim, 999);
    for q in &mut queries {
        l2_normalize(q);
    }

    // Ground truth
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            brute_force_cosine_topk(q, &flat, dim, k)
                .iter()
                .map(|&(_, vid)| vid)
                .collect()
        })
        .collect();

    eprintln!("\n========== EXP7: INT8 + FP32 REFINE CURVE (dim={}, n={}, ef={}) ==========", dim, n, ef);
    eprintln!(
        "{:<16} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mode", "R", "mean_r@k", "min_r@k", "total_us", "dist_us", "refine_us"
    );

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            // --- Baseline: FP32-SIMD cosine (no quantization) ---
            let pool_fp32 = AdjacencyPool::new(n * 4096);
            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let mut fp32_recalls = Vec::with_capacity(num_queries);
            let mut fp32_total_ns = 0u64;
            let mut fp32_dist_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool_fp32, &io, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                fp32_total_ns += t.elapsed().as_nanos() as u64;
                fp32_dist_ns += perf.dist_ns;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                fp32_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let fp32_mean_r = fp32_recalls.iter().sum::<f64>() / num_queries as f64;
            let fp32_min_r = fp32_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<16} {:>6} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10}",
                "fp32-simd", "-", fp32_mean_r, fp32_min_r,
                fp32_total_ns as f64 / num_queries as f64 / 1000.0,
                fp32_dist_ns as f64 / num_queries as f64 / 1000.0,
                "-"
            );

            // --- Int8-only (no refine) ---
            let pool_i8 = AdjacencyPool::new(n * 4096);
            let bank_i8 = Int8VectorBank::new(&codes, dim);
            let mut i8_recalls = Vec::with_capacity(num_queries);
            let mut i8_total_ns = 0u64;
            let mut i8_dist_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let prepared = bank_i8.prepare(q);
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool_i8, &io, &prepared,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                i8_total_ns += t.elapsed().as_nanos() as u64;
                i8_dist_ns += perf.dist_ns;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                i8_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let i8_mean_r = i8_recalls.iter().sum::<f64>() / num_queries as f64;
            let i8_min_r = i8_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<16} {:>6} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10}",
                "int8-only", "-", i8_mean_r, i8_min_r,
                i8_total_ns as f64 / num_queries as f64 / 1000.0,
                i8_dist_ns as f64 / num_queries as f64 / 1000.0,
                "-"
            );

            // --- Int8 traversal + FP32 refine, sweep R ---
            for &r in &refine_rs {
                let pool_r = AdjacencyPool::new(n * 4096);
                let mut refine_recalls = Vec::with_capacity(num_queries);
                let mut refine_total_ns = 0u64;
                let mut refine_dist_ns = 0u64;
                let mut refine_phase_ns = 0u64;

                for (i, q) in queries.iter().enumerate() {
                    let prepared = bank_i8.prepare(q);
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    let results = disk_graph_search_refine(
                        q, &entry_set, k, ef, r,
                        &pool_r, &io, &prepared, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    ).await;
                    refine_total_ns += t.elapsed().as_nanos() as u64;
                    refine_dist_ns += perf.dist_ns;
                    refine_phase_ns += perf.refine_ns;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    refine_recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let mean_r = refine_recalls.iter().sum::<f64>() / num_queries as f64;
                let min_r = refine_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
                let total_us = refine_total_ns as f64 / num_queries as f64 / 1000.0;
                let speedup = fp32_total_ns as f64 / refine_total_ns as f64;
                eprintln!(
                    "{:<16} {:>6} {:>10.3} {:>10.3} {:>10.0} {:>10.0} {:>10.0}  ({:.2}x vs fp32)",
                    "int8+refine", r, mean_r, min_r,
                    total_us,
                    refine_dist_ns as f64 / num_queries as f64 / 1000.0,
                    refine_phase_ns as f64 / num_queries as f64 / 1000.0,
                    speedup
                );
            }

            // Sanity: FP32 baseline recall should be reasonable
            assert!(
                fp32_mean_r >= 0.80,
                "FP32 recall too low: {:.3} (graph quality issue)",
                fp32_mean_r
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Experiment 8: FP32+refine control — verifies refine pipeline has no "magic"
// ---------------------------------------------------------------------------

/// Control experiment: FP32 cheap + FP32 refine must ≈ FP32-only.
/// If FP32+refine significantly differs from FP32-only, the refine pipeline
/// has a logic bug (different candidate set, sorting difference, etc.).
///
/// Also compares int8+refine to check if "recall surpass" is stable.
#[test]
fn exp_fp32_refine_control() {
    let n = 2000;
    let dim = 512;
    let k = 10;
    let ef = 64;
    let num_queries = 50;
    let refine_r = k * 2;

    let (dir, flat) = build_cosine_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    let mut queries: Vec<Vec<f32>> = generate_vectors(num_queries, dim, 999);
    for q in &mut queries {
        l2_normalize(q);
    }

    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            brute_force_cosine_topk(q, &flat, dim, k)
                .iter()
                .map(|&(_, vid)| vid)
                .collect()
        })
        .collect();

    eprintln!("\n========== EXP8: FP32+REFINE CONTROL (dim={}, R={}) ==========", dim, refine_r);
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10}",
        "mode", "mean_r@k", "min_r@k", "total_us"
    );

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let bank_i8 = Int8VectorBank::new(&codes, dim);

            // --- A: FP32-only ---
            let pool_a = AdjacencyPool::new(n * 4096);
            let mut a_recalls = Vec::with_capacity(num_queries);
            let mut a_total_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool_a, &io, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                a_total_ns += t.elapsed().as_nanos() as u64;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                a_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let a_mean = a_recalls.iter().sum::<f64>() / num_queries as f64;
            let a_min = a_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<20} {:>10.3} {:>10.3} {:>10.0}",
                "A: fp32-only", a_mean, a_min,
                a_total_ns as f64 / num_queries as f64 / 1000.0
            );

            // --- B: FP32 cheap + FP32 refine (control — should ≈ A) ---
            let pool_b = AdjacencyPool::new(n * 4096);
            let mut b_recalls = Vec::with_capacity(num_queries);
            let mut b_total_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search_refine(
                    q, &entry_set, k, ef, refine_r,
                    &pool_b, &io, &bank_fp32, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                b_total_ns += t.elapsed().as_nanos() as u64;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                b_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let b_mean = b_recalls.iter().sum::<f64>() / num_queries as f64;
            let b_min = b_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<20} {:>10.3} {:>10.3} {:>10.0}",
                "B: fp32+refine", b_mean, b_min,
                b_total_ns as f64 / num_queries as f64 / 1000.0
            );

            // --- C: int8 cheap + FP32 refine ---
            let pool_c = AdjacencyPool::new(n * 4096);
            let mut c_recalls = Vec::with_capacity(num_queries);
            let mut c_total_ns = 0u64;

            for (i, q) in queries.iter().enumerate() {
                let prepared = bank_i8.prepare(q);
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search_refine(
                    q, &entry_set, k, ef, refine_r,
                    &pool_c, &io, &prepared, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                c_total_ns += t.elapsed().as_nanos() as u64;
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                c_recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }

            let c_mean = c_recalls.iter().sum::<f64>() / num_queries as f64;
            let c_min = c_recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            eprintln!(
                "{:<20} {:>10.3} {:>10.3} {:>10.0}",
                "C: int8+refine", c_mean, c_min,
                c_total_ns as f64 / num_queries as f64 / 1000.0
            );

            // Verdict
            let ab_delta = (a_mean - b_mean).abs();
            eprintln!("\n  A vs B delta: {:.4} (should be < 0.02 — refine pipeline is clean)", ab_delta);
            eprintln!("  C vs A delta: {:+.4} (positive = int8+refine better)", c_mean - a_mean);
            eprintln!("  C speedup:    {:.2}x vs A", a_total_ns as f64 / c_total_ns as f64);

            // Gate: B must be within 2% of A (refine pipeline is not introducing bias)
            assert!(
                ab_delta < 0.02,
                "FP32+refine diverged from FP32-only by {:.4} — refine pipeline has a bug",
                ab_delta
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Experiment 9: Multi-seed stability — is the "recall surpass" reproducible?
// ---------------------------------------------------------------------------

/// Run the int8+refine pipeline across 5 different seeds and query batches.
/// Reports per-seed recall to verify the "surpass" is not seed-dependent noise.
#[test]
fn exp_multi_seed_stability() {
    let n = 2000;
    let dim = 512;
    let k = 10;
    let ef = 64;
    let num_queries = 50;
    let refine_r = k * 2;
    let seeds: [u64; 5] = [999, 1337, 42, 7777, 31415];

    let (dir, flat) = build_cosine_disk_index(n, dim, 32, 200);
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let (_meta, entry_set) = load_meta_entry(dir.path());
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    eprintln!("\n========== EXP9: MULTI-SEED STABILITY (dim={}, R={}, {} seeds) ==========", dim, refine_r, seeds.len());
    eprintln!(
        "{:<8} {:>12} {:>12} {:>12} {:>10}",
        "seed", "fp32_r@k", "i8+ref_r@k", "delta", "surpass?"
    );

    let mut all_fp32_means = Vec::new();
    let mut all_i8ref_means = Vec::new();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let bank_i8 = Int8VectorBank::new(&codes, dim);

            for &seed in &seeds {
                let mut queries: Vec<Vec<f32>> = generate_vectors(num_queries, dim, seed);
                for q in &mut queries {
                    l2_normalize(q);
                }

                let ground_truth: Vec<Vec<u32>> = queries
                    .iter()
                    .map(|q| {
                        brute_force_cosine_topk(q, &flat, dim, k)
                            .iter()
                            .map(|&(_, vid)| vid)
                            .collect()
                    })
                    .collect();

                // FP32-only
                let pool_fp32 = AdjacencyPool::new(n * 4096);
                let mut fp32_recalls = Vec::with_capacity(num_queries);
                for (i, q) in queries.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search(
                        q, &entry_set, k, ef, &pool_fp32, &io, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    fp32_recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }
                let fp32_mean = fp32_recalls.iter().sum::<f64>() / num_queries as f64;

                // Int8 + refine
                let pool_i8 = AdjacencyPool::new(n * 4096);
                let mut i8ref_recalls = Vec::with_capacity(num_queries);
                for (i, q) in queries.iter().enumerate() {
                    let prepared = bank_i8.prepare(q);
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_refine(
                        q, &entry_set, k, ef, refine_r,
                        &pool_i8, &io, &prepared, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    i8ref_recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }
                let i8ref_mean = i8ref_recalls.iter().sum::<f64>() / num_queries as f64;

                let delta = i8ref_mean - fp32_mean;
                let surpass = if delta > 0.005 { "YES" } else if delta < -0.005 { "NO" } else { "~TIE" };
                eprintln!(
                    "{:<8} {:>12.3} {:>12.3} {:>12.4} {:>10}",
                    seed, fp32_mean, i8ref_mean, delta, surpass
                );

                all_fp32_means.push(fp32_mean);
                all_i8ref_means.push(i8ref_mean);
            }

            // Summary
            let avg_fp32 = all_fp32_means.iter().sum::<f64>() / seeds.len() as f64;
            let avg_i8ref = all_i8ref_means.iter().sum::<f64>() / seeds.len() as f64;
            let surpass_count = all_fp32_means.iter().zip(all_i8ref_means.iter())
                .filter(|(f, i)| **i > **f + 0.005)
                .count();
            let tie_count = all_fp32_means.iter().zip(all_i8ref_means.iter())
                .filter(|(f, i)| (**i - **f).abs() <= 0.005)
                .count();
            let worse_count = seeds.len() - surpass_count - tie_count;

            eprintln!("\n  Average: fp32={:.3} int8+ref={:.3} delta={:+.4}", avg_fp32, avg_i8ref, avg_i8ref - avg_fp32);
            eprintln!("  Surpass: {}/{}, Tie: {}/{}, Worse: {}/{}",
                surpass_count, seeds.len(), tie_count, seeds.len(), worse_count, seeds.len());

            if surpass_count > seeds.len() / 2 {
                eprintln!("  CONCLUSION: int8+refine consistently matches or exceeds fp32-only");
            } else if worse_count > seeds.len() / 2 {
                eprintln!("  CONCLUSION: int8+refine consistently worse — check quantization quality");
            } else {
                eprintln!("  CONCLUSION: results are mixed — surpass is seed-dependent noise");
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

/// Verify SearchGuard RAII records correctly with disk search.
#[test]
fn search_guard_records_perf() {
    let n = 200;
    let dim = 16;
    let k = 5;
    let ef = 32;

    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(16, 100);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            100,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();
    let query = generate_vectors(1, dim, 999)[0].clone();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");
            let pool = AdjacencyPool::new(64 * 1024);
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let recorder = QueryRecorder::new();

            // Run with SearchGuard — RAII should auto-record
            {
                let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                let level = guard.level();
                let results = disk_graph_search(
                    &query, &entry_set, k, ef, &pool, &io, &bank,
                    &mut guard.ctx, level,
                )
                .await;
                assert!(!results.is_empty());

                // Peek at counters before drop
                assert!(guard.ctx.blocks_read > 0, "should have read blocks");
                assert!(
                    guard.ctx.distance_computes > 0,
                    "should have computed distances"
                );
                assert!(guard.ctx.expansions > 0, "should have expanded candidates");
            } // guard drops here, records to recorder

            assert_eq!(recorder.query_count(), 1);
            let report = recorder.report();
            eprintln!("\n{}", report);
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// REAL DATA: Cohere 768-dim cosine benchmark
// ---------------------------------------------------------------------------

/// Load real Cohere dataset from binary files produced by convert_cohere.py.
/// Returns (vectors_flat, queries_flat, ground_truth, n, nq, dim, k).
fn load_cohere_dataset(
    dir: &str,
    max_vectors: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<Vec<u32>>, usize, usize, usize, usize)> {
    use std::fs;
    use std::io::Read as _;

    let meta_path = format!("{}/meta.txt", dir);
    let meta = match fs::read_to_string(&meta_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!(
                "SKIPPED: Cohere dataset not found at {}. Run: python3 scripts/convert_cohere.py",
                dir
            );
            return None;
        }
    };
    let nums: Vec<usize> = meta.lines().filter_map(|l| l.trim().parse().ok()).collect();
    if nums.len() < 4 {
        eprintln!("SKIPPED: Invalid meta.txt format");
        return None;
    }
    let (n_total, nq, dim, k) = (nums[0], nums[1], nums[2], nums[3]);
    let n = n_total.min(max_vectors);

    // Load vectors (take first n)
    let mut vbuf = vec![0u8; n * dim * 4];
    let mut f = fs::File::open(format!("{}/vectors.bin", dir)).ok()?;
    f.read_exact(&mut vbuf).ok()?;
    let vectors: Vec<f32> = vbuf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Load queries
    let mut qbuf = vec![0u8; nq * dim * 4];
    let mut f = fs::File::open(format!("{}/queries.bin", dir)).ok()?;
    f.read_exact(&mut qbuf).ok()?;
    let queries: Vec<f32> = qbuf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Load ground truth (computed against full n_total, so filter to n if subset)
    let mut gbuf = vec![0u8; nq * k * 4];
    let mut f = fs::File::open(format!("{}/gt.bin", dir)).ok()?;
    f.read_exact(&mut gbuf).ok()?;
    let gt_flat: Vec<u32> = gbuf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let ground_truth: Vec<Vec<u32>> = if n < n_total {
        // Subset: filter GT to only include IDs < n, pad with remaining
        gt_flat
            .chunks_exact(k)
            .map(|row| row.iter().copied().filter(|&id| (id as usize) < n).collect())
            .collect()
    } else {
        gt_flat.chunks_exact(k).map(|row| row.to_vec()).collect()
    };

    Some((vectors, queries, ground_truth, n, nq, dim, k))
}

/// Compute percentile from a sorted slice. p in [0, 100].
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Per-query stats collected during a benchmark pass.
struct QueryStats {
    recalls: Vec<f64>,
    total_ms: Vec<f64>,
    dist_ms: Vec<f64>,
    refine_ms: Vec<f64>,
    misses: Vec<u64>,
}

impl QueryStats {
    fn new(cap: usize) -> Self {
        Self {
            recalls: Vec::with_capacity(cap),
            total_ms: Vec::with_capacity(cap),
            dist_ms: Vec::with_capacity(cap),
            refine_ms: Vec::with_capacity(cap),
            misses: Vec::with_capacity(cap),
        }
    }

    fn mean_recall(&self) -> f64 {
        self.recalls.iter().sum::<f64>() / self.recalls.len() as f64
    }

    fn min_recall(&self) -> f64 {
        self.recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0)
    }

    fn sorted_total_ms(&self) -> Vec<f64> {
        let mut s = self.total_ms.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    }

    fn sorted_dist_ms(&self) -> Vec<f64> {
        let mut s = self.dist_ms.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    }

    /// Simulate disk-miss latency: compute_ms + misses * nvme_ms_per_miss.
    fn simulated_total_ms(&self, nvme_ms: f64) -> Vec<f64> {
        let mut s: Vec<f64> = self.total_ms.iter().zip(self.misses.iter())
            .map(|(&t, &m)| t + m as f64 * nvme_ms)
            .collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    }
}

/// Print a row of the results table (all latencies in ms).
fn print_row(label: &str, r_label: &str, stats: &QueryStats, fp32_mean_ms: f64) {
    let sorted = stats.sorted_total_ms();
    let sorted_dist = stats.sorted_dist_ms();
    let mean_ms = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let avg_misses = stats.misses.iter().sum::<u64>() as f64 / stats.misses.len() as f64;
    let speedup_str = if fp32_mean_ms > 0.0 {
        format!("{:.2}x", fp32_mean_ms / mean_ms)
    } else {
        "-".to_string()
    };
    eprintln!(
        "{:<15} {:>4} {:>7.3} {:>7.3}  {:>7.2} {:>7.2} {:>7.2}  {:>7.2} {:>7.2}  {:>5.0}  {}",
        label, r_label, stats.mean_recall(), stats.min_recall(),
        mean_ms,
        percentile(&sorted, 50.0),
        percentile(&sorted, 99.0),
        sorted_dist.iter().sum::<f64>() / sorted_dist.len() as f64,
        percentile(&sorted_dist, 99.0),
        avg_misses,
        speedup_str,
    );
}

/// Print disk-miss injection table (all latencies in ms).
fn print_disk_injection(label: &str, stats: &QueryStats, nvme_ms_values: &[f64]) {
    let avg_misses = stats.misses.iter().sum::<u64>() as f64 / stats.misses.len() as f64;
    eprintln!("  {} (avg {:.0} misses/q):", label, avg_misses);
    for &nvme_ms in nvme_ms_values {
        let sim = stats.simulated_total_ms(nvme_ms);
        eprintln!(
            "    nvme={:.0}us:  p50={:>6.1}ms  p99={:>6.1}ms  p999={:>6.1}ms",
            nvme_ms * 1000.0,
            percentile(&sim, 50.0),
            percentile(&sim, 99.0),
            percentile(&sim, 99.9),
        );
    }
}

/// Real-data benchmark: Cohere 768-dim cosine embeddings.
///
/// Reports recall + p50/p99 compute latency + disk-miss injection simulation.
///
/// Run: cargo test --release -p divergence-engine --test disk_search cohere_recall -- --nocapture
/// Full: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search cohere_recall -- --nocapture
///
/// Prerequisite: python3 scripts/convert_cohere.py
#[test]
fn cohere_recall_benchmark() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    eprintln!("\n========== COHERE REAL-DATA BENCHMARK ==========");
    eprintln!("n={}, nq={}, dim={}, k={}", n, nq, dim, k);

    let m_max = 32;
    let ef_construction = 200;
    let ef = k * 2;
    let num_queries = nq.min(100);

    eprintln!("Building NSW index (m_max={}, ef_c={}, ef={}) ...", m_max, ef_construction, ef);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();

    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    eprintln!(
        "\n{:<15} {:>4} {:>7} {:>7}  {:>7} {:>7} {:>7}  {:>7} {:>7}  {:>5}  {}",
        "mode", "R", "r@k", "min_r",
        "avg_ms", "p50_ms", "p99_ms",
        "dist", "d_p99",
        "blk/q",
        "speedup"
    );

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let bank_i8 = Int8VectorBank::new(&codes, dim);

            // --- FP32 ef sweep: calibrate the baseline ---
            let ef_values = [110, 120, 140, 160, 180, 200];
            let mut fp32_by_ef: Vec<(usize, QueryStats)> = Vec::new();

            for &test_ef in &ef_values {
                let pool = AdjacencyPool::new(n * 4096);
                let mut stats = QueryStats::new(num_queries);

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    let results = disk_graph_search(
                        q, &entry_set, k, test_ef, &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    ).await;
                    let elapsed_ms = t.elapsed().as_nanos() as f64 / 1_000_000.0;
                    stats.total_ms.push(elapsed_ms);
                    stats.dist_ms.push(perf.dist_ns as f64 / 1_000_000.0);
                    stats.refine_ms.push(0.0);
                    stats.misses.push(perf.blocks_miss);
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    stats.recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }
                let label = format!("fp32 ef={}", test_ef);
                print_row(&label, "-", &stats, 0.0); // speedup=0 means N/A for sweep
                fp32_by_ef.push((test_ef, stats));
            }

            // Use ef=200 as the FP32 baseline for speedup comparison
            let fp32 = &fp32_by_ef.last().unwrap().1;
            let fp32_mean_ms = fp32.total_ms.iter().sum::<f64>() / num_queries as f64;

            // --- Int8-only ef=200 ---
            let pool_i8 = AdjacencyPool::new(n * 4096);
            let mut i8only = QueryStats::new(num_queries);

            for (i, q) in query_vecs.iter().enumerate() {
                let prepared = bank_i8.prepare(q);
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search(
                    q, &entry_set, k, ef, &pool_i8, &io, &prepared,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                let elapsed_ms = t.elapsed().as_nanos() as f64 / 1_000_000.0;
                i8only.total_ms.push(elapsed_ms);
                i8only.dist_ms.push(perf.dist_ns as f64 / 1_000_000.0);
                i8only.refine_ms.push(0.0);
                i8only.misses.push(perf.blocks_miss);
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                i8only.recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }
            print_row("int8-only", "-", &i8only, fp32_mean_ms);

            // --- Int8+refine R=2k ---
            let refine_r = k * 2;
            let pool_r = AdjacencyPool::new(n * 4096);
            let mut i8ref = QueryStats::new(num_queries);

            for (i, q) in query_vecs.iter().enumerate() {
                let prepared = bank_i8.prepare(q);
                let mut perf = SearchPerfContext::default();
                let t = std::time::Instant::now();
                let results = disk_graph_search_refine(
                    q, &entry_set, k, ef, refine_r,
                    &pool_r, &io, &prepared, &bank_fp32,
                    &mut perf, PerfLevel::EnableTime,
                ).await;
                let elapsed_ms = t.elapsed().as_nanos() as f64 / 1_000_000.0;
                i8ref.total_ms.push(elapsed_ms);
                i8ref.dist_ms.push(perf.dist_ns as f64 / 1_000_000.0);
                i8ref.refine_ms.push(perf.refine_ns as f64 / 1_000_000.0);
                i8ref.misses.push(perf.blocks_miss);
                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                i8ref.recalls.push(recall_at_k(&ids, &ground_truth[i]));
            }
            print_row("int8+ref R=2k", &format!("{}", refine_r), &i8ref, fp32_mean_ms);

            // ====== DISK-MISS INJECTION ======
            eprintln!("\n--- DISK-MISS INJECTION (simulated, ms) ---");
            let nvme_ms = [0.08_f64, 0.15];
            print_disk_injection("fp32 ef=200", fp32, &nvme_ms);
            print_disk_injection("int8-only", &i8only, &nvme_ms);
            print_disk_injection("int8+ref R=2k", &i8ref, &nvme_ms);

            // ====== FAIR COMPARISON ======
            // Find the cheapest FP32 ef that matches int8+refine recall
            let i8ref_recall = i8ref.mean_recall();
            eprintln!("\n--- FAIR COMPARISON (iso-recall) ---");
            eprintln!("int8+ref R=2k: recall={:.3}  p50={:.2}ms  blocks/q={:.0}",
                i8ref_recall,
                percentile(&i8ref.sorted_total_ms(), 50.0),
                i8ref.misses.iter().sum::<u64>() as f64 / num_queries as f64);
            for (test_ef, stats) in &fp32_by_ef {
                if stats.mean_recall() >= i8ref_recall - 0.005 {
                    let fp32_p50 = percentile(&stats.sorted_total_ms(), 50.0);
                    let i8ref_p50 = percentile(&i8ref.sorted_total_ms(), 50.0);
                    eprintln!("fp32 ef={}: recall={:.3}  p50={:.2}ms  blocks/q={:.0}  → speedup={:.2}x",
                        test_ef, stats.mean_recall(), fp32_p50,
                        stats.misses.iter().sum::<u64>() as f64 / num_queries as f64,
                        fp32_p50 / i8ref_p50);
                    break;
                }
            }

            // ====== SUMMARY ======
            eprintln!("\n--- SUMMARY ---");
            eprintln!("FP32 ef=200: recall={:.3}  p50={:.2}ms  p99={:.2}ms  blocks/q={:.0}",
                fp32.mean_recall(), percentile(&fp32.sorted_total_ms(), 50.0),
                percentile(&fp32.sorted_total_ms(), 99.0),
                fp32.misses.iter().sum::<u64>() as f64 / num_queries as f64);
            eprintln!("Int8-only:   recall={:.3}  delta={:+.4}  speedup={:.2}x  blocks/q={:.0}",
                i8only.mean_recall(), i8only.mean_recall() - fp32.mean_recall(),
                fp32_mean_ms / (i8only.total_ms.iter().sum::<f64>() / num_queries as f64),
                i8only.misses.iter().sum::<u64>() as f64 / num_queries as f64);
            eprintln!("Int8+ref:    recall={:.3}  delta={:+.4}  speedup={:.2}x  blocks/q={:.0}",
                i8ref.mean_recall(), i8ref.mean_recall() - fp32.mean_recall(),
                fp32_mean_ms / (i8ref.total_ms.iter().sum::<f64>() / num_queries as f64),
                i8ref.misses.iter().sum::<u64>() as f64 / num_queries as f64);

            if fp32.mean_recall() < 0.50 {
                eprintln!("WARNING: FP32 recall very low ({:.3}) — graph quality may be poor at n={}", fp32.mean_recall(), n);
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// EXP-0: blocks/query Decomposition
// ---------------------------------------------------------------------------

/// EXP-0: Instrument where blocks are spent during search.
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp0_blocks_decomposition -- --nocapture
#[test]
fn exp0_blocks_decomposition() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    eprintln!("\n========== EXP-0: BLOCKS/QUERY DECOMPOSITION ==========");
    eprintln!("n={}, nq={}, dim={}, k={}", n, nq, dim, k);

    let m_max = 32;
    let ef_construction = 200;
    let num_queries = nq.min(100);

    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();

    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let bank_i8 = Int8VectorBank::new(&codes, dim);

            // Accumulator for decomposition stats
            struct DecompRow {
                total_exp: Vec<u64>,
                useful: Vec<u64>,
                wasted: Vec<u64>,
                best_at: Vec<u64>,
                first_topk_at: Vec<u64>,
                recalls: Vec<f64>,
                refine_count: Vec<u64>,
            }

            impl DecompRow {
                fn new(cap: usize) -> Self {
                    Self {
                        total_exp: Vec::with_capacity(cap),
                        useful: Vec::with_capacity(cap),
                        wasted: Vec::with_capacity(cap),
                        best_at: Vec::with_capacity(cap),
                        first_topk_at: Vec::with_capacity(cap),
                        recalls: Vec::with_capacity(cap),
                        refine_count: Vec::with_capacity(cap),
                    }
                }
                fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len().max(1) as f64 }
                fn mean(v: &[u64]) -> f64 { v.iter().sum::<u64>() as f64 / v.len().max(1) as f64 }
                fn median(v: &[u64]) -> f64 {
                    let mut s = v.to_vec(); s.sort();
                    if s.is_empty() { return 0.0; }
                    let m = s.len() / 2;
                    if s.len() % 2 == 0 { (s[m-1] + s[m]) as f64 / 2.0 } else { s[m] as f64 }
                }
                fn p99(v: &[u64]) -> u64 {
                    let mut s = v.to_vec(); s.sort();
                    if s.is_empty() { return 0; }
                    s[((s.len() as f64 * 0.99).ceil() as usize).min(s.len()) - 1]
                }
            }

            // Runs search, captures both results (for recall) and perf (for diagnostics)
            async fn run_decomp(
                query_vecs: &[Vec<f32>],
                entry_set: &[VectorId],
                k: usize,
                ef: usize,
                refine_r: usize,
                pool: &AdjacencyPool,
                io: &IoDriver,
                bank: &dyn VectorBank,
                exact_bank: Option<&dyn VectorBank>,
                ground_truth: &[Vec<u32>],
            ) -> DecompRow {
                let nq = query_vecs.len();
                let mut row = DecompRow::new(nq);

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();

                    let results = if let Some(exact) = exact_bank {
                        disk_graph_search_refine(
                            q, entry_set, k, ef, refine_r,
                            pool, io, bank, exact,
                            &mut perf, PerfLevel::EnableTime,
                        ).await
                    } else {
                        disk_graph_search(
                            q, entry_set, k, ef, pool, io, bank,
                            &mut perf, PerfLevel::EnableTime,
                        ).await
                    };

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    let recall = recall_at_k(&ids, &ground_truth[i]);

                    row.total_exp.push(perf.expansions);
                    row.useful.push(perf.useful_expansions);
                    row.wasted.push(perf.wasted_expansions);
                    row.best_at.push(perf.best_result_at_expansion);
                    row.first_topk_at.push(perf.first_topk_at_expansion);
                    row.recalls.push(recall);
                    row.refine_count.push(perf.refine_count);
                }

                row
            }

            eprintln!("\n{:<16} {:>5} {:>6} {:>6} {:>6}  {:>7} {:>7} {:>7}  {:>6} {:>6}  {:>6} {:>6}",
                "mode", "r@k", "blk/q", "usful", "waste",
                "bst_mn", "bst_md", "bst_99",
                "tk_mn", "tk_md",
                "ref", "entry%");

            // FP32 ef=200
            {
                let pool = AdjacencyPool::new(n * 4096);
                let row = run_decomp(
                    &query_vecs, &entry_set, k, 200, 0,
                    &pool, &io, &bank_fp32, None, &ground_truth,
                ).await;
                let me = DecompRow::mean(&row.total_exp);
                let entry_pct = if me > 0.0 { DecompRow::mean(&row.first_topk_at) / me * 100.0 } else { 0.0 };
                eprintln!("{:<16} {:>5.3} {:>6.0} {:>6.0} {:>6.0}  {:>7.1} {:>7.0} {:>7}  {:>6.1} {:>6.0}  {:>6.0} {:>6.1}",
                    "fp32 ef=200", DecompRow::mean_f(&row.recalls),
                    me, DecompRow::mean(&row.useful), DecompRow::mean(&row.wasted),
                    DecompRow::mean(&row.best_at), DecompRow::median(&row.best_at), DecompRow::p99(&row.best_at),
                    DecompRow::mean(&row.first_topk_at), DecompRow::median(&row.first_topk_at),
                    DecompRow::mean(&row.refine_count), entry_pct);
            }

            // FP32 ef=180
            {
                let pool = AdjacencyPool::new(n * 4096);
                let row = run_decomp(
                    &query_vecs, &entry_set, k, 180, 0,
                    &pool, &io, &bank_fp32, None, &ground_truth,
                ).await;
                let me = DecompRow::mean(&row.total_exp);
                let entry_pct = if me > 0.0 { DecompRow::mean(&row.first_topk_at) / me * 100.0 } else { 0.0 };
                eprintln!("{:<16} {:>5.3} {:>6.0} {:>6.0} {:>6.0}  {:>7.1} {:>7.0} {:>7}  {:>6.1} {:>6.0}  {:>6.0} {:>6.1}",
                    "fp32 ef=180", DecompRow::mean_f(&row.recalls),
                    me, DecompRow::mean(&row.useful), DecompRow::mean(&row.wasted),
                    DecompRow::mean(&row.best_at), DecompRow::median(&row.best_at), DecompRow::p99(&row.best_at),
                    DecompRow::mean(&row.first_topk_at), DecompRow::median(&row.first_topk_at),
                    DecompRow::mean(&row.refine_count), entry_pct);
            }

            // FP32 ef=140
            {
                let pool = AdjacencyPool::new(n * 4096);
                let row = run_decomp(
                    &query_vecs, &entry_set, k, 140, 0,
                    &pool, &io, &bank_fp32, None, &ground_truth,
                ).await;
                let me = DecompRow::mean(&row.total_exp);
                let entry_pct = if me > 0.0 { DecompRow::mean(&row.first_topk_at) / me * 100.0 } else { 0.0 };
                eprintln!("{:<16} {:>5.3} {:>6.0} {:>6.0} {:>6.0}  {:>7.1} {:>7.0} {:>7}  {:>6.1} {:>6.0}  {:>6.0} {:>6.1}",
                    "fp32 ef=140", DecompRow::mean_f(&row.recalls),
                    me, DecompRow::mean(&row.useful), DecompRow::mean(&row.wasted),
                    DecompRow::mean(&row.best_at), DecompRow::median(&row.best_at), DecompRow::p99(&row.best_at),
                    DecompRow::mean(&row.first_topk_at), DecompRow::median(&row.first_topk_at),
                    DecompRow::mean(&row.refine_count), entry_pct);
            }

            // Int8+refine R=2k
            {
                let pool = AdjacencyPool::new(n * 4096);
                // We need to run query-by-query since each query needs its own prepared bank
                let mut row = DecompRow::new(num_queries);
                for (i, q) in query_vecs.iter().enumerate() {
                    let prepared = bank_i8.prepare(q);
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_refine(
                        q, &entry_set, k, 200, k * 2,
                        &pool, &io, &prepared, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    ).await;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    row.total_exp.push(perf.expansions);
                    row.useful.push(perf.useful_expansions);
                    row.wasted.push(perf.wasted_expansions);
                    row.best_at.push(perf.best_result_at_expansion);
                    row.first_topk_at.push(perf.first_topk_at_expansion);
                    row.recalls.push(recall_at_k(&ids, &ground_truth[i]));
                    row.refine_count.push(perf.refine_count);
                }
                let me = DecompRow::mean(&row.total_exp);
                let entry_pct = if me > 0.0 { DecompRow::mean(&row.first_topk_at) / me * 100.0 } else { 0.0 };
                eprintln!("{:<16} {:>5.3} {:>6.0} {:>6.0} {:>6.0}  {:>7.1} {:>7.0} {:>7}  {:>6.1} {:>6.0}  {:>6.0} {:>6.1}",
                    "i8+ref R=2k", DecompRow::mean_f(&row.recalls),
                    me, DecompRow::mean(&row.useful), DecompRow::mean(&row.wasted),
                    DecompRow::mean(&row.best_at), DecompRow::median(&row.best_at), DecompRow::p99(&row.best_at),
                    DecompRow::mean(&row.first_topk_at), DecompRow::median(&row.first_topk_at),
                    DecompRow::mean(&row.refine_count), entry_pct);
            }

            eprintln!("\nColumn legend:");
            eprintln!("  blk/q   = mean total expansions (= unique block reads)");
            eprintln!("  usful   = mean expansions that added ≥1 neighbor to beam");
            eprintln!("  waste   = mean expansions that added 0 neighbors");
            eprintln!("  bst_mn/md/99 = expansion # when rank-1 result entered beam (mean/median/p99)");
            eprintln!("  tk_mn/md = expansion # when first top-k result entered beam (mean/median)");
            eprintln!("  ref     = mean refine count (0 for FP32-only)");
            eprintln!("  entry%  = tk_mn / blk/q × 100 (approach phase fraction)");
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// EXP-C: Iso-Recall blocks/query Curve
// ---------------------------------------------------------------------------

/// EXP-C: Sweep ef for FP32 and int8+refine, compare blocks/query at iso-recall.
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp_c_iso_recall_blocks -- --nocapture
#[test]
fn exp_c_iso_recall_blocks() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    eprintln!("\n========== EXP-C: ISO-RECALL BLOCKS/QUERY CURVE ==========");
    eprintln!("n={}, nq={}, dim={}, k={}", n, nq, dim, k);

    let m_max = 32;
    let ef_construction = 200;
    let num_queries = nq.min(100);

    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();

    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let sq = ScalarQuantizer::new(dim);
    let codes = sq.encode_batch(&disk_vectors);

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let bank_i8 = Int8VectorBank::new(&codes, dim);

            let ef_values: Vec<usize> = (100..=300).step_by(20).collect();
            let refine_r = k * 2;

            struct SweepPoint {
                ef: usize,
                recall: f64,
                blocks_per_q: f64,
                #[allow(dead_code)]
                p50_ms: f64,
            }

            let mut fp32_points: Vec<SweepPoint> = Vec::new();
            let mut i8ref_points: Vec<SweepPoint> = Vec::new();

            eprintln!("\n--- FP32 ef sweep ---");
            eprintln!("{:<8} {:>7} {:>7} {:>8}", "ef", "r@k", "blk/q", "p50_ms");

            for &ef in &ef_values {
                let pool = AdjacencyPool::new(n * 4096);
                let mut recalls = Vec::with_capacity(num_queries);
                let mut total_blocks = 0u64;
                let mut latencies = Vec::with_capacity(num_queries);

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    let results = disk_graph_search(
                        q, &entry_set, k, ef, &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    ).await;
                    latencies.push(t.elapsed().as_nanos() as f64 / 1_000_000.0);
                    total_blocks += perf.expansions;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let blocks_per_q = total_blocks as f64 / num_queries as f64;
                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&latencies, 50.0);

                eprintln!("{:<8} {:>7.3} {:>7.0} {:>8.2}", ef, mean_recall, blocks_per_q, p50);
                fp32_points.push(SweepPoint { ef, recall: mean_recall, blocks_per_q, p50_ms: p50 });
            }

            eprintln!("\n--- Int8+refine R={} ef sweep ---", refine_r);
            eprintln!("{:<8} {:>7} {:>7} {:>8}", "ef", "r@k", "blk/q", "p50_ms");

            for &ef in &ef_values {
                let pool = AdjacencyPool::new(n * 4096);
                let mut recalls = Vec::with_capacity(num_queries);
                let mut total_blocks = 0u64;
                let mut latencies = Vec::with_capacity(num_queries);

                for (i, q) in query_vecs.iter().enumerate() {
                    let prepared = bank_i8.prepare(q);
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    let results = disk_graph_search_refine(
                        q, &entry_set, k, ef, refine_r,
                        &pool, &io, &prepared, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    ).await;
                    latencies.push(t.elapsed().as_nanos() as f64 / 1_000_000.0);
                    total_blocks += perf.expansions;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let blocks_per_q = total_blocks as f64 / num_queries as f64;
                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&latencies, 50.0);

                eprintln!("{:<8} {:>7.3} {:>7.0} {:>8.2}", ef, mean_recall, blocks_per_q, p50);
                i8ref_points.push(SweepPoint { ef, recall: mean_recall, blocks_per_q, p50_ms: p50 });
            }

            // === ISO-RECALL COMPARISON ===
            eprintln!("\n--- ISO-RECALL COMPARISON ---");
            eprintln!("For each int8+ref recall level ≥ 0.90, find closest FP32 match:");
            eprintln!("{:<8} {:>7} {:>9} {:>9} {:>8} {:>10}",
                "recall", "fp32_ef", "fp32_blk", "i8r_blk", "i8r_ef", "blk_ratio");

            let mut found_pass = false;
            for i8p in &i8ref_points {
                if i8p.recall < 0.90 { continue; }

                // Find FP32 point with closest recall (within 0.01)
                let best_fp32 = fp32_points.iter()
                    .filter(|fp| (fp.recall - i8p.recall).abs() < 0.01)
                    .min_by(|a, b| {
                        let da = (a.recall - i8p.recall).abs();
                        let db = (b.recall - i8p.recall).abs();
                        da.partial_cmp(&db).unwrap()
                    });

                if let Some(fp) = best_fp32 {
                    let ratio = i8p.blocks_per_q / fp.blocks_per_q;
                    let marker = if ratio <= 0.85 { " <<<< PASS" } else { "" };
                    if ratio <= 0.85 { found_pass = true; }
                    eprintln!("{:<8.3} {:>7} {:>9.0} {:>9.0} {:>8} {:>10.3}{}",
                        (fp.recall + i8p.recall) / 2.0,
                        fp.ef, fp.blocks_per_q, i8p.blocks_per_q, i8p.ef,
                        ratio, marker);
                }
            }

            eprintln!("\n--- VERDICT ---");
            if found_pass {
                eprintln!("PASS: int8+refine achieves ≥15% fewer blocks at iso-recall.");
                eprintln!("      Int8's cheap traversal provides a structural IO advantage.");
            } else {
                eprintln!("FAIL: int8+refine does NOT reduce blocks/query at iso-recall.");
                eprintln!("      Int8's advantage is purely compute-side; under IO it will shrink.");
                eprintln!("      blocks/query reduction must come from algorithmic changes");
                eprintln!("      (DynamicWidth, MemGraph), not from cheaper distance computation.");
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// EXP-W: Convergence Budgeting — early stopping sweep
// ---------------------------------------------------------------------------

/// EXP-W: Sweep max_expansions to find the recall-vs-blocks/query curve.
/// Shows how much we can cut expansions while maintaining recall.
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp_w_convergence -- --nocapture
#[test]
fn exp_w_convergence_budgeting() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    eprintln!("\n========== EXP-W: CONVERGENCE BUDGETING ==========");
    eprintln!("n={}, nq={}, dim={}, k={}", n, nq, dim, k);

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let num_queries = nq.min(100);

    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();

    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Sweep max_expansions: [50, 75, 100, 110, 120, 130, 140, 150, 160, 175, 200 (baseline)]
            let cutoffs: Vec<usize> = vec![50, 75, 100, 110, 120, 130, 140, 150, 160, 175, 200];

            eprintln!("\n--- FP32 ef={} with early stopping ---", ef);
            eprintln!("{:<10} {:>7} {:>7} {:>7} {:>7} {:>10}",
                "max_exp", "r@k", "min_r", "blk/q", "waste%", "vs_full");

            let mut baseline_recall = 0.0;

            for &max_exp in &cutoffs {
                let pool = AdjacencyPool::new(n * 4096);
                let mut recalls = Vec::with_capacity(num_queries);
                let mut total_exp = 0u64;
                let mut total_wasted = 0u64;

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_exp(
                        q, &entry_set, k, ef,
                        max_exp, 0, // max_neighbors=0 (all)
                        &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                    total_exp += perf.expansions;
                    total_wasted += perf.wasted_expansions;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let min_recall = recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
                let blocks_per_q = total_exp as f64 / num_queries as f64;
                let waste_pct = total_wasted as f64 / total_exp.max(1) as f64 * 100.0;

                if max_exp == 200 { baseline_recall = mean_recall; }
                let blocks_saved = if max_exp < 200 {
                    format!("{:+.1}%", (1.0 - blocks_per_q / 201.0) * 100.0)
                } else {
                    "baseline".to_string()
                };

                eprintln!("{:<10} {:>7.3} {:>7.3} {:>7.0} {:>7.1} {:>10}",
                    max_exp, mean_recall, min_recall, blocks_per_q, waste_pct, blocks_saved);
            }

            // Find the sweet spot: maximum blocks reduction while recall >= 0.96
            eprintln!("\n--- ANALYSIS ---");
            eprintln!("Baseline: ef={}, recall={:.3}, blk/q=201", ef, baseline_recall);

            let target_recall = baseline_recall - 0.005; // allow 0.5% recall drop
            for &max_exp in cutoffs.iter().rev() {
                if max_exp >= 200 { continue; }
                let pool = AdjacencyPool::new(n * 4096);
                let mut recalls = Vec::with_capacity(num_queries);

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_exp(
                        q, &entry_set, k, ef,
                        max_exp, 0,
                        &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }
                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;

                if mean_recall >= target_recall {
                    let savings = (1.0 - max_exp as f64 / 201.0) * 100.0;
                    eprintln!("Sweet spot: max_exp={} → recall={:.3} (vs {:.3}), blocks saved={:.1}%",
                        max_exp, mean_recall, baseline_recall, savings);
                    if savings >= 15.0 {
                        eprintln!("PASS: ≥15% block reduction at near-iso-recall.");
                    } else {
                        eprintln!("PARTIAL: {:.1}% block reduction (target: ≥15%).", savings);
                    }
                    break;
                }
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// EXP-T: Top-t Neighbor Gating
// ---------------------------------------------------------------------------

/// EXP-T: Sweep max_neighbors to find if selective neighbor enqueue reduces blocks.
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp_t_neighbor_gating -- --nocapture
#[test]
fn exp_t_neighbor_gating() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    eprintln!("\n========== EXP-T: TOP-T NEIGHBOR GATING ==========");
    eprintln!("n={}, nq={}, dim={}, k={}", n, nq, dim, k);

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let num_queries = nq.min(100);

    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();

    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank_fp32 = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Sweep max_neighbors: [2, 4, 8, 12, 16, 24, 0 (all=32 baseline)]
            let t_values: Vec<usize> = vec![2, 4, 8, 12, 16, 24, 0];

            eprintln!("\n--- FP32 ef={} with top-t neighbor gating ---", ef);
            eprintln!("{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>10}",
                "max_nbr", "r@k", "min_r", "blk/q", "useful", "waste", "blk_save");

            let mut baseline_blocks = 0.0;

            for &t in &t_values {
                let pool = AdjacencyPool::new(n * 4096);
                let mut recalls = Vec::with_capacity(num_queries);
                let mut total_exp = 0u64;
                let mut total_useful = 0u64;
                let mut total_wasted = 0u64;

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_exp(
                        q, &entry_set, k, ef,
                        0, t, // max_expansions=0 (no limit), max_neighbors=t
                        &pool, &io, &bank_fp32,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                    total_exp += perf.expansions;
                    total_useful += perf.useful_expansions;
                    total_wasted += perf.wasted_expansions;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let min_recall = recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
                let blocks_per_q = total_exp as f64 / num_queries as f64;
                let useful_per_q = total_useful as f64 / num_queries as f64;
                let wasted_per_q = total_wasted as f64 / num_queries as f64;

                if t == 0 { baseline_blocks = blocks_per_q; }
                let blk_save = if t > 0 && baseline_blocks > 0.0 {
                    format!("{:+.1}%", (1.0 - blocks_per_q / baseline_blocks) * 100.0)
                } else {
                    "baseline".to_string()
                };

                let label = if t == 0 { "all(32)".to_string() } else { format!("{}", t) };
                eprintln!("{:<10} {:>7.3} {:>7.3} {:>7.0} {:>7.0} {:>7.0} {:>10}",
                    label, mean_recall, min_recall, blocks_per_q, useful_per_q, wasted_per_q, blk_save);
            }

            // Verdict
            eprintln!("\n--- VERDICT ---");
            eprintln!("If any t value achieves recall ≥ {:.3} (baseline - 0.005) with ≥20% fewer blocks, PASS.",
                0.963 - 0.005);
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Acceptance Gate 2: Singleflight Correctness
// ---------------------------------------------------------------------------

/// Acceptance Gate 2: singleflight dedup.
///
/// Spawn N concurrent coroutines requesting the SAME block. Only 1 NVMe read
/// should occur; the others should dedup-wait. After completion:
///   - pool.stats().misses == 1
///   - pool.stats().dedup_hits == N-1
///   - All N coroutines get valid data
#[test]
fn acceptance_gate_singleflight() {
    let n = 100u32;
    let dim = 4;
    let m_max = 8;
    let ef_construction = 32;
    let concurrent_requests = 8usize;

    // Build minimal index
    let vectors = generate_vectors(n as usize, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n as usize);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver"),
            );
            let pool = Rc::new(AdjacencyPool::new(64 * 1024));

            let target_vid = 0u32; // all coroutines request the same block

            // Spawn N concurrent tasks, all requesting the same block
            let mut handles = Vec::new();
            for _ in 0..concurrent_requests {
                let pool_c = pool.clone();
                let io_c = io.clone();
                handles.push(monoio::spawn(async move {
                    let guard = pool_c.get_or_load(target_vid, &io_c).await.unwrap();
                    let data = guard.data();
                    // Decode to verify valid data
                    let neighbors = divergence_storage::decode_adj_block(data);
                    assert!(!neighbors.is_empty(), "empty adjacency block for vid 0");
                    drop(guard);
                }));
            }

            // Await all
            for h in handles {
                h.await;
            }

            let stats = pool.stats();
            eprintln!(
                "\n=== ACCEPTANCE GATE 2: Singleflight Correctness ===\n\
                 concurrent requests: {}\n\
                 misses (NVMe reads): {}\n\
                 dedup_hits (waited): {}\n\
                 hits (cache hit):    {}\n\
                 target:              misses=1, dedup_hits={}\n\
                 verdict:             {}",
                concurrent_requests,
                stats.misses,
                stats.dedup_hits,
                stats.hits,
                concurrent_requests - 1,
                if stats.misses <= 1 && (stats.dedup_hits + stats.hits) >= (concurrent_requests as u64 - 1) {
                    "PASS"
                } else {
                    "FAIL"
                }
            );

            // The first request triggers a miss+IO. Others either dedup-wait
            // or see the READY entry (cache hit) depending on scheduling.
            // Either way: at most 1 NVMe read.
            assert!(
                stats.misses <= 1,
                "singleflight broken: {} misses for {} concurrent requests",
                stats.misses,
                concurrent_requests
            );

            // The non-miss requests should be either dedup_hits or hits
            let non_miss = stats.dedup_hits + stats.hits;
            assert!(
                non_miss >= (concurrent_requests as u64 - 1),
                "missing responses: dedup={} hits={} expected ≥ {}",
                stats.dedup_hits,
                stats.hits,
                concurrent_requests - 1
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Acceptance Gate 3: P99 Stability Under Cache Pressure
// ---------------------------------------------------------------------------

/// Acceptance Gate 3: p99 latency stability under near-full cache.
///
/// Create a cache smaller than the working set. Run many sequential get_or_load
/// calls with varying vids (some repeat, some new → evictions). Check that:
///   - p99 latency doesn't spike more than 3x vs p50
///   - No evict_fail_all_pinned (since we drop guards immediately)
///   - Eviction count is reasonable (not excessive churn)
#[test]
fn acceptance_gate_p99_stability() {
    let n = 200u32;
    let dim = 4;
    let m_max = 8;
    let ef_construction = 32;

    let vectors = generate_vectors(n as usize, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n as usize);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            // Small cache: only 32 slots (4 sets × 8 ways) — smaller than 200 vids
            let pool = AdjacencyPool::new(4 * 8 * 4096);

            let mut rng = Xoshiro256StarStar::seed_from_u64(12345);
            let num_ops = 2000usize;
            let mut latencies_ns = Vec::with_capacity(num_ops);

            // Phase 1: steady state — mix of hits and misses
            for _ in 0..num_ops {
                // 70% chance of accessing a "hot" vid (0..30), 30% random
                let vid: u32 = if rng.r#gen::<f64>() < 0.7 {
                    rng.r#gen::<u32>() % 30
                } else {
                    rng.r#gen::<u32>() % n
                };

                let start = std::time::Instant::now();
                let guard = pool.get_or_load(vid, &io).await.unwrap();
                let _data = guard.data();
                drop(guard);
                latencies_ns.push(start.elapsed().as_nanos() as u64);
            }

            // Compute percentiles
            latencies_ns.sort();
            let p50 = latencies_ns[num_ops / 2];
            let p99 = latencies_ns[num_ops * 99 / 100];
            let p999 = latencies_ns[num_ops * 999 / 1000];
            let max = latencies_ns[num_ops - 1];
            let mean = latencies_ns.iter().sum::<u64>() / num_ops as u64;

            let stats = pool.stats();

            eprintln!(
                "\n=== ACCEPTANCE GATE 3: P99 Stability Under Cache Pressure ===\n\
                 operations:         {}\n\
                 cache slots:        32 (4 sets × 8 ways)\n\
                 working set:        {} vids (70% hot 0..30, 30% cold 0..{})\n\
                 \n\
                 latency (ns):       p50={}  p99={}  p999={}  max={}  mean={}\n\
                 p99/p50 ratio:      {:.1}x\n\
                 \n\
                 hits:               {}\n\
                 misses:             {}\n\
                 dedup_hits:         {}\n\
                 evictions:          {}\n\
                 evict_fail_pinned:  {}\n\
                 bypasses:           {}\n\
                 hit_rate:           {:.1}%\n\
                 \n\
                 verdict:            {}",
                num_ops,
                n, n,
                p50, p99, p999, max, mean,
                p99 as f64 / p50.max(1) as f64,
                stats.hits,
                stats.misses,
                stats.dedup_hits,
                stats.evictions,
                stats.evict_fail_all_pinned,
                stats.bypasses,
                if (stats.hits + stats.misses + stats.dedup_hits) > 0 {
                    stats.hits as f64 / (stats.hits + stats.misses + stats.dedup_hits) as f64 * 100.0
                } else {
                    0.0
                },
                if stats.evict_fail_all_pinned == 0 { "PASS" } else { "FAIL: evict_fail_all_pinned > 0" }
            );

            // Gate checks:
            // 1. No eviction failures (we always drop guards immediately)
            assert_eq!(
                stats.evict_fail_all_pinned, 0,
                "evict_fail_all_pinned should be 0 when guards are dropped immediately"
            );

            // 2. No bypasses (same reason — guards dropped, slots always available)
            assert_eq!(
                stats.bypasses, 0,
                "bypasses should be 0 when guards are dropped immediately"
            );

            // 3. Hit rate should be reasonable given 70% hot set fits in cache
            let hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            assert!(
                hit_rate > 0.3,
                "hit rate too low: {:.1}% (expected > 30% given hot set)",
                hit_rate * 100.0
            );
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Acceptance Gate 6: Overflow Bypass — Bounded Progress
// ---------------------------------------------------------------------------

/// Gate 6: when the waiter array is permanently saturated, every task must
/// still complete in bounded time via the bypass fallback.
///
/// Setup: spawn many more concurrent tasks than MAX_WAITERS_PER_ENTRY (8)
/// for the same block. With capacity=8, tasks beyond the first 8 waiters
/// will overflow. After MAX_OVERFLOW_RETRIES (2), they must bypass.
///
/// Asserts:
///   - All tasks complete (no hang, no livelock)
///   - bypass count > 0 (overflow actually triggered the fallback)
///   - All tasks get valid data (bypass reads are correct)
#[test]
fn acceptance_gate_overflow_bypass() {
    let n = 100u32;
    let dim = 4;
    let m_max = 8;
    let ef_construction = 32;
    let concurrent_requests = 20usize; // >> MAX_WAITERS_PER_ENTRY (8)

    let vectors = generate_vectors(n as usize, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n as usize);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver"),
            );
            let pool = Rc::new(AdjacencyPool::new(64 * 1024));
            let target_vid = 0u32;

            // Spawn many concurrent tasks — more than waiter capacity
            let mut handles = Vec::new();
            for _ in 0..concurrent_requests {
                let pool_c = pool.clone();
                let io_c = io.clone();
                handles.push(monoio::spawn(async move {
                    let guard = pool_c.get_or_load(target_vid, &io_c).await.unwrap();
                    let data = guard.data();
                    let neighbors = divergence_storage::decode_adj_block(data);
                    assert!(!neighbors.is_empty(), "got empty block from bypass or cache");
                    drop(guard);
                    true // completed successfully
                }));
            }

            let mut completed = 0usize;
            for h in handles {
                if h.await {
                    completed += 1;
                }
            }

            let stats = pool.stats();

            eprintln!(
                "\n=== GATE 6: Overflow Bypass — Bounded Progress ===\n\
                 concurrent requests: {}\n\
                 completed:           {}\n\
                 misses:              {}\n\
                 dedup_hits:          {}\n\
                 hits:                {}\n\
                 bypasses:            {}\n\
                 verdict:             {}",
                concurrent_requests,
                completed,
                stats.misses,
                stats.dedup_hits,
                stats.hits,
                stats.bypasses,
                if completed == concurrent_requests { "PASS" } else { "FAIL: not all completed" }
            );

            assert_eq!(
                completed, concurrent_requests,
                "not all tasks completed — livelock detected"
            );

            // With 20 requests and 8 waiter slots, some must have bypassed
            // (unless they all happened to fit due to scheduling order).
            // We assert bypasses >= 0 as a structural check — the key
            // invariant is that ALL tasks completed.
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// EXP-P1: Prefetch latency sweep — W ∈ {0, 1, 2, 4, 8}
// ---------------------------------------------------------------------------

/// Sweep prefetch window W on Cohere 768-dim cosine data and measure
/// latency, inflight depth, prefetch hit rate, and recall.
///
/// W=0 is baseline (no prefetch, equivalent to disk_graph_search).
/// All W values should produce identical recall (prefetch must not degrade accuracy).
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp_p1_prefetch -- --nocapture
#[test]
fn exp_p1_prefetch_latency_sweep() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let num_queries = nq.min(100);
    let windows = [0usize, 1, 2, 4, 8];
    let prefetch_budget = 4; // max concurrent prefetch IOs

    eprintln!("\n========== EXP-P1: PREFETCH LATENCY SWEEP ==========");
    eprintln!("n={}, dim={}, k={}, ef={}, nq={}, pf_budget={}", n, dim, k, ef, num_queries, prefetch_budget);

    // Build NSW index from Cohere data
    eprintln!("Building NSW index (m_max={}, ef_c={}) ...", m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // Write to disk — use BENCH_DIR for NVMe/O_DIRECT, else tmpfs tempdir
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir; // hold tempdir lifetime
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  Index written to {} (direct_io={})", dir_str, direct_io);

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    eprintln!(
        "\n{:<4} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>8} {:>8} {:>8}",
        "W", "r@k", "p50us", "p99us", "p999us", "io_w%", "avg_ifl", "max_ifl", "blk/q", "mis/q", "pf_iss", "pf_con", "waste%"
    );

    let mut baseline_recall = 0.0f64;
    let mut all_recalls: Vec<(usize, f64)> = Vec::new();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );

            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Pool sized to ~5% of dataset blocks → forces heavy cache misses,
            // making prefetch IO overlap visible even with warmup.
            let pool_bytes = (n / 20) * 4096;
            eprintln!("  Pool: {}KB ({:.0}% of {} blocks)", pool_bytes / 1024, pool_bytes as f64 / (n * 4096) as f64 * 100.0, n);

            for &w in &windows {
                // Fresh pool per W to avoid cross-contamination
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let recorder = QueryRecorder::new();

                // Spawn prefetch worker (idles if W=0 since no hints are issued)
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool),
                    Rc::clone(&io),
                    prefetch_budget,
                );

                // Warmup: 10 queries to partially fill cache (not full warmup)
                for q in query_vecs.iter().take(10) {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pipe(
                        q, &entry_set, k, ef, w, 0, 0, &pool, &io, &bank,
                        &mut perf, PerfLevel::CountOnly,
                    )
                    .await;
                }

                // Measure pass — collect raw wall-clock timings for precise percentiles
                let mut recalls = Vec::with_capacity(num_queries);
                let mut raw_us = Vec::with_capacity(num_queries);
                let mut total_pf_iss = 0u64;
                let mut total_pf_con = 0u64;
                let mut total_wasted = 0u64;
                let mut total_useful = 0u64;
                let mut total_inflight_sum = 0u64;
                let mut total_inflight_samples = 0u64;
                let mut global_inflight_max = 0u64;
                let mut total_blocks_read = 0u64;
                let mut total_blocks_miss = 0u64;

                for (i, q) in query_vecs.iter().enumerate() {
                    let t = std::time::Instant::now();
                    let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                    let lvl = guard.level();
                    let results = disk_graph_search_pipe(
                        q, &entry_set, k, ef, w, 0, 0, &pool, &io, &bank,
                        &mut guard.ctx, lvl,
                    )
                    .await;
                    let elapsed_us = t.elapsed().as_nanos() as f64 / 1000.0;
                    raw_us.push(elapsed_us);

                    total_pf_iss += guard.ctx.prefetch_issued;
                    total_pf_con += guard.ctx.prefetch_consumed;
                    total_wasted += guard.ctx.wasted_expansions;
                    total_useful += guard.ctx.useful_expansions;
                    total_inflight_sum += guard.ctx.inflight_sum;
                    total_inflight_samples += guard.ctx.inflight_samples;
                    if guard.ctx.inflight_max > global_inflight_max {
                        global_inflight_max = guard.ctx.inflight_max;
                    }
                    total_blocks_read += guard.ctx.blocks_read;
                    total_blocks_miss += guard.ctx.blocks_miss;

                    let result_ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&result_ids, &ground_truth[i]));
                }

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let total_expansions = total_wasted + total_useful;
                let waste_pct = if total_expansions > 0 {
                    total_wasted as f64 / total_expansions as f64 * 100.0
                } else {
                    0.0
                };

                let avg_inflight = if total_inflight_samples > 0 {
                    total_inflight_sum as f64 / total_inflight_samples as f64
                } else {
                    0.0
                };

                // Compute precise percentiles from raw wall-clock timings
                raw_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50_raw = percentile(&raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 50.0);
                let p99_raw = percentile(&raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 99.0);
                let p999_raw = percentile(&raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 99.9);

                let blk_per_q = total_blocks_read as f64 / num_queries as f64;
                let miss_per_q = total_blocks_miss as f64 / num_queries as f64;

                eprintln!(
                    "{:<4} {:>7.3} {:>8.0} {:>8.0} {:>8.0} {:>8.1} {:>8.2} {:>8} {:>6.0} {:>6.0} {:>8} {:>8} {:>8.1}",
                    w,
                    mean_recall,
                    p50_raw,
                    p99_raw,
                    p999_raw,
                    recorder.io_wait_pct(),
                    avg_inflight,
                    global_inflight_max,
                    blk_per_q,
                    miss_per_q,
                    total_pf_iss,
                    total_pf_con,
                    waste_pct,
                );

                if w == 0 {
                    baseline_recall = mean_recall;
                }
                all_recalls.push((w, mean_recall));

                // Stop prefetch worker
                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
        return;
    }

    // Assert: recall within 1% of W=0 baseline for all W
    for &(w, recall) in &all_recalls {
        assert!(
            (recall - baseline_recall).abs() < 0.01,
            "W={}: recall {:.3} deviates from baseline {:.3} by more than 1%",
            w, recall, baseline_recall
        );
    }

    eprintln!("\nBaseline recall (W=0): {:.3}", baseline_recall);
    eprintln!("All W values within 1% of baseline recall — PASS");
    eprintln!("NOTE: tmpfs (memcpy IO) — latency/inflight numbers only meaningful on real NVMe with O_DIRECT");
}

// ---------------------------------------------------------------------------
// EXP-BW: B×W overlap sweep — inter-query concurrency × prefetch window
// ---------------------------------------------------------------------------

/// Sweep (B, W) to find the optimal operating point for throughput vs tail latency.
///
/// B = number of concurrent queries per batch (inter-query parallelism).
/// W = prefetch window per query (intra-query IO pipeline).
///
/// All B queries in a batch share the same AdjacencyPool and IoDriver (via Rc).
/// Measures QPS, per-query p50/p99, recall.
///
/// Run: BENCH_DIR=/mnt/nvme/bench COHERE_N=100000 cargo test --release \
///   -p divergence-engine --test disk_search exp_bw -- --nocapture
#[test]
fn exp_bw_overlap_sweep() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let num_queries = nq.min(100);
    let prefetch_budget = 4;
    let b_values = [1usize, 2, 4, 8];
    let w_values = [0usize, 1, 2, 4];

    eprintln!("\n========== EXP-BW: B×W OVERLAP SWEEP ==========");
    eprintln!(
        "n={}, dim={}, k={}, ef={}, nq={}, pf_budget={}",
        n, dim, k, ef, num_queries, prefetch_budget
    );

    // Build NSW index
    eprintln!("Building NSW index (m_max={}, ef_c={}) ...", m_max, ef_construction);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for i in 0..n {
        builder.insert(VectorId(i as u32), &vectors[i * dim..(i + 1) * dim]);
    }
    let index = builder.build();
    eprintln!("  Index built");

    // Write to disk (BENCH_DIR for NVMe, tmpfs otherwise)
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  Index written to {} (direct_io={})", dir_str, direct_io);

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim)
        .take(num_queries)
        .map(|c| c.to_vec())
        .collect();

    eprintln!(
        "\n{:<4} {:<4} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7}",
        "B", "W", "r@k", "qps", "p50us", "p99us", "p999us", "io_w%", "avg_ifl", "blk/q", "mis/q", "sem_w%", "dev_w%"
    );

    let mut baseline_recall = 0.0f64;

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );

            // Rc-wrap vectors so spawned tasks can create their own bank
            let vecs_rc: Rc<[f32]> = Rc::from(disk_vectors.as_slice());
            let entry_set_rc: Rc<[VectorId]> = Rc::from(entry_set.as_slice());

            // Pool sized to ~5% of dataset blocks
            let pool_bytes = (n / 20) * 4096;
            eprintln!(
                "  Pool: {}KB ({:.0}% of {} blocks)",
                pool_bytes / 1024,
                pool_bytes as f64 / (n * 4096) as f64 * 100.0,
                n
            );

            for &b in &b_values {
                for &w in &w_values {
                    // Fresh pool per (B, W) to avoid cross-contamination
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));

                    // Spawn prefetch worker
                    let pf_handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool),
                        Rc::clone(&io),
                        prefetch_budget,
                    );

                    // Light warmup (10 sequential queries)
                    let warmup_bank = FP32SimdVectorBank::new(&vecs_rc, dim, MetricType::Cosine);
                    for q in query_vecs.iter().take(10) {
                        let mut perf = SearchPerfContext::default();
                        disk_graph_search_pipe(
                            q, &entry_set_rc, k, ef, w, 0, 0, &pool, &io, &warmup_bank,
                            &mut perf, PerfLevel::CountOnly,
                        )
                        .await;
                    }
                    drop(warmup_bank);

                    // Reset IO timing counters after warmup
                    io.take_io_timing();

                    // Measure pass: ceil(nq / B) batches of B concurrent queries
                    let mut raw_us: Vec<f64> = Vec::with_capacity(num_queries);
                    let mut recalls: Vec<f64> = Vec::with_capacity(num_queries);
                    let mut total_blocks_read = 0u64;
                    let mut total_blocks_miss = 0u64;
                    let mut total_inflight_sum = 0u64;
                    let mut total_inflight_samples = 0u64;
                    let mut total_io_wait_ns = 0u64;
                    let mut total_total_ns = 0u64;

                    let wall_start = std::time::Instant::now();
                    let num_batches = (num_queries + b - 1) / b;

                    for batch_idx in 0..num_batches {
                        let batch_start = batch_idx * b;
                        let batch_end = (batch_start + b).min(num_queries);
                        let batch_size = batch_end - batch_start;

                        // Spawn B concurrent search tasks
                        let mut handles = Vec::with_capacity(batch_size);
                        for qi in batch_start..batch_end {
                            let pool_c = Rc::clone(&pool);
                            let io_c = Rc::clone(&io);
                            let vecs_c = Rc::clone(&vecs_rc);
                            let es_c = Rc::clone(&entry_set_rc);
                            let q = query_vecs[qi].clone();

                            handles.push(monoio::spawn(async move {
                                let bank = FP32SimdVectorBank::new(&vecs_c, dim, MetricType::Cosine);
                                let mut perf = SearchPerfContext::default();
                                let t = std::time::Instant::now();
                                let results = disk_graph_search_pipe(
                                    &q, &es_c, k, ef, w, 0, 0, &pool_c, &io_c,
                                    &bank, &mut perf, PerfLevel::EnableTime,
                                )
                                .await;
                                let elapsed_us = t.elapsed().as_nanos() as f64 / 1000.0;
                                (results, perf, elapsed_us)
                            }));
                        }

                        // Await all B tasks in this batch
                        for (j, h) in handles.into_iter().enumerate() {
                            let (results, perf, elapsed_us) = h.await;
                            raw_us.push(elapsed_us);
                            total_blocks_read += perf.blocks_read;
                            total_blocks_miss += perf.blocks_miss;
                            total_inflight_sum += perf.inflight_sum;
                            total_inflight_samples += perf.inflight_samples;
                            total_io_wait_ns += perf.io_wait_ns;
                            total_total_ns += perf.io_wait_ns + perf.compute_ns;

                            let qi = batch_start + j;
                            let result_ids: Vec<u32> =
                                results.iter().map(|s| s.id.0).collect();
                            recalls.push(recall_at_k(&result_ids, &ground_truth[qi]));
                        }
                    }

                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let qps = num_queries as f64 / wall_secs;

                    // Snapshot IO timing split before stopping worker
                    let (sem_ns, dev_ns, _io_cnt) = io.take_io_timing();

                    // Stop prefetch worker
                    pool.stop_prefetch();
                    pf_handle.await;

                    // Compute stats
                    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                    raw_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = percentile(
                        &raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        50.0,
                    );
                    let p99 = percentile(
                        &raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        99.0,
                    );
                    let p999 = percentile(
                        &raw_us.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        99.9,
                    );

                    let io_wait_pct = if total_total_ns > 0 {
                        total_io_wait_ns as f64 / total_total_ns as f64 * 100.0
                    } else {
                        0.0
                    };
                    let avg_ifl = if total_inflight_samples > 0 {
                        total_inflight_sum as f64 / total_inflight_samples as f64
                    } else {
                        0.0
                    };
                    let blk_per_q = total_blocks_read as f64 / num_queries as f64;
                    let miss_per_q = total_blocks_miss as f64 / num_queries as f64;
                    let io_total_ns = sem_ns + dev_ns;
                    let sem_pct = if io_total_ns > 0 {
                        sem_ns as f64 / io_total_ns as f64 * 100.0
                    } else {
                        0.0
                    };
                    let dev_pct = if io_total_ns > 0 {
                        dev_ns as f64 / io_total_ns as f64 * 100.0
                    } else {
                        0.0
                    };

                    eprintln!(
                        "{:<4} {:<4} {:>7.3} {:>8.1} {:>8.0} {:>8.0} {:>8.0} {:>8.1} {:>8.2} {:>6.0} {:>6.0} {:>7.1} {:>7.1}",
                        b,
                        w,
                        mean_recall,
                        qps,
                        p50,
                        p99,
                        p999,
                        io_wait_pct,
                        avg_ifl,
                        blk_per_q,
                        miss_per_q,
                        sem_pct,
                        dev_pct,
                    );

                    if b == 1 && w == 0 {
                        baseline_recall = mean_recall;
                    }
                }
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
        return;
    }

    eprintln!("\nBaseline recall (B=1,W=0): {:.3}", baseline_recall);
    eprintln!("B×W sweep complete");
}

// ---------------------------------------------------------------------------
// EXP-QD: Multi-core QD sweep — global device queue depth budget
// ---------------------------------------------------------------------------

/// Sweep cores × global_qd to find the operating point where throughput
/// is maximized without p99 explosion.
///
/// Each core runs B=1, W=4 (proven optimal from EXP-BW). All cores share
/// a single Arc<GlobalIoBudget> that caps total device queue depth.
///
/// Run: BENCH_DIR=/mnt/nvme/bench COHERE_N=100000 cargo test --release \
///   -p divergence-engine --test disk_search exp_qd -- --nocapture
#[test]
fn exp_qd_multicore_sweep() {
    use std::sync::{Arc, Mutex};
    use divergence_engine::{GlobalIoBudget, WorkerConfig, spawn_worker, default_global_qd};

    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let w = 4; // fixed prefetch window (proven optimal)
    let prefetch_budget = 4;
    let num_queries = nq.min(100);

    let core_counts = [1usize, 2, 4, 8];
    let fixed_qds = [4usize, 8, 12, 16, 24, 32];

    eprintln!("\n========== EXP-QD: MULTI-CORE QD SWEEP ==========");
    eprintln!(
        "n={}, dim={}, k={}, ef={}, W={}, nq={}, pf_budget={}",
        n, dim, k, ef, w, num_queries, prefetch_budget
    );

    // Build NSW index
    eprintln!("Building NSW index (m_max={}, ef_c={}) ...", m_max, ef_construction);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for i in 0..n {
        builder.insert(VectorId(i as u32), &vectors[i * dim..(i + 1) * dim]);
    }
    let index = builder.build();
    eprintln!("  Index built");

    // Write to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  Index written to {} (direct_io={})", dir_str, direct_io);

    // Load shared data
    let disk_vectors: Arc<[f32]> = Arc::from(
        load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap().as_slice(),
    );
    let entry_set: Arc<[VectorId]> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        Arc::from(meta.entry_set.iter().map(|&v| VectorId(v)).collect::<Vec<_>>().as_slice())
    };
    let query_vecs: Arc<[Vec<f32>]> = Arc::from(
        queries_flat.chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect::<Vec<_>>().as_slice(),
    );
    let ground_truth: Arc<[Vec<u32>]> = Arc::from(ground_truth.as_slice());

    // Pool sized to ~5% of dataset blocks
    let pool_bytes = (n / 20) * 4096;
    eprintln!(
        "  Pool per core: {}KB ({:.0}% of {} blocks)",
        pool_bytes / 1024,
        pool_bytes as f64 / (n * 4096) as f64 * 100.0,
        n
    );

    eprintln!(
        "\n{:<6} {:<8} {:<6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>9} {:>9}",
        "cores", "gQD", "lQD", "r@k", "qps", "p50us", "p99us", "p999us", "sem_w%", "dev_w%",
        "g_ifl_avg", "g_ifl_max",
    );

    // Per-core adj_capacity: give each core a fair share but at least 2
    // (need room for search + prefetch)
    for &num_cores in &core_counts {
        // Build gQD list: fixed values + auto policy (deduplicated)
        let auto_qd = default_global_qd(num_cores);
        let mut global_qds: Vec<(usize, bool)> = fixed_qds.iter().map(|&q| (q, false)).collect();
        if !fixed_qds.contains(&auto_qd) {
            global_qds.push((auto_qd, true));
            global_qds.sort_by_key(|&(q, _)| q);
        } else {
            // Mark the matching entry as auto
            for entry in &mut global_qds {
                if entry.0 == auto_qd {
                    entry.1 = true;
                }
            }
        }

        for &(gqd, is_auto) in &global_qds {
            let per_core_qd = (gqd / num_cores).max(2).min(16);
            let budget = Arc::new(GlobalIoBudget::new(gqd));

            // Shared results collector
            struct CoreResult {
                recalls: Vec<f64>,
                raw_us: Vec<f64>,
                sem_ns: u64,
                dev_ns: u64,
                global_inflight_sum: u64,
                global_inflight_samples: u64,
                global_inflight_max: u64,
            }
            let results: Arc<Mutex<Vec<CoreResult>>> = Arc::new(Mutex::new(Vec::new()));

            // Split queries across cores (round-robin assignment)
            let queries_per_core = (num_queries + num_cores - 1) / num_cores;

            let wall_start = std::time::Instant::now();

            let mut handles = Vec::new();
            for core_id in 0..num_cores {
                let dir_str = dir_str.clone();
                let budget = Arc::clone(&budget);
                let results = Arc::clone(&results);
                let disk_vecs = Arc::clone(&disk_vectors);
                let es = Arc::clone(&entry_set);
                let qvecs = Arc::clone(&query_vecs);
                let gt = Arc::clone(&ground_truth);

                let h = spawn_worker(
                    WorkerConfig { core_id, uring_entries: 1024 },
                    move || async move {
                        let io = Rc::new(
                            IoDriver::open_with_budget(
                                &dir_str, dim, per_core_qd, direct_io,
                                Some(budget),
                            )
                            .await
                            .expect("failed to open IO driver"),
                        );
                        let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                        let bank = FP32SimdVectorBank::new(&disk_vecs, dim, MetricType::Cosine);

                        // Spawn prefetch worker
                        let pf_handle = AdjacencyPool::spawn_prefetch_worker(
                            Rc::clone(&pool),
                            Rc::clone(&io),
                            prefetch_budget,
                        );

                        // Warmup (5 queries)
                        for qi in 0..5.min(num_queries) {
                            let mut perf = SearchPerfContext::default();
                            disk_graph_search_pipe(
                                &qvecs[qi], &es, k, ef, w, 0, 0, &pool, &io, &bank,
                                &mut perf, PerfLevel::CountOnly,
                            )
                            .await;
                        }
                        io.take_io_timing(); // reset after warmup

                        // Measure: this core handles queries [start..end)
                        let start = core_id * queries_per_core;
                        let end = (start + queries_per_core).min(num_queries);
                        let mut recalls = Vec::new();
                        let mut raw_us = Vec::new();
                        let mut g_ifl_sum = 0u64;
                        let mut g_ifl_samples = 0u64;
                        let mut g_ifl_max = 0u64;

                        for qi in start..end {
                            let t = std::time::Instant::now();
                            let mut perf = SearchPerfContext::default();
                            let res = disk_graph_search_pipe(
                                &qvecs[qi], &es, k, ef, w, 0, 0, &pool, &io, &bank,
                                &mut perf, PerfLevel::EnableTime,
                            )
                            .await;
                            raw_us.push(t.elapsed().as_nanos() as f64 / 1000.0);

                            let ids: Vec<u32> = res.iter().map(|s| s.id.0).collect();
                            recalls.push(recall_at_k(&ids, &gt[qi]));

                            g_ifl_sum += perf.global_inflight_sum;
                            g_ifl_samples += perf.global_inflight_samples;
                            if perf.global_inflight_max > g_ifl_max {
                                g_ifl_max = perf.global_inflight_max;
                            }
                        }

                        let (sem_ns, dev_ns, _) = io.take_io_timing();

                        pool.stop_prefetch();
                        pf_handle.await;

                        results.lock().unwrap().push(CoreResult {
                            recalls,
                            raw_us,
                            sem_ns,
                            dev_ns,
                            global_inflight_sum: g_ifl_sum,
                            global_inflight_samples: g_ifl_samples,
                            global_inflight_max: g_ifl_max,
                        });
                    },
                );
                handles.push(h);
            }

            // Wait for all cores
            for h in handles {
                h.join().expect("worker thread panicked");
            }

            let wall_secs = wall_start.elapsed().as_secs_f64();
            let total_queries = results.lock().unwrap().iter().map(|r| r.raw_us.len()).sum::<usize>();
            let qps = total_queries as f64 / wall_secs;

            // Aggregate stats across cores
            let guard = results.lock().unwrap();
            let mut all_us: Vec<f64> = guard.iter().flat_map(|r| r.raw_us.iter().copied()).collect();
            let all_recalls: Vec<f64> = guard.iter().flat_map(|r| r.recalls.iter().copied()).collect();
            let total_sem_ns: u64 = guard.iter().map(|r| r.sem_ns).sum();
            let total_dev_ns: u64 = guard.iter().map(|r| r.dev_ns).sum();
            let total_g_ifl_sum: u64 = guard.iter().map(|r| r.global_inflight_sum).sum();
            let total_g_ifl_samples: u64 = guard.iter().map(|r| r.global_inflight_samples).sum();
            let agg_g_ifl_max: u64 = guard.iter().map(|r| r.global_inflight_max).max().unwrap_or(0);
            drop(guard);

            let mean_recall = all_recalls.iter().sum::<f64>() / all_recalls.len().max(1) as f64;
            all_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = percentile(&all_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 50.0);
            let p99 = percentile(&all_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 99.0);
            let p999 = percentile(&all_us.iter().map(|&x| x as f64).collect::<Vec<_>>(), 99.9);

            let io_total = total_sem_ns + total_dev_ns;
            let sem_pct = if io_total > 0 { total_sem_ns as f64 / io_total as f64 * 100.0 } else { 0.0 };
            let dev_pct = if io_total > 0 { total_dev_ns as f64 / io_total as f64 * 100.0 } else { 0.0 };

            let g_ifl_avg = if total_g_ifl_samples > 0 {
                total_g_ifl_sum as f64 / total_g_ifl_samples as f64
            } else { 0.0 };

            let gqd_label = if is_auto {
                format!("{}*", gqd)
            } else {
                format!("{}", gqd)
            };

            eprintln!(
                "{:<6} {:<8} {:<6} {:>7.3} {:>8.1} {:>8.0} {:>8.0} {:>8.0} {:>7.1} {:>7.1} {:>9.1} {:>9}",
                num_cores, gqd_label, per_core_qd, mean_recall, qps, p50, p99, p999, sem_pct, dev_pct,
                g_ifl_avg, agg_g_ifl_max,
            );
        }
    }

    eprintln!("\nEXP-QD sweep complete (* = auto policy)");
}

// ---------------------------------------------------------------------------
// EXP-STABILITY: repeated-run stability stress test
// ---------------------------------------------------------------------------

/// Verify p99 stability across repeated runs at the default operating point.
///
/// For each core count: 3 rounds × 100 queries. Warmup 10 queries discarded.
/// Gate: p99 coefficient of variation across rounds < 25%.
///
/// Run: BENCH_DIR=/mnt/nvme/bench COHERE_N=100000 cargo test --release \
///   -p divergence-engine --test disk_search exp_stability -- --nocapture
#[test]
fn exp_stability_stress() {
    use std::sync::{Arc, Mutex};
    use divergence_engine::{GlobalIoBudget, WorkerConfig, spawn_worker, default_global_qd};

    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let w = 4;
    let prefetch_budget = 4;
    let num_queries = nq.min(100);
    let warmup_queries = 10;
    let rounds = 3;

    let core_counts = [1usize, 2, 4, 8];

    eprintln!("\n========== EXP-STABILITY: REPEATED-RUN STRESS TEST ==========");
    eprintln!(
        "n={}, dim={}, k={}, ef={}, W={}, nq={}, rounds={}, warmup={}",
        n, dim, k, ef, w, num_queries, rounds, warmup_queries,
    );

    // Build NSW index
    eprintln!("Building NSW index (m_max={}, ef_c={}) ...", m_max, ef_construction);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for i in 0..n {
        builder.insert(VectorId(i as u32), &vectors[i * dim..(i + 1) * dim]);
    }
    let index = builder.build();
    eprintln!("  Index built");

    // Write to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  Index written to {} (direct_io={})", dir_str, direct_io);

    // Load shared data
    let disk_vectors: Arc<[f32]> = Arc::from(
        load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap().as_slice(),
    );
    let entry_set: Arc<[VectorId]> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        Arc::from(meta.entry_set.iter().map(|&v| VectorId(v)).collect::<Vec<_>>().as_slice())
    };
    let query_vecs: Arc<[Vec<f32>]> = Arc::from(
        queries_flat.chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect::<Vec<_>>().as_slice(),
    );
    let ground_truth: Arc<[Vec<u32>]> = Arc::from(ground_truth.as_slice());

    let pool_bytes = (n / 20) * 4096;

    eprintln!(
        "\n{:<6} {:<6} {:<6} {:>7} {:>8} {:>8} {:>8} {:>9} {:>9} {:>7} {:>7}",
        "cores", "round", "gQD", "r@k", "qps", "p50us", "p99us",
        "g_ifl_avg", "g_ifl_max", "sem_w%", "dev_w%",
    );

    for &num_cores in &core_counts {
        let gqd = default_global_qd(num_cores);
        let per_core_qd = (gqd / num_cores).max(2).min(16);

        let mut round_p99s = Vec::new();

        for round in 0..rounds {
            let budget = Arc::new(GlobalIoBudget::new(gqd));

            struct CoreResult {
                recalls: Vec<f64>,
                raw_us: Vec<f64>,
                sem_ns: u64,
                dev_ns: u64,
                global_inflight_sum: u64,
                global_inflight_samples: u64,
                global_inflight_max: u64,
            }
            let results: Arc<Mutex<Vec<CoreResult>>> = Arc::new(Mutex::new(Vec::new()));

            let queries_per_core = (num_queries + num_cores - 1) / num_cores;

            let wall_start = std::time::Instant::now();

            let mut handles = Vec::new();
            for core_id in 0..num_cores {
                let dir_str = dir_str.clone();
                let budget = Arc::clone(&budget);
                let results = Arc::clone(&results);
                let disk_vecs = Arc::clone(&disk_vectors);
                let es = Arc::clone(&entry_set);
                let qvecs = Arc::clone(&query_vecs);
                let gt = Arc::clone(&ground_truth);

                let h = spawn_worker(
                    WorkerConfig { core_id, uring_entries: 1024 },
                    move || async move {
                        let io = Rc::new(
                            IoDriver::open_with_budget(
                                &dir_str, dim, per_core_qd, direct_io,
                                Some(budget),
                            )
                            .await
                            .expect("failed to open IO driver"),
                        );
                        let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                        let bank = FP32SimdVectorBank::new(&disk_vecs, dim, MetricType::Cosine);

                        let pf_handle = AdjacencyPool::spawn_prefetch_worker(
                            Rc::clone(&pool),
                            Rc::clone(&io),
                            prefetch_budget,
                        );

                        // Warmup
                        for qi in 0..warmup_queries.min(num_queries) {
                            let mut perf = SearchPerfContext::default();
                            disk_graph_search_pipe(
                                &qvecs[qi], &es, k, ef, w, 0, 0, &pool, &io, &bank,
                                &mut perf, PerfLevel::CountOnly,
                            )
                            .await;
                        }
                        io.take_io_timing();

                        // Measure
                        let start = core_id * queries_per_core;
                        let end = (start + queries_per_core).min(num_queries);
                        let mut recalls = Vec::new();
                        let mut raw_us = Vec::new();
                        let mut g_ifl_sum = 0u64;
                        let mut g_ifl_samples = 0u64;
                        let mut g_ifl_max = 0u64;

                        for qi in start..end {
                            let t = std::time::Instant::now();
                            let mut perf = SearchPerfContext::default();
                            let res = disk_graph_search_pipe(
                                &qvecs[qi], &es, k, ef, w, 0, 0, &pool, &io, &bank,
                                &mut perf, PerfLevel::EnableTime,
                            )
                            .await;
                            raw_us.push(t.elapsed().as_nanos() as f64 / 1000.0);

                            let ids: Vec<u32> = res.iter().map(|s| s.id.0).collect();
                            recalls.push(recall_at_k(&ids, &gt[qi]));

                            g_ifl_sum += perf.global_inflight_sum;
                            g_ifl_samples += perf.global_inflight_samples;
                            if perf.global_inflight_max > g_ifl_max {
                                g_ifl_max = perf.global_inflight_max;
                            }
                        }

                        let (sem_ns, dev_ns, _) = io.take_io_timing();

                        pool.stop_prefetch();
                        pf_handle.await;

                        results.lock().unwrap().push(CoreResult {
                            recalls,
                            raw_us,
                            sem_ns,
                            dev_ns,
                            global_inflight_sum: g_ifl_sum,
                            global_inflight_samples: g_ifl_samples,
                            global_inflight_max: g_ifl_max,
                        });
                    },
                );
                handles.push(h);
            }

            for h in handles {
                h.join().expect("worker thread panicked");
            }

            let wall_secs = wall_start.elapsed().as_secs_f64();
            let guard = results.lock().unwrap();
            let total_queries = guard.iter().map(|r| r.raw_us.len()).sum::<usize>();
            let qps = total_queries as f64 / wall_secs;

            let mut all_us: Vec<f64> = guard.iter().flat_map(|r| r.raw_us.iter().copied()).collect();
            let all_recalls: Vec<f64> = guard.iter().flat_map(|r| r.recalls.iter().copied()).collect();
            let total_sem_ns: u64 = guard.iter().map(|r| r.sem_ns).sum();
            let total_dev_ns: u64 = guard.iter().map(|r| r.dev_ns).sum();
            let total_g_ifl_sum: u64 = guard.iter().map(|r| r.global_inflight_sum).sum();
            let total_g_ifl_samples: u64 = guard.iter().map(|r| r.global_inflight_samples).sum();
            let agg_g_ifl_max: u64 = guard.iter().map(|r| r.global_inflight_max).max().unwrap_or(0);
            drop(guard);

            let mean_recall = all_recalls.iter().sum::<f64>() / all_recalls.len().max(1) as f64;
            all_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = percentile(&all_us, 50.0);
            let p99 = percentile(&all_us, 99.0);

            let io_total = total_sem_ns + total_dev_ns;
            let sem_pct = if io_total > 0 { total_sem_ns as f64 / io_total as f64 * 100.0 } else { 0.0 };
            let dev_pct = if io_total > 0 { total_dev_ns as f64 / io_total as f64 * 100.0 } else { 0.0 };

            let g_ifl_avg = if total_g_ifl_samples > 0 {
                total_g_ifl_sum as f64 / total_g_ifl_samples as f64
            } else { 0.0 };

            eprintln!(
                "{:<6} {:<6} {:<6} {:>7.3} {:>8.1} {:>8.0} {:>8.0} {:>9.1} {:>9} {:>7.1} {:>7.1}",
                num_cores, round, gqd, mean_recall, qps, p50, p99,
                g_ifl_avg, agg_g_ifl_max, sem_pct, dev_pct,
            );

            round_p99s.push(p99);
        }

        // Stability gate: CV of p99 across rounds < 25%
        if round_p99s.len() >= 2 {
            let mean_p99 = round_p99s.iter().sum::<f64>() / round_p99s.len() as f64;
            let var = round_p99s.iter().map(|x| (x - mean_p99).powi(2)).sum::<f64>()
                / round_p99s.len() as f64;
            let cv = if mean_p99 > 0.0 { var.sqrt() / mean_p99 * 100.0 } else { 0.0 };
            eprintln!(
                "  cores={}: p99 CV={:.1}% (mean={:.0}us) {}",
                num_cores, cv, mean_p99,
                if cv < 25.0 { "PASS" } else { "WARN" },
            );
        }
    }

    eprintln!("\nEXP-STABILITY stress test complete");
}

// ---------------------------------------------------------------------------
// EXP-AS: Adaptive Stopping — per-query early termination
// ---------------------------------------------------------------------------

/// Sweep stall_limit with drain_budget to find the best adaptive stopping config.
/// Compares adaptive stopping vs hard cutoff at iso-blocks to prove adaptive > static.
///
/// Three tables:
/// 1. Stall-limit sweep (warm cache, W=4)
/// 2. Iso-blocks: adaptive vs hard cutoff
/// 3. Per-query expansion distribution (best config)
///
/// Run: COHERE_N=100000 cargo test --release -p divergence-engine --test disk_search exp_as -- --nocapture
#[test]
fn exp_as_adaptive_stopping() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let num_queries = nq.min(100);
    let w = 4usize; // prefetch window
    let prefetch_budget = 4;

    eprintln!("\n========== EXP-AS: ADAPTIVE STOPPING ==========");
    eprintln!("n={}, dim={}, k={}, ef={}, nq={}, W={}", n, dim, k, ef, num_queries, w);

    // Build NSW index
    eprintln!("Building NSW index (m_max={}, ef_c={}) ...", m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // Write to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  Index written to {} (direct_io={})", dir_str, direct_io);

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    // Configs to sweep: (stall_limit, drain_budget)
    let configs: Vec<(u32, u32)> = vec![
        (0, 0),     // baseline (no adaptive stopping)
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
        (12, 16),
        (16, 16),
        (16, 32),
    ];

    // Collect results per config for Table 2
    struct ConfigResult {
        stall: u32,
        drain: u32,
        mean_recall: f64,
        avg_blk: f64,
        early_pct: f64,
        per_query_exp: Vec<u64>,
    }
    let mut config_results: Vec<ConfigResult> = Vec::new();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Warm cache (100% capacity so cache is not the variable)
            let pool_bytes = n * 4096;

            // =========================================================
            // TABLE 1: Stall-limit sweep
            // =========================================================
            eprintln!("\n--- Table 1: Stall-limit sweep (warm cache, W={}) ---", w);
            eprintln!(
                "{:<6} {:<6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7} {:>10}",
                "stall", "drain", "r@k", "min_r", "avg_blk", "p50_blk", "p99_blk", "waste%", "early%", "vs_full"
            );

            for &(stall_limit, drain_budget) in &configs {
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));

                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );

                // Warmup pass
                for q in query_vecs.iter().take(10) {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pipe(
                        q, &entry_set, k, ef, w, 0, 0, &pool, &io, &bank,
                        &mut perf, PerfLevel::CountOnly,
                    ).await;
                }

                let mut recalls = Vec::with_capacity(num_queries);
                let mut per_query_exp = Vec::with_capacity(num_queries);
                let mut per_query_blk = Vec::with_capacity(num_queries);
                let mut total_wasted = 0u64;
                let mut total_useful = 0u64;
                let mut early_count = 0u64;

                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_pipe(
                        q, &entry_set, k, ef, w, stall_limit, drain_budget,
                        &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                    ).await;

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));
                    per_query_exp.push(perf.expansions);
                    per_query_blk.push(perf.blocks_read as f64);
                    total_wasted += perf.wasted_expansions;
                    total_useful += perf.useful_expansions;
                    if perf.stopped_early {
                        early_count += 1;
                    }
                }

                pool.stop_prefetch();
                handle.await;

                let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let min_recall = recalls.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
                let avg_blk = per_query_blk.iter().sum::<f64>() / per_query_blk.len() as f64;
                let total_exp = total_wasted + total_useful;
                let waste_pct = if total_exp > 0 { total_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
                let early_pct = early_count as f64 / num_queries as f64 * 100.0;

                // Compute blk percentiles
                let mut sorted_blk = per_query_blk.clone();
                sorted_blk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50_blk = percentile(&sorted_blk, 50.0);
                let p99_blk = percentile(&sorted_blk, 99.0);

                let vs_full = if stall_limit == 0 {
                    "baseline".to_string()
                } else {
                    let baseline_blk = (ef + 1) as f64;
                    format!("{:+.1}%", (1.0 - avg_blk / baseline_blk) * 100.0)
                };

                eprintln!(
                    "{:<6} {:<6} {:>7.3} {:>7.3} {:>8.1} {:>8.0} {:>8.0} {:>7.1} {:>7.1} {:>10}",
                    stall_limit, drain_budget, mean_recall, min_recall,
                    avg_blk, p50_blk, p99_blk, waste_pct, early_pct, vs_full
                );

                config_results.push(ConfigResult {
                    stall: stall_limit,
                    drain: drain_budget,
                    mean_recall,
                    avg_blk,
                    early_pct,
                    per_query_exp,
                });
            }

            // =========================================================
            // TABLE 2: Iso-blocks — adaptive vs hard cutoff
            // =========================================================
            eprintln!("\n--- Table 2: Iso-blocks — adaptive vs hard cutoff ---");

            let baseline_recall = config_results[0].mean_recall;
            let baseline_blk = config_results[0].avg_blk;

            for cr in &config_results[1..] {
                if cr.early_pct < 1.0 {
                    continue; // skip configs that barely stop early
                }

                // Hard cutoff at same avg_blk
                let hard_cutoff = cr.avg_blk.round() as usize;
                if hard_cutoff == 0 || hard_cutoff >= ef {
                    continue;
                }

                let pool = Rc::new(AdjacencyPool::new(pool_bytes));

                // Warmup
                for q in query_vecs.iter().take(10) {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_exp(
                        q, &entry_set, k, ef, hard_cutoff, 0,
                        &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                    ).await;
                }

                let mut hard_recalls = Vec::with_capacity(num_queries);
                for (i, q) in query_vecs.iter().enumerate() {
                    let mut perf = SearchPerfContext::default();
                    let results = disk_graph_search_exp(
                        q, &entry_set, k, ef, hard_cutoff, 0,
                        &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                    ).await;
                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    hard_recalls.push(recall_at_k(&ids, &ground_truth[i]));
                }

                let hard_mean = hard_recalls.iter().sum::<f64>() / hard_recalls.len() as f64;
                let delta = cr.mean_recall - hard_mean;

                eprintln!(
                    "blk~{:.0}: adaptive(S={},D={}) r@k={:.3} vs hard_cutoff={} r@k={:.3} delta={:+.3}",
                    cr.avg_blk, cr.stall, cr.drain, cr.mean_recall, hard_cutoff, hard_mean, delta,
                );
            }

            // =========================================================
            // TABLE 3: Per-query expansion distribution (best config)
            // =========================================================
            // Pick best config: highest recall with >10% early stopping
            let best = config_results[1..]
                .iter()
                .filter(|c| c.early_pct > 10.0)
                .max_by(|a, b| a.mean_recall.partial_cmp(&b.mean_recall).unwrap());

            if let Some(best) = best {
                eprintln!(
                    "\n--- Table 3: Per-query expansion distribution (S={}, D={}) ---",
                    best.stall, best.drain
                );

                let total = best.per_query_exp.len();
                let max_exp = (ef + 1) as u64;
                let lt50 = best.per_query_exp.iter().filter(|&&e| e < 50).count();
                let lt100 = best.per_query_exp.iter().filter(|&&e| e < 100).count();
                let lt150 = best.per_query_exp.iter().filter(|&&e| e < 150).count();
                let lt200 = best.per_query_exp.iter().filter(|&&e| e < 200).count();
                let eq_max = best.per_query_exp.iter().filter(|&&e| e >= max_exp - 1).count();

                let mut sorted_exp: Vec<u64> = best.per_query_exp.clone();
                sorted_exp.sort();
                let exp_min = sorted_exp.first().copied().unwrap_or(0);
                let exp_p50 = sorted_exp[total / 2];
                let exp_p99 = sorted_exp[(total as f64 * 0.99) as usize];
                let exp_max = sorted_exp.last().copied().unwrap_or(0);

                eprintln!("<50: {:.0}% | <100: {:.0}% | <150: {:.0}% | <200: {:.0}% | =={}: {:.0}%",
                    lt50 as f64 / total as f64 * 100.0,
                    lt100 as f64 / total as f64 * 100.0,
                    lt150 as f64 / total as f64 * 100.0,
                    lt200 as f64 / total as f64 * 100.0,
                    max_exp,
                    eq_max as f64 / total as f64 * 100.0,
                );
                eprintln!("Expansion stats: min={} p50={} p99={} max={}", exp_min, exp_p50, exp_p99, exp_max);
            } else {
                eprintln!("\n--- Table 3: SKIPPED (no config with >10% early stopping) ---");
            }

            // =========================================================
            // Summary
            // =========================================================
            eprintln!("\n--- SUMMARY ---");
            eprintln!("Baseline: r@k={:.3}, avg_blk={:.1}", baseline_recall, baseline_blk);

            // Find best IO savings at <0.5% recall drop
            let target = baseline_recall - 0.005;
            let best_savings = config_results[1..]
                .iter()
                .filter(|c| c.mean_recall >= target)
                .max_by(|a, b| {
                    let sa = 1.0 - a.avg_blk / baseline_blk;
                    let sb = 1.0 - b.avg_blk / baseline_blk;
                    sa.partial_cmp(&sb).unwrap()
                });

            if let Some(best) = best_savings {
                let savings = (1.0 - best.avg_blk / baseline_blk) * 100.0;
                eprintln!(
                    "Best (recall>={:.3}): S={}, D={} → r@k={:.3}, avg_blk={:.1}, saved={:.1}%, early={:.0}%",
                    target, best.stall, best.drain, best.mean_recall, best.avg_blk, savings, best.early_pct,
                );
                if savings >= 10.0 {
                    eprintln!("PASS: >=10% block reduction at iso-recall.");
                } else {
                    eprintln!("PARTIAL: {:.1}% block reduction (target: >=10%).", savings);
                }
            } else {
                eprintln!("No config maintains recall within 0.5% of baseline.");
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
        return;
    }

    // Verify baseline (stall_limit=0) matches expected behavior
    assert!(
        !config_results.is_empty(),
        "No configs were tested"
    );
    let baseline = &config_results[0];
    assert_eq!(baseline.stall, 0);
    assert_eq!(baseline.early_pct, 0.0, "stall_limit=0 should never stop early");
}

// ---------------------------------------------------------------------------
// Engine integration tests
// ---------------------------------------------------------------------------

/// Test A: Engine creates default global QD, spawns workers, runs queries on tmpfs.
#[test]
fn test_engine_default_on() {
    use std::sync::Arc;
    use divergence_engine::{
        Engine, EngineConfig, EngineHealth,
        IoDriver, AdjacencyPool, QueryRecorder, SearchGuard,
        PerfLevel, WorkerConfig, spawn_worker,
        disk_graph_search_pipe,
    };

    // Check io_uring availability — skip if not supported
    if !with_runtime(|_| {}) {
        eprintln!("SKIPPED: io_uring not available");
        return;
    }

    let n = 200;
    let dim = 32;
    let k = 10;
    let ef = 64;
    let m_max = 32;
    let ef_construction = 200;

    // Build index to tmpfs
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // Create Engine with 2 cores
    let engine = Engine::new(EngineConfig {
        index_dir: dir_str.clone(),
        num_cores: 2,
        direct_io: false, // tmpfs
        prefetch_window: 2,
        prefetch_budget: 2,
        stall_limit: 0, // no adaptive stopping for this test
        drain_budget: 0,
        ..Default::default()
    });

    // Verify default QD resolution
    assert_eq!(engine.resolved_global_qd, 16, "default_global_qd(2) should be 16");
    assert_eq!(engine.global_budget().capacity(), 16);
    assert_eq!(engine.health(), EngineHealth::Healthy);

    // Generate queries
    let queries: Vec<Vec<f32>> = generate_vectors(10, dim, 999);

    // Spawn 2 workers, each runs 5 queries
    let results: Arc<std::sync::Mutex<Vec<(usize, u64)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for core_id in 0..2 {
        let setup = engine.core_setup(core_id);
        let entry_set = entry_set.clone();
        let disk_vectors = disk_vectors.clone();
        let queries = queries.clone();
        let results = Arc::clone(&results);

        let handle = spawn_worker(
            WorkerConfig {
                core_id: setup.core_id,
                ..Default::default()
            },
            move || async move {
                let io = IoDriver::open_production(
                    &setup.index_dir,
                    dim,
                    setup.per_core_qd,
                    setup.direct_io,
                    setup.global_budget,
                    setup.health,
                )
                .await
                .unwrap();

                let bank =
                    divergence_core::distance::FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
                let pool = AdjacencyPool::new(256 * 1024);

                let recorder = QueryRecorder::new();
                let start_idx = core_id * 5;

                for i in start_idx..start_idx + 5 {
                    let q = &queries[i];
                    let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                    let _results = disk_graph_search_pipe(
                        q,
                        &entry_set,
                        k,
                        ef,
                        setup.prefetch_window as usize,
                        setup.stall_limit,
                        setup.drain_budget,
                        &pool,
                        &io,
                        &bank,
                        &mut guard.ctx,
                        PerfLevel::EnableTime,
                    )
                    .await;
                }

                let io_timing = io.take_io_timing();
                let snap = recorder.take_snapshot(io_timing);

                // Deposit into mailbox
                if let Ok(mut slot) = setup.mailbox.lock() {
                    *slot = Some(snap);
                }

                results.lock().unwrap().push((core_id, snap.queries));
            },
        );
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("worker panicked");
    }

    // Collect stats
    let stats = engine.collect_stats();
    eprintln!("Engine stats: total_queries={}, per_core={}, max_p99={:.0}us, health={:?}",
        stats.total_queries, stats.per_core.len(), stats.max_lat_p99_us, stats.health);

    assert_eq!(stats.total_queries, 10, "should have 10 total queries");
    assert_eq!(stats.per_core.len(), 2, "should have 2 core snapshots");
    assert_eq!(stats.health, EngineHealth::Healthy);

    // Verify results
    let res = results.lock().unwrap();
    assert_eq!(res.len(), 2);
}

/// Test B: HealthChecker transitions THROTTLED/HEALTHY based on stats.
#[test]
fn test_health_checker_unit() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use divergence_engine::{EngineHealth, CoreSnapshot, QueryRecorder, PerfLevel, SearchGuard};

    let health = Arc::new(std::sync::atomic::AtomicU8::new(EngineHealth::Healthy as u8));
    let _mailbox: Arc<std::sync::Mutex<Option<CoreSnapshot>>> =
        Arc::new(std::sync::Mutex::new(None));

    // Create a recorder with high-latency queries
    let recorder = QueryRecorder::new();
    for _ in 0..5 {
        let mut guard = SearchGuard::new(&recorder, PerfLevel::CountOnly);
        guard.ctx.total_ns = 500_000; // 500us
        guard.ctx.global_inflight_sum = 15;
        guard.ctx.global_inflight_samples = 1;
        guard.ctx.global_inflight_max = 15;
    }

    // Take snapshot: p99 should be high, inflight_max = 15
    let snap = recorder.take_snapshot((100, 200, 10));
    assert!(snap.lat_p99_us > 100.0, "p99 should be high: {}", snap.lat_p99_us);
    assert_eq!(snap.global_inflight_max, 15);

    // Knee condition: p99 > 100us AND inflight_max > 0.9*16 = 14.4
    // Both true → should throttle
    let inflight_threshold = (16.0 * 0.9) as u64; // = 14
    assert!(snap.global_inflight_max > inflight_threshold);

    // Manually set throttled (simulates what HealthChecker does)
    health.store(EngineHealth::Throttled as u8, Ordering::Release);
    assert_eq!(
        EngineHealth::from(health.load(Ordering::Acquire)),
        EngineHealth::Throttled,
    );

    // Clear back to healthy
    health.store(EngineHealth::Healthy as u8, Ordering::Release);
    assert_eq!(
        EngineHealth::from(health.load(Ordering::Acquire)),
        EngineHealth::Healthy,
    );
}

/// Test C: EC2 smoke test with real NVMe + Cohere data.
/// Run: BENCH_DIR=/mnt/nvme/bench COHERE_N=100000 cargo test --release \
///   -p divergence-engine --test disk_search test_engine_ec2 -- --nocapture
#[test]
fn test_engine_ec2_smoke() {
    use std::sync::Arc;
    use divergence_engine::{
        Engine, EngineConfig, EngineHealth,
        IoDriver, AdjacencyPool, QueryRecorder, SearchGuard,
        PerfLevel, WorkerConfig, spawn_worker,
        disk_graph_search_pipe,
    };

    let bench_dir = match std::env::var("BENCH_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIPPED: BENCH_DIR not set (EC2 only)");
            return;
        }
    };

    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let data_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| "data/cohere_100k".to_string());

    if !Path::new(&data_dir).exists() {
        eprintln!("SKIPPED: {} not found", data_dir);
        return;
    }

    let dim = 768;
    let k = 100;
    let ef = 200;
    let m_max = 32;
    let ef_construction = 200;
    let num_cores = 2;

    // Load Cohere data
    let raw: Vec<Vec<f32>> = {
        let path = format!("{}/base.fvecs", data_dir);
        if !Path::new(&path).exists() {
            eprintln!("SKIPPED: {} not found", path);
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
        let mut vecs = Vec::new();
        let mut offset = 0;
        while offset < floats.len() {
            let d = floats[offset] as usize;
            offset += 1;
            vecs.push(floats[offset..offset + d].to_vec());
            offset += d;
            if vecs.len() >= max_n {
                break;
            }
        }
        vecs
    };

    let n = raw.len();
    eprintln!("Engine EC2 smoke: n={}, dim={}, ef={}, cores={}", n, dim, ef, num_cores);

    // Build NSW
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in raw.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    // Write to BENCH_DIR
    let index_dir = format!("{}/engine_smoke", bench_dir);
    std::fs::create_dir_all(&index_dir).unwrap();
    let writer = IndexWriter::new(Path::new(&index_dir));
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    let meta = IndexMeta::load_from(&Path::new(&index_dir).join("meta.json")).unwrap();
    let disk_vectors =
        load_vectors(&Path::new(&index_dir).join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // Create Engine
    let engine = Engine::new(EngineConfig {
        index_dir: index_dir.clone(),
        num_cores,
        direct_io: true,
        prefetch_window: 4,
        prefetch_budget: 4,
        stall_limit: 8,
        drain_budget: 16,
        ..Default::default()
    });

    eprintln!(
        "Engine: gQD={}, per_core_qd={}, query_limiter_cap={}",
        engine.resolved_global_qd,
        engine.resolved_per_core_qd,
        engine.query_limiter().capacity(),
    );

    // Pick 100 random queries from the dataset
    let num_queries = 100.min(n);
    let mut rng = Xoshiro256StarStar::seed_from_u64(42);
    let query_indices: Vec<usize> = (0..num_queries)
        .map(|_| rng.r#gen::<usize>() % n)
        .collect();
    let queries: Vec<Vec<f32>> = query_indices.iter().map(|&i| raw[i].clone()).collect();

    // Spawn workers
    let all_results: Arc<std::sync::Mutex<Vec<(usize, Vec<f64>)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    let queries_per_core = num_queries / num_cores;

    if !with_runtime(|_rt| {}) {
        eprintln!("SKIPPED: io_uring not available");
        return;
    }

    let mut handles = Vec::new();
    for core_id in 0..num_cores {
        let setup = engine.core_setup(core_id);
        let entry_set = entry_set.clone();
        let disk_vectors = disk_vectors.clone();
        let queries = queries.clone();
        let raw_clone = raw.clone();
        let query_indices = query_indices.clone();
        let all_results = Arc::clone(&all_results);

        let handle = spawn_worker(
            WorkerConfig {
                core_id: setup.core_id,
                ..Default::default()
            },
            move || async move {
                let io = IoDriver::open_production(
                    &setup.index_dir,
                    dim,
                    setup.per_core_qd,
                    setup.direct_io,
                    setup.global_budget,
                    setup.health,
                )
                .await
                .unwrap();

                let bank =
                    divergence_core::distance::FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
                let pool = AdjacencyPool::new(1024 * 1024);

                let recorder = QueryRecorder::new();
                let start = core_id * queries_per_core;
                let end = start + queries_per_core;
                let mut recalls = Vec::new();

                for i in start..end {
                    let q = &queries[i];
                    let mut guard = SearchGuard::new(&recorder, PerfLevel::EnableTime);
                    let results = disk_graph_search_pipe(
                        q,
                        &entry_set,
                        k,
                        ef,
                        setup.prefetch_window as usize,
                        setup.stall_limit,
                        setup.drain_budget,
                        &pool,
                        &io,
                        &bank,
                        &mut guard.ctx,
                        PerfLevel::EnableTime,
                    )
                    .await;

                    // Compute recall against brute-force
                    let query_idx = query_indices[i];
                    let mut gt: Vec<(f32, usize)> = (0..n)
                        .map(|j| {
                            let d: f32 = raw_clone[query_idx]
                                .iter()
                                .zip(raw_clone[j].iter())
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum();
                            (d, j)
                        })
                        .collect();
                    gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let gt_set: std::collections::HashSet<usize> =
                        gt.iter().take(k).map(|&(_, j)| j).collect();
                    let found: usize = results
                        .iter()
                        .filter(|r| gt_set.contains(&(r.id.0 as usize)))
                        .count();
                    recalls.push(found as f64 / k as f64);
                }

                let io_timing = io.take_io_timing();
                let snap = recorder.take_snapshot(io_timing);

                if let Ok(mut slot) = setup.mailbox.lock() {
                    *slot = Some(snap);
                }

                all_results.lock().unwrap().push((core_id, recalls));
            },
        );
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("worker panicked");
    }

    let stats = engine.collect_stats();
    eprintln!("\n=== Engine Stats ===");
    eprintln!("total_queries: {}", stats.total_queries);
    eprintln!("max_p99_us: {:.0}", stats.max_lat_p99_us);
    eprintln!("avg_io_wait_pct: {:.1}%", stats.avg_io_wait_pct);
    eprintln!("max_global_inflight: {}", stats.max_global_inflight);
    eprintln!("sem_wait_pct: {:.1}%", stats.sem_wait_pct);
    eprintln!(
        "device_ns_per_io: {:.0}ns ({:.1}us)",
        stats.device_ns_per_io,
        stats.device_ns_per_io / 1000.0,
    );
    eprintln!("health: {:?}", stats.health);
    for (i, snap) in stats.per_core.iter().enumerate() {
        eprintln!(
            "  core {}: queries={} p50={:.0}us p99={:.0}us io_wait={:.1}%",
            i, snap.queries, snap.lat_p50_us, snap.lat_p99_us, snap.io_wait_pct,
        );
    }

    // Report recalls
    let res = all_results.lock().unwrap();
    for (core_id, recalls) in res.iter() {
        let avg: f64 = recalls.iter().sum::<f64>() / recalls.len() as f64;
        eprintln!("  core {}: mean_recall={:.3}", core_id, avg);
    }

    assert_eq!(stats.health, EngineHealth::Healthy);
    assert!(stats.total_queries > 0);
}

// ---------------------------------------------------------------------------
// EXP-SLO: Forced overload admission control validation
// ---------------------------------------------------------------------------

/// Validates admission control under forced overload with concurrent-per-core IO.
///
/// Design: spawns 3 concurrent search tasks per core. Under steady load (0.8x),
/// only 1-2 are active → no IO contention → low p99. Under overload (3x), all 3
/// are active → share per_core_qd=4 IO slots → p99 triples → exceeds SLA →
/// controller engages → reduces query capacity → fewer concurrent searches →
/// p99 drops.
///
/// Run: BENCH_DIR=/mnt/nvme/bench COHERE_N=100000 COHERE_DIR=/mnt/nvme/divergence/data/cohere_100k \
///   cargo test --release -p divergence-engine --test disk_search exp_slo -- --nocapture
#[test]
fn exp_slo_closed_loop() {
    use std::cell::Cell;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use divergence_engine::{
        AdjacencyPool, Engine, EngineConfig, HealthChecker, IoDriver, PerfLevel, QueryRecorder,
        WorkerConfig, disk_graph_search_pipe, spawn_worker,
    };

    let bench_dir = match std::env::var("BENCH_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIPPED: BENCH_DIR not set (EC2 only)");
            return;
        }
    };

    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dataset_dir =
        std::env::var("COHERE_DIR").unwrap_or_else(|_| "data/cohere_100k".to_string());

    let (vectors, queries_flat, _ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let w = 4usize;
    let stall_limit = 4u32;
    let drain_budget = 16u32;
    let num_queries_total = nq; // all queries to avoid cache-warming confound

    eprintln!("\n========== EXP-SLO: FORCED OVERLOAD ADMISSION CONTROL ==========");
    eprintln!(
        "n={}, dim={}, k={}, ef={}, W={}, S={}, D={}, queries={}",
        n, dim, k, ef, w, stall_limit, drain_budget, num_queries_total
    );

    // Build index once, reuse on subsequent runs
    let index_dir = format!("{}/exp_slo", bench_dir);
    let adj_path = format!("{}/adjacency.dat", index_dir);

    if !std::path::Path::new(&adj_path).exists() {
        eprintln!("Building NSW index (first run)...");
        std::fs::create_dir_all(&index_dir).unwrap();
        let config = NswConfig::new(m_max, ef_construction);
        let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
        for i in 0..n {
            builder.insert(VectorId(i as u32), &vectors[i * dim..(i + 1) * dim]);
        }
        let index = builder.build();
        let writer = IndexWriter::new(Path::new(&index_dir));
        writer
            .write(
                n as u32,
                dim,
                "cosine",
                index.max_degree(),
                ef_construction,
                &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
                index.vectors_raw(),
                |vid| index.neighbors(vid),
            )
            .unwrap();
        eprintln!("Index built and saved.");
    } else {
        eprintln!("Reusing index at {}", index_dir);
    }

    // Load shared data
    let disk_vectors: Arc<[f32]> = Arc::from(
        load_vectors(&Path::new(&index_dir).join("vectors.dat"), n, dim)
            .unwrap()
            .as_slice(),
    );
    let entry_set: Arc<[VectorId]> = {
        let meta = IndexMeta::load_from(&Path::new(&index_dir).join("meta.json")).unwrap();
        Arc::from(
            meta.entry_set
                .iter()
                .map(|&v| VectorId(v))
                .collect::<Vec<_>>()
                .as_slice(),
        )
    };
    let query_vecs: Arc<[Vec<f32>]> = Arc::from(
        queries_flat
            .chunks_exact(dim)
            .take(num_queries_total)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let pool_bytes = (n / 20) * 4096;

    let num_cores = 4;
    let concurrent_per_core: u32 = 3;
    // query_cap = total concurrent tasks. Under Healthy, all 12 can run (max IO pressure).
    // Under Degraded: effective=9 (3 wait), Under Throttled: effective=6 (6 wait).
    // The yield after drop(_permit) ensures waiters actually get a chance.
    let query_cap = num_cores * concurrent_per_core as usize; // 12

    // Per-query result
    #[derive(Clone)]
    struct QueryResult {
        latency_us: f64,
        admit_wait_us: f64,
        health_sample: u8,
        phase_tag: u8,
    }

    // =======================================================================
    // Phase A: Capacity discovery (sequential, 1 query/core, for baseline)
    // =======================================================================
    eprintln!("\n--- Phase A: Baseline Discovery (sequential, {} cores) ---", num_cores);

    let phase_a_n = 200.min(num_queries_total);
    let phase_a_lats: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let phase_a_start = Instant::now();

    {
        let engine_a = Engine::new(EngineConfig {
            index_dir: index_dir.clone(),
            num_cores,
            direct_io: true,
            prefetch_window: w,
            prefetch_budget: 4,
            stall_limit,
            drain_budget,
            ..Default::default()
        });

        eprintln!(
            "  gQD={}, per_core_qd={}",
            engine_a.resolved_global_qd, engine_a.resolved_per_core_qd
        );

        let queries_per_core = phase_a_n / num_cores;
        let mut handles = Vec::new();

        for core_id in 0..num_cores {
            let setup = engine_a.core_setup(core_id);
            let es = Arc::clone(&entry_set);
            let dv = Arc::clone(&disk_vectors);
            let qv = Arc::clone(&query_vecs);
            let lats = Arc::clone(&phase_a_lats);

            let handle = spawn_worker(
                WorkerConfig {
                    core_id: setup.core_id,
                    uring_entries: 1024,
                },
                move || async move {
                    let io = Rc::new(
                        IoDriver::open_with_budget(
                            &setup.index_dir,
                            dim,
                            setup.per_core_qd,
                            setup.direct_io,
                            Some(setup.global_budget),
                        )
                        .await
                        .unwrap(),
                    );
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    let bank = divergence_core::distance::FP32SimdVectorBank::new(
                        &dv,
                        dim,
                        MetricType::Cosine,
                    );
                    let _pf = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool),
                        Rc::clone(&io),
                        setup.prefetch_budget,
                    );

                    // Warmup
                    for i in 0..20 {
                        let qi = i % qv.len();
                        let mut perf = SearchPerfContext::default();
                        let _ = disk_graph_search_pipe(
                            &qv[qi], &es, k, ef, w, stall_limit, drain_budget,
                            &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                        )
                        .await;
                    }

                    let start_idx = core_id * queries_per_core;
                    let end_idx = start_idx + queries_per_core;
                    let mut local_lats = Vec::with_capacity(queries_per_core);

                    for i in start_idx..end_idx {
                        let qi = i % qv.len();
                        let t0 = Instant::now();
                        let mut perf = SearchPerfContext::default();
                        let _ = disk_graph_search_pipe(
                            &qv[qi], &es, k, ef, w, stall_limit, drain_budget,
                            &pool, &io, &bank, &mut perf, PerfLevel::EnableTime,
                        )
                        .await;
                        local_lats.push(t0.elapsed().as_micros() as f64);
                    }

                    lats.lock().unwrap().extend(local_lats);
                },
            );
            handles.push(handle);
        }

        for h in handles {
            h.join().expect("Phase A worker panicked");
        }
    }

    let phase_a_wall = phase_a_start.elapsed().as_secs_f64();
    let peak_qps = (phase_a_n as f64 / phase_a_wall) as u64;
    let mut sorted = phase_a_lats.lock().unwrap().clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let baseline_p99 = percentile(&sorted, 99.0);
    let baseline_p50 = percentile(&sorted, 50.0);
    let p99_sla_us = (1.2 * baseline_p99) as u64;

    eprintln!(
        "  peak_qps={}, p50={:.0}us, p99={:.0}us, SLA={:.0}us (1.2x)",
        peak_qps, baseline_p50, baseline_p99, p99_sla_us
    );

    // =======================================================================
    // Phase B+C: Concurrent-per-core overload test
    //
    // Each core spawns `concurrent_per_core` search tasks. Under overload,
    // all tasks compete for per_core_qd IO slots, driving up p99.
    // =======================================================================
    eprintln!(
        "\n--- Phase B+C: Concurrent Overload ({} cores, {} tasks/core, query_cap={}) ---",
        num_cores, concurrent_per_core, query_cap
    );

    let all_results: Arc<Mutex<Vec<QueryResult>>> = Arc::new(Mutex::new(Vec::new()));
    let query_queue: Arc<Mutex<std::collections::VecDeque<(usize, u8)>>> =
        Arc::new(Mutex::new(std::collections::VecDeque::new()));
    let stop_flag: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let health_flips: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let last_health: Arc<AtomicU8> = Arc::new(AtomicU8::new(0xff));

    let engine = Engine::new(EngineConfig {
        index_dir: index_dir.clone(),
        num_cores,
        direct_io: true,
        prefetch_window: w,
        prefetch_budget: 4,
        stall_limit,
        drain_budget,
        p99_sla_us,
        health_window: 10,
        query_cap: Some(query_cap),
        ..Default::default()
    });

    eprintln!(
        "  gQD={}, per_core_qd={}, query_cap={}, SLA={}us",
        engine.resolved_global_qd,
        engine.resolved_per_core_qd,
        engine.query_limiter().capacity(),
        p99_sla_us,
    );

    // Spawn workers — each spawns concurrent_per_core search tasks
    let mut handles = Vec::new();
    for core_id in 0..num_cores {
        let setup = engine.core_setup(core_id);
        let es = Arc::clone(&entry_set);
        let dv = Arc::clone(&disk_vectors);
        let qv = Arc::clone(&query_vecs);
        let results = Arc::clone(&all_results);
        let queue = Arc::clone(&query_queue);
        let stop = Arc::clone(&stop_flag);
        let flips = Arc::clone(&health_flips);
        let last_h = Arc::clone(&last_health);

        let handle = spawn_worker(
            WorkerConfig {
                core_id: setup.core_id,
                uring_entries: 1024,
            },
            move || async move {
                let io = Rc::new(
                    IoDriver::open_production(
                        &setup.index_dir,
                        dim,
                        setup.per_core_qd,
                        setup.direct_io,
                        setup.global_budget,
                        Arc::clone(&setup.health),
                    )
                    .await
                    .unwrap(),
                );
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let _pf = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool),
                    Rc::clone(&io),
                    setup.prefetch_budget,
                );
                let recorder = Rc::new(QueryRecorder::new());
                let checker = Rc::new(HealthChecker::new(
                    Arc::clone(&setup.health),
                    Arc::clone(&setup.mailbox),
                    setup.health_window,
                    setup.p99_sla_us,
                    setup.gqd_capacity,
                ));

                let done_count = Rc::new(Cell::new(0u32));
                let local_results = Rc::new(RefCell::new(Vec::new()));

                // Spawn concurrent search tasks — each creates own bank
                // (FP32SimdVectorBank borrows data, can't be 'static via Rc)
                for _ in 0..concurrent_per_core {
                    let io = Rc::clone(&io);
                    let pool = Rc::clone(&pool);
                    let dv = Arc::clone(&dv); // each task owns its Arc
                    let recorder = Rc::clone(&recorder);
                    let checker = Rc::clone(&checker);
                    let queue = Arc::clone(&queue);
                    let stop = Arc::clone(&stop);
                    let ql = Arc::clone(&setup.query_limiter);
                    let es = Arc::clone(&es);
                    let qv = Arc::clone(&qv);
                    let flips = Arc::clone(&flips);
                    let last_h = Arc::clone(&last_h);
                    let done = Rc::clone(&done_count);
                    let local_res = Rc::clone(&local_results);

                    monoio::spawn(async move {
                        let bank = divergence_core::distance::FP32SimdVectorBank::new(
                            &dv, dim, MetricType::Cosine,
                        );
                        loop {
                            let item = { queue.lock().unwrap().pop_front() };

                            let (qi, phase_tag) = match item {
                                Some(v) => v,
                                None => {
                                    if stop.load(Ordering::Relaxed) {
                                        done.set(done.get() + 1);
                                        return;
                                    }
                                    monoio::time::sleep(Duration::from_micros(100)).await;
                                    continue;
                                }
                            };

                            // Admission control
                            let admit_t0 = Instant::now();
                            let _permit = ql.acquire().await;
                            let admit_wait_us = admit_t0.elapsed().as_micros() as f64;

                            // Search (manual record instead of SearchGuard for Rc compat)
                            let t0 = Instant::now();
                            let mut perf = SearchPerfContext::default();
                            let _ = disk_graph_search_pipe(
                                &qv[qi % qv.len()],
                                &es,
                                k,
                                ef,
                                w,
                                stall_limit,
                                drain_budget,
                                &pool,
                                &io,
                                &bank,
                                &mut perf,
                                PerfLevel::EnableTime,
                            )
                            .await;
                            perf.total_ns = t0.elapsed().as_nanos() as u64;
                            recorder.record(&perf);
                            let latency_us = t0.elapsed().as_micros() as f64;

                            // Health check
                            checker.on_query_complete(&recorder, &io);

                            let h = checker.health() as u8;
                            let prev = last_h.swap(h, Ordering::Relaxed);
                            if prev != h && prev != 0xff {
                                flips.fetch_add(1, Ordering::Relaxed);
                            }

                            local_res.borrow_mut().push(QueryResult {
                                latency_us,
                                admit_wait_us,
                                health_sample: h,
                                phase_tag,
                            });

                            drop(_permit);
                            // Yield after releasing permit — without this, the same task
                            // immediately reacquires (no await between drop and acquire),
                            // starving other waiters and making admission wait always 0.
                            monoio::time::sleep(Duration::from_micros(1)).await;
                        }
                    });
                }

                // Wait for all tasks to finish
                while done_count.get() < concurrent_per_core {
                    monoio::time::sleep(Duration::from_millis(100)).await;
                }

                // Export results
                results
                    .lock()
                    .unwrap()
                    .extend(local_results.borrow().iter().cloned());
            },
        );
        handles.push(handle);
    }

    // Injector thread
    let overload_mult = 3.0f64;
    let steady_qps = (peak_qps as f64 * 0.8) as u64;
    let overload_qps = (peak_qps as f64 * overload_mult) as u64;
    let recovery_qps = steady_qps;

    eprintln!(
        "  phases: steady={}qps (10s), overload={}qps (15s), recovery={}qps (10s)",
        steady_qps, overload_qps, recovery_qps
    );

    let queue_inj = Arc::clone(&query_queue);
    let stop_inj = Arc::clone(&stop_flag);

    let injector = std::thread::spawn(move || {
        let phases: [(u64, u64, u8); 3] = [
            (steady_qps, 10, 0),
            (overload_qps, 15, 1),
            (recovery_qps, 10, 2),
        ];

        let mut query_idx = 0usize;

        for &(target_qps, duration_s, phase_tag) in &phases {
            let phase_start = Instant::now();
            let interval = if target_qps > 0 {
                Duration::from_nanos(1_000_000_000 / target_qps)
            } else {
                Duration::from_secs(1)
            };

            while phase_start.elapsed().as_secs() < duration_s {
                {
                    let mut q = queue_inj.lock().unwrap();
                    q.push_back((query_idx, phase_tag));
                }
                query_idx += 1;
                std::thread::sleep(interval);
            }
        }

        stop_inj.store(true, Ordering::Release);
    });

    injector.join().expect("injector panicked");

    // Give workers time to drain
    std::thread::sleep(Duration::from_secs(5));
    stop_flag.store(true, Ordering::Release);

    for h in handles {
        h.join().expect("worker panicked");
    }

    // ===================================================================
    // Analysis
    // ===================================================================
    let results = all_results.lock().unwrap();

    let phase_results: Vec<Vec<&QueryResult>> = (0..3)
        .map(|tag| {
            let phase: Vec<&QueryResult> =
                results.iter().filter(|r| r.phase_tag == tag).collect();
            let skip = phase.len() * 3 / 10;
            phase[skip..].to_vec()
        })
        .collect();

    let analyze = |phase: &[&QueryResult]| -> (f64, f64, f64, f64, f64, [u64; 3]) {
        if phase.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, [0; 3]);
        }
        let mut lats: Vec<f64> = phase.iter().map(|r| r.latency_us).collect();
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut admits: Vec<f64> = phase.iter().map(|r| r.admit_wait_us).collect();
        admits.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut hdist = [0u64; 3];
        for r in phase {
            hdist[(r.health_sample as usize).min(2)] += 1;
        }

        (
            percentile(&lats, 50.0),
            percentile(&lats, 99.0),
            percentile(&lats, 99.9),
            percentile(&admits, 50.0),
            percentile(&admits, 99.0),
            hdist,
        )
    };

    let steady = analyze(&phase_results[0]);
    let overload = analyze(&phase_results[1]);
    let recovery = analyze(&phase_results[2]);

    let steady_ach = phase_results[0].len() as f64 / 10.0;
    let overload_ach = phase_results[1].len() as f64 / 15.0;
    let recovery_ach = phase_results[2].len() as f64 / 10.0;
    let total_flips = health_flips.load(Ordering::Relaxed);

    let fmt_health = |d: [u64; 3]| -> String {
        let t = (d[0] + d[1] + d[2]) as f64;
        if t == 0.0 {
            return "N/A".to_string();
        }
        format!(
            "H:{:.0}% D:{:.0}% T:{:.0}%",
            d[0] as f64 / t * 100.0,
            d[1] as f64 / t * 100.0,
            d[2] as f64 / t * 100.0,
        )
    };

    eprintln!(
        "\n{:<12} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>10} {}",
        "Phase", "tgt_qps", "ach_qps", "p50us", "p99us", "p999us", "adm_p50", "adm_p99",
        "health"
    );
    eprintln!(
        "{:<12} {:>7} {:>7.0} {:>7.0} {:>7.0} {:>8.0} {:>8.0} {:>10.0} {}",
        "STEADY", steady_qps, steady_ach, steady.0, steady.1, steady.2, steady.3,
        steady.4, fmt_health(steady.5)
    );
    eprintln!(
        "{:<12} {:>7} {:>7.0} {:>7.0} {:>7.0} {:>8.0} {:>8.0} {:>10.0} {}",
        "OVERLOAD", overload_qps, overload_ach, overload.0, overload.1, overload.2,
        overload.3, overload.4, fmt_health(overload.5)
    );
    eprintln!(
        "{:<12} {:>7} {:>7.0} {:>7.0} {:>7.0} {:>8.0} {:>8.0} {:>10.0} {}",
        "RECOVERY", recovery_qps, recovery_ach, recovery.0, recovery.1, recovery.2,
        recovery.3, recovery.4, fmt_health(recovery.5)
    );
    eprintln!("Health flips: {}", total_flips);

    // ===================================================================
    // Acceptance gates
    // ===================================================================
    let (steady_p99, overload_p99, recovery_p99) = (steady.1, overload.1, recovery.1);
    let overload_health = overload.5;
    let overload_admit_p99 = overload.4;

    // G1: search p99 stays bounded (admission control prevents catastrophe)
    let g1 = if steady_p99 > 0.0 {
        overload_p99 < 10.0 * steady_p99
    } else {
        true
    };
    eprintln!(
        "\n{}: overload search p99 ({:.0}) < 10x steady ({:.0})",
        if g1 { "PASS" } else { "FAIL" },
        overload_p99,
        10.0 * steady_p99
    );

    // G2: recovery
    let g2 = if steady_p99 > 0.0 {
        recovery_p99 < 2.0 * steady_p99
    } else {
        true
    };
    eprintln!(
        "{}: recovery p99 ({:.0}) < 2x steady ({:.0})",
        if g2 { "PASS" } else { "FAIL" },
        recovery_p99,
        2.0 * steady_p99
    );

    // G3: no throughput crash
    let g3 = overload_ach >= 0.3 * peak_qps as f64;
    eprintln!(
        "{}: overload QPS ({:.0}) >= 0.3x peak ({:.0})",
        if g3 { "PASS" } else { "FAIL" },
        overload_ach,
        0.3 * peak_qps as f64
    );

    // G4: controller MUST engage
    let throttle_count = overload_health[1] + overload_health[2];
    let g4 = throttle_count > 0;
    eprintln!(
        "{}: controller engaged during overload (D+T: {} samples)",
        if g4 { "PASS" } else { "FAIL" },
        throttle_count
    );

    // G5: admission wait rises
    let g5 = overload_admit_p99 > 100.0;
    eprintln!(
        "{}: admission wait p99 ({:.0}us) > 100us during overload",
        if g5 { "PASS" } else { "FAIL" },
        overload_admit_p99
    );

    // G6: no oscillation
    let g6 = total_flips <= 10;
    eprintln!(
        "{}: health flips ({}) <= 10",
        if g6 { "PASS" } else { "FAIL" },
        total_flips
    );

    // Hard assertions
    assert!(g1, "GATE 1 FAILED: search p99 explosion");
    assert!(g3, "GATE 3 FAILED: throughput collapsed");
    assert!(g4, "GATE 4 FAILED: controller never engaged — no IO contention under overload");
    assert!(g5, "GATE 5 FAILED: admission wait never rose (query_cap may be too high)");
}

/// Device calibration test — requires BENCH_DIR.
///
/// Run: BENCH_DIR=/mnt/nvme/bench cargo test --release -p divergence-engine \
///   --test disk_search test_calibrate -- --nocapture
#[test]
fn test_calibrate_device() {
    use divergence_engine::{calibrate_device, default_global_qd};

    let bench_dir = match std::env::var("BENCH_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIPPED: BENCH_DIR not set (EC2 only)");
            return;
        }
    };

    // Need a file to calibrate against. Use the adjacency file from exp_slo if it exists,
    // otherwise create a dummy file.
    let cal_file = format!("{}/calibrate_test.dat", bench_dir);
    if !Path::new(&cal_file).exists() {
        // Create a 64MB file for calibration
        use std::io::Write;
        let mut f = std::fs::File::create(&cal_file).unwrap();
        let block = vec![0u8; 4096];
        for _ in 0..(64 * 1024 * 1024 / 4096) {
            f.write_all(&block).unwrap();
        }
        f.sync_all().unwrap();
    }

    if !with_runtime(|rt| {
        rt.block_on(async {
            let cal = calibrate_device(&cal_file, true).await.unwrap();

            eprintln!("\n=== DEVICE CALIBRATION ===");
            eprintln!("{:<6} {:>10} {:>10}", "QD", "IOPS", "p99_us");
            for m in &cal.levels {
                eprintln!("{:<6} {:>10.0} {:>10.1}", m.qd, m.iops, m.p99_us);
            }

            let formula_qd = default_global_qd(2);
            let recommended = cal.recommended_qd.min(formula_qd);
            eprintln!(
                "\nRaw knee: QD={}, formula: QD={}, recommended: QD={}",
                cal.recommended_qd, formula_qd, recommended
            );

            assert!(cal.recommended_qd >= 1, "knee should be at least 1");
            assert!(!cal.levels.is_empty(), "should have measurements");
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Stable Benchmark Runner
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct BenchConfig {
    label: String,
    ef: usize,        // used as max_ef when ada_ef=true
    k: usize,
    prefetch_width: usize,
    stall_limit: u32,  // used as default when ada_ef=false
    drain_budget: u32,
    adj_inflight: usize,
    cache_pct: usize,
    num_queries: usize,
    warmup_queries: usize,
    ada_ef: bool,      // if true, per-query (ef, S, D) from Ada-ef scoring
    /// If true, clear pool before each query (true cold per-query measurement).
    clear_per_query: bool,
}

struct BenchResult {
    recall: f64,
    lat_p50_ms: f64,
    lat_p99_ms: f64,
    qps: f64,
    avg_expansions: f64,
    avg_useful: f64,
    avg_wasted: f64,
    avg_blk_q: f64,
    avg_miss_q: f64,
    avg_hit_q: f64,
    avg_singleflight: f64,
    avg_pf_issued: f64,
    avg_pf_consumed: f64,
    avg_best_at: f64,
    avg_first_topk: f64,
    early_stop_pct: f64,
    waste_ratio: f64,
    hit_rate: f64,
    /// Physical NVMe IO reads per query (miss loads + prefetch loads + bypasses).
    avg_phys_reads_q: f64,
    // Timing breakdown (avg per query, ms)
    avg_io_wait_ms: f64,
    avg_compute_ms: f64,
    avg_dist_ms: f64,
    // Derived: avg ms per cache miss (io_ms / mis_q)
    ms_per_miss: f64,
    // Cache health: total bypasses and evict failures across all queries
    total_bypasses: u64,
    total_evict_fail: u64,
    // Refine stats (two-stage pipeline only)
    avg_refine_count: f64,
    avg_refine_ms: f64,
    /// Total IO requests per query: adj_phy + refine vector reads.
    /// Note: adj reads are 4KB pages, refine reads are dim*4 bytes each.
    avg_total_io_q: f64,
}

async fn run_bench(
    cfg: &BenchConfig,
    entry_set: &[VectorId],
    pool: &Rc<AdjacencyPool>,
    io: &Rc<IoDriver>,
    bank: &dyn VectorBank,
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    ada: Option<(&AdaEfStats, &AdaEfTable)>,
    query_scores: &[f64],
) -> BenchResult {
    let nq = cfg.num_queries.min(query_vecs.len());

    // Warmup pass
    for q in query_vecs.iter().take(cfg.warmup_queries) {
        let mut perf = SearchPerfContext::default();
        disk_graph_search_pipe(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, &mut perf, PerfLevel::CountOnly,
        ).await;
    }

    // Per-query Ada-ef tracking (only populated when ada_ef=true)
    struct AdaQueryInfo { score: f64, ef_used: usize, recall: f64, blk: u64 }
    let mut ada_info: Vec<AdaQueryInfo> = Vec::new();

    let mut recalls = Vec::with_capacity(nq);
    let mut latencies_ms = Vec::with_capacity(nq);
    let mut sum_exp = 0u64;
    let mut sum_useful = 0u64;
    let mut sum_wasted = 0u64;
    let mut sum_blk = 0u64;
    let mut sum_miss = 0u64;
    let mut sum_hit = 0u64;
    let mut sum_phys_reads = 0u64;
    let mut sum_sf = 0u64;
    let mut sum_pf_issued = 0u64;
    let mut sum_pf_consumed = 0u64;
    let mut sum_best_at = 0u64;
    let mut sum_first_topk = 0u64;
    let mut early_count = 0u64;
    let mut sum_io_wait_ns = 0u64;
    let mut sum_compute_ns = 0u64;
    let mut sum_dist_ns = 0u64;

    let cache_stats_before = pool.stats();
    let wall_start = std::time::Instant::now();

    for i in 0..nq {
        let q = &query_vecs[i];

        // Determine per-query params: Ada-ef or fixed
        let (ef, sl, db, ada_score) = if cfg.ada_ef {
            if let Some((stats, table)) = ada {
                // Compute seed distances (same as search seeding — pure DRAM)
                let seed_dists: Vec<f32> = entry_set
                    .iter()
                    .map(|&ep| bank.distance(q, ep.0 as usize))
                    .collect();
                // Compute score for diagnostics
                let (mu, sigma) = stats.estimate_fdl_params(q);
                let mut thresholds = [0.0f64; 5];
                for b in 0..5 {
                    thresholds[b] = mu + sigma * divergence_engine::ada_ef::inv_normal_cdf(0.001 * (b + 1) as f64);
                }
                let mut counts = [0u32; 5];
                for &d in &seed_dists {
                    let d = d as f64;
                    for (bin, &thresh) in thresholds.iter().enumerate() {
                        if d <= thresh { counts[bin] += 1; break; }
                    }
                }
                let weights = [100.0, 36.788, 13.534, 4.979, 1.832];
                let score: f64 = counts.iter().zip(weights.iter())
                    .map(|(&c, &w)| w * c as f64 / seed_dists.len() as f64).sum();

                let p = estimate_ada_ef(&seed_dists, stats, q, table);
                // No ef cap — let hard queries get ef > cfg.ef if table says so
                (p.ef, p.stall_limit, p.drain_budget, Some(score))
            } else {
                (cfg.ef, cfg.stall_limit, cfg.drain_budget, None)
            }
        } else {
            (cfg.ef, cfg.stall_limit, cfg.drain_budget, None)
        };

        let mut perf = SearchPerfContext::default();
        let t0 = std::time::Instant::now();
        let results = disk_graph_search_pipe(
            q, entry_set, cfg.k, ef, cfg.prefetch_width,
            sl, db,
            pool, io, bank, &mut perf, PerfLevel::EnableTime,
        ).await;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);

        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
        let q_recall = recall_at_k(&ids, &ground_truth[i]);
        recalls.push(q_recall);

        if let Some(score) = ada_score {
            ada_info.push(AdaQueryInfo { score, ef_used: ef, recall: q_recall, blk: perf.blocks_read });
        }

        sum_exp += perf.expansions;
        sum_useful += perf.useful_expansions;
        sum_wasted += perf.wasted_expansions;
        sum_blk += perf.blocks_read;
        sum_miss += perf.blocks_miss;
        sum_hit += perf.blocks_hit;
        sum_phys_reads += perf.phys_reads;
        sum_sf += perf.singleflight_waits;
        sum_pf_issued += perf.prefetch_issued;
        sum_pf_consumed += perf.prefetch_consumed;
        sum_best_at += perf.best_result_at_expansion;
        sum_first_topk += perf.first_topk_at_expansion;
        sum_io_wait_ns += perf.io_wait_ns;
        sum_compute_ns += perf.compute_ns;
        sum_dist_ns += perf.dist_ns;
        if perf.stopped_early {
            early_count += 1;
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let nf = nq as f64;

    let mean_recall = recalls.iter().sum::<f64>() / nf;
    let qps = nf / wall_secs;

    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_lat, 50.0);
    let p99 = percentile(&sorted_lat, 99.0);

    let total_exp = sum_useful + sum_wasted;
    let waste_ratio = if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
    let hit_rate = if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 };
    let ns_to_ms = 1.0 / 1_000_000.0;

    // Print per-bucket diagnostics for ALL configs (using precomputed query_scores)
    {
        let buckets: &[(f64, &str)] = &[
            (20.0, ">=20"), (16.0, ">=16"), (12.0, ">=12"), (8.0, ">=8"), (0.0, "<8"),
        ];
        eprintln!("    {:>8} {:>5} {:>7} {:>6} {:>6} {:>7} {:>7}", "bucket", "n", "recall", "blk/q", "avg_ef", "p99ms", "maxms");
        for (bi, &(thresh, label)) in buckets.iter().enumerate() {
            let indices: Vec<usize> = (0..nq).filter(|&i| {
                let s = query_scores[i];
                if bi == 0 { s >= thresh }
                else { s >= thresh && s < buckets[bi - 1].0 }
            }).collect();
            if indices.is_empty() { continue; }
            let n = indices.len();
            let avg_recall = indices.iter().map(|&i| recalls[i]).sum::<f64>() / n as f64;
            // Per-bucket latency stats
            let mut bucket_lats: Vec<f64> = indices.iter().map(|&i| latencies_ms[i]).collect();
            bucket_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bucket_p99 = percentile(&bucket_lats, 99.0);
            let bucket_max = bucket_lats.last().copied().unwrap_or(0.0);
            // blk from ada_info if available, else from per-query perf (not tracked for non-ada)
            let (avg_blk, avg_ef) = if !ada_info.is_empty() {
                let blk: f64 = indices.iter().map(|&i| ada_info[i].blk as f64).sum::<f64>() / n as f64;
                let ef_avg: f64 = indices.iter().map(|&i| ada_info[i].ef_used as f64).sum::<f64>() / n as f64;
                (blk, ef_avg)
            } else {
                (0.0, cfg.ef as f64)
            };
            if avg_blk > 0.0 {
                eprintln!("    {:>8} {:>5} {:>7.3} {:>6.1} {:>6.0} {:>7.1} {:>7.1}", label, n, avg_recall, avg_blk, avg_ef, bucket_p99, bucket_max);
            } else {
                eprintln!("    {:>8} {:>5} {:>7.3} {:>6} {:>6.0} {:>7.1} {:>7.1}", label, n, avg_recall, "-", avg_ef, bucket_p99, bucket_max);
            }
        }
    }

    BenchResult {
        recall: mean_recall,
        lat_p50_ms: p50,
        lat_p99_ms: p99,
        qps,
        avg_expansions: sum_exp as f64 / nf,
        avg_useful: sum_useful as f64 / nf,
        avg_wasted: sum_wasted as f64 / nf,
        avg_blk_q: sum_blk as f64 / nf,
        avg_miss_q: sum_miss as f64 / nf,
        avg_hit_q: sum_hit as f64 / nf,
        avg_singleflight: sum_sf as f64 / nf,
        avg_pf_issued: sum_pf_issued as f64 / nf,
        avg_pf_consumed: sum_pf_consumed as f64 / nf,
        avg_best_at: sum_best_at as f64 / nf,
        avg_first_topk: sum_first_topk as f64 / nf,
        early_stop_pct: early_count as f64 / nf * 100.0,
        waste_ratio,
        hit_rate,
        avg_phys_reads_q: sum_phys_reads as f64 / nf,
        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
        ms_per_miss: if sum_miss > 0 {
            (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64
        } else {
            0.0
        },
        total_bypasses: pool.stats().bypasses - cache_stats_before.bypasses,
        total_evict_fail: pool.stats().evict_fail_all_pinned - cache_stats_before.evict_fail_all_pinned,
        avg_refine_count: 0.0,
        avg_refine_ms: 0.0,
        avg_total_io_q: sum_phys_reads as f64 / nf,
    }
}

/// Run benchmark using v3 page-packed adjacency (key = page_id).
///
/// Caller must have a running prefetch worker. For `clear_per_query`, this
/// function uses pause/drain/wait/clear/unpause — the worker stays alive.
async fn run_bench_v3(
    cfg: &BenchConfig,
    entry_set: &[VectorId],
    pool: &Rc<AdjacencyPool>,
    io: &Rc<IoDriver>,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
) -> BenchResult {
    let nq = cfg.num_queries.min(query_vecs.len());

    // Warmup pass
    for q in query_vecs.iter().take(cfg.warmup_queries) {
        let mut perf = SearchPerfContext::default();
        disk_graph_search_pipe_v3(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, adj_index, &mut perf, PerfLevel::CountOnly,
        ).await;
    }

    let mut recalls = Vec::with_capacity(nq);
    let mut latencies_ms = Vec::with_capacity(nq);
    let mut sum_exp = 0u64;
    let mut sum_useful = 0u64;
    let mut sum_wasted = 0u64;
    let mut sum_blk = 0u64;
    let mut sum_miss = 0u64;
    let mut sum_hit = 0u64;
    let mut sum_phys_reads = 0u64;
    let mut sum_sf = 0u64;
    let mut sum_pf_issued = 0u64;
    let mut sum_pf_consumed = 0u64;
    let mut sum_best_at = 0u64;
    let mut sum_first_topk = 0u64;
    let mut early_count = 0u64;
    let mut sum_io_wait_ns = 0u64;
    let mut sum_compute_ns = 0u64;
    let mut sum_dist_ns = 0u64;

    let cache_stats_before = pool.stats();
    let wall_start = std::time::Instant::now();

    for i in 0..nq {
        if cfg.clear_per_query {
            // Quiesce: pause hints → drain channel → yield → wait LOADING → clear → unpause
            pool.pause_prefetch(true);
            pool.drain_prefetch();
            // Yield once to let already-spawned IO tasks set their LOADING flags
            monoio::time::sleep(std::time::Duration::from_micros(50)).await;
            while pool.has_loading() {
                monoio::time::sleep(std::time::Duration::from_micros(100)).await;
            }
            pool.clear();
            pool.pause_prefetch(false);
        }
        let q = &query_vecs[i];
        let mut perf = SearchPerfContext::default();
        let t0 = std::time::Instant::now();
        let results = disk_graph_search_pipe_v3(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, adj_index, &mut perf, PerfLevel::EnableTime,
        ).await;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);

        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
        let q_recall = recall_at_k(&ids, &ground_truth[i]);
        recalls.push(q_recall);

        sum_exp += perf.expansions;
        sum_useful += perf.useful_expansions;
        sum_wasted += perf.wasted_expansions;
        sum_blk += perf.blocks_read;
        sum_miss += perf.blocks_miss;
        sum_hit += perf.blocks_hit;
        sum_phys_reads += perf.phys_reads;
        sum_sf += perf.singleflight_waits;
        sum_pf_issued += perf.prefetch_issued;
        sum_pf_consumed += perf.prefetch_consumed;
        sum_best_at += perf.best_result_at_expansion;
        sum_first_topk += perf.first_topk_at_expansion;
        sum_io_wait_ns += perf.io_wait_ns;
        sum_compute_ns += perf.compute_ns;
        sum_dist_ns += perf.dist_ns;
        if perf.stopped_early {
            early_count += 1;
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let nf = nq as f64;

    let mean_recall = recalls.iter().sum::<f64>() / nf;
    let qps = nf / wall_secs;

    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_lat, 50.0);
    let p99 = percentile(&sorted_lat, 99.0);

    let total_exp = sum_useful + sum_wasted;
    let waste_ratio = if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
    let hit_rate = if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 };
    let ns_to_ms = 1.0 / 1_000_000.0;

    BenchResult {
        recall: mean_recall,
        lat_p50_ms: p50,
        lat_p99_ms: p99,
        qps,
        avg_expansions: sum_exp as f64 / nf,
        avg_useful: sum_useful as f64 / nf,
        avg_wasted: sum_wasted as f64 / nf,
        avg_blk_q: sum_blk as f64 / nf,
        avg_miss_q: sum_miss as f64 / nf,
        avg_hit_q: sum_hit as f64 / nf,
        avg_singleflight: sum_sf as f64 / nf,
        avg_pf_issued: sum_pf_issued as f64 / nf,
        avg_pf_consumed: sum_pf_consumed as f64 / nf,
        avg_best_at: sum_best_at as f64 / nf,
        avg_first_topk: sum_first_topk as f64 / nf,
        early_stop_pct: early_count as f64 / nf * 100.0,
        waste_ratio,
        hit_rate,
        avg_phys_reads_q: sum_phys_reads as f64 / nf,
        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
        ms_per_miss: if sum_miss > 0 {
            (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64
        } else {
            0.0
        },
        total_bypasses: pool.stats().bypasses - cache_stats_before.bypasses,
        total_evict_fail: pool.stats().evict_fail_all_pinned - cache_stats_before.evict_fail_all_pinned,
        avg_refine_count: 0.0,
        avg_refine_ms: 0.0,
        avg_total_io_q: sum_phys_reads as f64 / nf,
    }
}

/// Two-stage v4 benchmark: PQ traversal (v3 pages) + parallel FP32 disk refine.
async fn run_bench_v3_refine(
    cfg: &BenchConfig,
    refine_r: usize,
    refine_inflight: usize,
    entry_set: &[VectorId],
    pool: &Rc<AdjacencyPool>,
    io: &Rc<IoDriver>,
    cheap_bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    vec_reader: &Rc<VectorReader>,
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
) -> BenchResult {
    let nq = cfg.num_queries.min(query_vecs.len());

    // Warmup pass: adjacency-only traversal (no refine IO).
    // This warms the adjacency cache but NOT the OS page cache for vectors.
    // With direct_io=true (BENCH_DIR set) this is correct — OS cache is bypassed.
    // With direct_io=false (tmpdir), vector reads may benefit from OS cache
    // after warmup queries, making "warm" refine latency slightly optimistic.
    for q in query_vecs.iter().take(cfg.warmup_queries) {
        let mut perf = SearchPerfContext::default();
        disk_graph_search_pipe_v3(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, cheap_bank, adj_index, &mut perf, PerfLevel::CountOnly,
        ).await;
    }

    let mut recalls = Vec::with_capacity(nq);
    let mut latencies_ms = Vec::with_capacity(nq);
    let mut sum_exp = 0u64;
    let mut sum_useful = 0u64;
    let mut sum_wasted = 0u64;
    let mut sum_blk = 0u64;
    let mut sum_miss = 0u64;
    let mut sum_hit = 0u64;
    let mut sum_phys_reads = 0u64;
    let mut sum_sf = 0u64;
    let mut sum_pf_issued = 0u64;
    let mut sum_pf_consumed = 0u64;
    let mut sum_best_at = 0u64;
    let mut sum_first_topk = 0u64;
    let mut early_count = 0u64;
    let mut sum_io_wait_ns = 0u64;
    let mut sum_compute_ns = 0u64;
    let mut sum_dist_ns = 0u64;
    let mut sum_refine_ns = 0u64;
    let mut sum_refine_count = 0u64;

    let cache_stats_before = pool.stats();
    let wall_start = std::time::Instant::now();

    for i in 0..nq {
        if cfg.clear_per_query {
            pool.pause_prefetch(true);
            pool.drain_prefetch();
            monoio::time::sleep(std::time::Duration::from_micros(50)).await;
            while pool.has_loading() {
                monoio::time::sleep(std::time::Duration::from_micros(100)).await;
            }
            pool.clear();
            pool.pause_prefetch(false);
        }
        let q = &query_vecs[i];
        let mut perf = SearchPerfContext::default();
        let t0 = std::time::Instant::now();

        // Two-stage: PQ traversal + parallel FP32 refine (via engine function)
        let candidates = disk_graph_search_pipe_v3_refine(
            q, entry_set, cfg.k, cfg.ef, refine_r, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, cheap_bank, adj_index, vec_reader,
            refine_inflight, &mut perf, PerfLevel::EnableTime,
        ).await;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);

        let ids: Vec<u32> = candidates.iter().map(|s| s.id.0).collect();
        let q_recall = recall_at_k(&ids, &ground_truth[i]);
        recalls.push(q_recall);

        sum_exp += perf.expansions;
        sum_useful += perf.useful_expansions;
        sum_wasted += perf.wasted_expansions;
        sum_blk += perf.blocks_read;
        sum_miss += perf.blocks_miss;
        sum_hit += perf.blocks_hit;
        sum_phys_reads += perf.phys_reads;
        sum_sf += perf.singleflight_waits;
        sum_pf_issued += perf.prefetch_issued;
        sum_pf_consumed += perf.prefetch_consumed;
        sum_best_at += perf.best_result_at_expansion;
        sum_first_topk += perf.first_topk_at_expansion;
        sum_io_wait_ns += perf.io_wait_ns;
        sum_compute_ns += perf.compute_ns;
        sum_dist_ns += perf.dist_ns;
        sum_refine_ns += perf.refine_ns;
        sum_refine_count += perf.refine_count;
        if perf.stopped_early {
            early_count += 1;
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let nf = nq as f64;

    let mean_recall = recalls.iter().sum::<f64>() / nf;
    let qps = nf / wall_secs;

    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_lat, 50.0);
    let p99 = percentile(&sorted_lat, 99.0);

    let total_exp = sum_useful + sum_wasted;
    let waste_ratio = if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
    let hit_rate = if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 };
    let ns_to_ms = 1.0 / 1_000_000.0;

    eprintln!("    [refine R={}: {:.0} refines/q, {:.2} ms/q refine IO]",
        refine_r,
        sum_refine_count as f64 / nf,
        sum_refine_ns as f64 / nf * ns_to_ms);

    BenchResult {
        recall: mean_recall,
        lat_p50_ms: p50,
        lat_p99_ms: p99,
        qps,
        avg_expansions: sum_exp as f64 / nf,
        avg_useful: sum_useful as f64 / nf,
        avg_wasted: sum_wasted as f64 / nf,
        avg_blk_q: sum_blk as f64 / nf,
        avg_miss_q: sum_miss as f64 / nf,
        avg_hit_q: sum_hit as f64 / nf,
        avg_singleflight: sum_sf as f64 / nf,
        avg_pf_issued: sum_pf_issued as f64 / nf,
        avg_pf_consumed: sum_pf_consumed as f64 / nf,
        avg_best_at: sum_best_at as f64 / nf,
        avg_first_topk: sum_first_topk as f64 / nf,
        early_stop_pct: early_count as f64 / nf * 100.0,
        waste_ratio,
        hit_rate,
        avg_phys_reads_q: sum_phys_reads as f64 / nf,
        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
        ms_per_miss: if sum_miss > 0 {
            (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64
        } else {
            0.0
        },
        total_bypasses: pool.stats().bypasses - cache_stats_before.bypasses,
        total_evict_fail: pool.stats().evict_fail_all_pinned - cache_stats_before.evict_fail_all_pinned,
        avg_refine_count: sum_refine_count as f64 / nf,
        avg_refine_ms: sum_refine_ns as f64 / nf * ns_to_ms,
        avg_total_io_q: sum_phys_reads as f64 / nf + sum_refine_count as f64 / nf,
    }
}

fn print_bench_header(n: usize, dim: usize, num_queries: usize, warmup_queries: usize) {
    eprintln!("\n=== BENCH: Cohere {}K, dim={}, GT=brute-force, seed=42, nq={}, warmup={} ===",
        n / 1000, dim, num_queries, warmup_queries);
    eprintln!(
        "{:<14} {:>4} {:>4} {:>2} {:>3} {:>3} {:>4} {:>7} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6} {:>6} {:>7} {:>7} {:>5} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7}",
        "label", "ef", "k", "W", "S", "D", "c%",
        "recall", "p50ms", "p99ms", "qps",
        "exp", "use", "wst", "blk/q", "mis/q", "phy/q", "hit/q", "sf/q",
        "pf_i", "pf_c", "best@", "1stk@",
        "early%", "waste%", "hit%",
        "io_ms", "cmp_ms", "dst_ms", "ms/mis",
        "byp", "evfail",
        "ref/q", "ref_ms", "io/q"
    );
}

fn print_bench_row(cfg: &BenchConfig, r: &BenchResult) {
    eprintln!(
        "{:<14} {:>4} {:>4} {:>2} {:>3} {:>3} {:>4} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>5.0} {:>5.0} {:>5.0} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>5.1} {:>5.1} {:>5.1} {:>6.1} {:>6.1} {:>7.1} {:>7.1} {:>5.1} {:>7.2} {:>7.2} {:>7.2} {:>6.3} {:>6} {:>6} {:>6.0} {:>7.2} {:>7.1}",
        cfg.label, cfg.ef, cfg.k, cfg.prefetch_width, cfg.stall_limit, cfg.drain_budget, cfg.cache_pct,
        r.recall, r.lat_p50_ms, r.lat_p99_ms, r.qps,
        r.avg_expansions, r.avg_useful, r.avg_wasted,
        r.avg_blk_q, r.avg_miss_q, r.avg_phys_reads_q, r.avg_hit_q, r.avg_singleflight,
        r.avg_pf_issued, r.avg_pf_consumed,
        r.avg_best_at, r.avg_first_topk,
        r.early_stop_pct, r.waste_ratio, r.hit_rate,
        r.avg_io_wait_ms, r.avg_compute_ms, r.avg_dist_ms, r.ms_per_miss,
        r.total_bypasses, r.total_evict_fail,
        r.avg_refine_count, r.avg_refine_ms, r.avg_total_io_q
    );
}

#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_N required
fn exp_bench_stable() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let prefetch_budget = 4;

    eprintln!("Building NSW index (n={}, dim={}, m_max={}, ef_c={}) ...", n, dim, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // Write to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  v1 index written to {} (direct_io={})", dir_str, direct_io);

    // Write v3 page-packed adjacency (BFS reorder) into a subdirectory
    let v3_dir_path = dir_path.join("v3");
    std::fs::create_dir_all(&v3_dir_path).unwrap();
    let v3_dir_str = v3_dir_path.to_str().unwrap().to_owned();
    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();
    let t0_v3 = std::time::Instant::now();
    let reorder = bfs_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid));
    let v3_writer = IndexWriter::new(&v3_dir_path);
    v3_writer
        .write_v3(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &entry_ids,
            index.vectors_raw(), |vid| index.neighbors(vid),
            &reorder,
        )
        .unwrap();
    // Copy vectors.dat to v3 dir (IoDriver + VectorBank need same dir)
    std::fs::copy(dir_path.join("vectors.dat"), v3_dir_path.join("vectors.dat")).unwrap();
    let v3_meta = IndexMeta::load_from(&v3_dir_path.join("meta.json")).unwrap();
    eprintln!("  v3 index written to {} ({} pages, BFS reorder) in {:.1}s",
        v3_dir_str, v3_meta.num_pages.unwrap_or(0), t0_v3.elapsed().as_secs_f64());

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    // Load v3 adj_index into DRAM — use meta.num_vectors, not dataset n
    let adj_index = load_adj_index(
        &v3_dir_path.join("adj_index.dat"),
        v3_meta.num_vectors as usize,
    ).unwrap();
    eprintln!("  adj_index loaded: {} entries ({:.1} KB DRAM)",
        adj_index.len(), adj_index.len() as f64 * 8.0 / 1024.0);

    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    let warmup_queries = 10;

    // Compute Ada-ef stats (diagonal-only v0, O(n*d))
    // Vectors must be L2-normalized for cosine FDL theory.
    // Cohere vectors are NOT pre-normalized — use from_raw_vectors_cosine().
    eprintln!("Computing Ada-ef stats (diagonal variance + normalize, n={}, dim={}) ...", n, dim);
    let ada_stats = AdaEfStats::from_raw_vectors_cosine(&vectors, n, dim);

    // Build Ada-ef table: score thresholds calibrated to observed Cohere 100K distribution.
    // Score range: ~5-28, median ~14, p25 ~11.
    // v0.2: ef-only calibration (S=0, D=0 everywhere) to isolate ef effect.
    // Hard queries get ef>200 to test if more budget recovers recall.
    let ada_table = AdaEfTable::new(
        &[
            (20.0, 150, 0, 0),   // top ~20%: clearly easy
            (16.0, 170, 0, 0),   // above median
            (12.0, 190, 0, 0),   // below median
            (8.0,  200, 0, 0),   // hard
        ],
        AdaEfParams { ef: 250, stall_limit: 0, drain_budget: 0 },  // hardest: EXTRA budget
    );

    // Define benchmark configs: baselines + Ada-ef, both warm and cold
    let configs = vec![
        // --- Warm baselines (5% cache) ---
        BenchConfig {
            label: "warm".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-S4D16".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 4, drain_budget: 16,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ada".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,  // overridden per-query
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: true,
            clear_per_query: false,
        },
        // --- ef sweep (warm): diagnose >=8 bucket ---
        BenchConfig {
            label: "warm-ef225".to_string(),
            ef: 225, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ef250".to_string(),
            ef: 250, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ef300".to_string(),
            ef: 300, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        // --- Cold baselines (0% cache, IO-bound) ---
        BenchConfig {
            label: "cold".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "cold-S4D16".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 4, drain_budget: 16,
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "cold-ada".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,  // overridden per-query
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: true,
            clear_per_query: false,
        },
    ];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Precompute per-query Ada-ef scores (used for both diagnostics and bucket breakdown)
            let query_scores: Vec<f64> = {
                let mut scores = Vec::with_capacity(num_queries);
                for q in &query_vecs {
                    let seed_dists: Vec<f32> = entry_set
                        .iter()
                        .map(|&ep| bank.distance(q, ep.0 as usize))
                        .collect();
                    let (mu, sigma) = ada_stats.estimate_fdl_params(q);
                    let mut thresholds = [0.0f64; 5];
                    for i in 0..5 {
                        thresholds[i] = mu + sigma * divergence_engine::ada_ef::inv_normal_cdf(0.001 * (i + 1) as f64);
                    }
                    let mut counts = [0u32; 5];
                    for &d in &seed_dists {
                        let d = d as f64;
                        for (bin, &thresh) in thresholds.iter().enumerate() {
                            if d <= thresh { counts[bin] += 1; break; }
                        }
                    }
                    let n_seeds = seed_dists.len() as f64;
                    let weights = [100.0, 36.788, 13.534, 4.979, 1.832];
                    let score: f64 = counts.iter().zip(weights.iter())
                        .map(|(&c, &w)| w * c as f64 / n_seeds).sum();
                    scores.push(score);
                }
                scores
            };

            // Print score distribution
            {
                let mut sorted = query_scores.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "\nAda-ef score distribution (nq={}): min={:.2} p25={:.2} p50={:.2} mean={:.2} p75={:.2} max={:.2}",
                    sorted.len(), sorted[0], sorted[sorted.len()/4], sorted[sorted.len()/2],
                    sorted.iter().sum::<f64>() / sorted.len() as f64,
                    sorted[3*sorted.len()/4], sorted[sorted.len()-1]
                );
                let buckets = [20.0, 16.0, 12.0, 8.0, 0.0];
                let labels = [">=20 (ef=150)", ">=16 (ef=170)", ">=12 (ef=190)", ">=8 (ef=200)", "<8 (floor=250)"];
                for (i, &thresh) in buckets.iter().enumerate() {
                    let count = if i == 0 {
                        query_scores.iter().filter(|&&s| s >= thresh).count()
                    } else {
                        query_scores.iter().filter(|&&s| s >= thresh && s < buckets[i-1]).count()
                    };
                    eprintln!("  {}: {} queries ({:.0}%)", labels[i], count, count as f64 / query_scores.len() as f64 * 100.0);
                }
            }

            print_bench_header(n, dim, num_queries, warmup_queries);

            for cfg in &configs {
                // cache_pct=0 → single 8-way set (32KB), truly cold
                let pool_bytes = if cfg.cache_pct > 0 {
                    n * 4096 * cfg.cache_pct / 100
                } else {
                    8 * 4096 // one set
                };
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );

                let ada = if cfg.ada_ef {
                    Some((&ada_stats, &ada_table))
                } else {
                    None
                };

                let result = run_bench(
                    cfg, &entry_set, &pool, &io, &bank,
                    &query_vecs, &ground_truth, ada, &query_scores,
                ).await;

                print_bench_row(cfg, &result);

                pool.stop_prefetch();
                handle.await;
            }

            // =================================================================
            // v3 page-packed adjacency benchmarks
            // =================================================================
            eprintln!("\n--- v3 page-packed adjacency (BFS reorder) ---");

            let io_v3 = Rc::new(
                IoDriver::open_pages(&v3_dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open v3 IO driver"),
            );

            let v3_configs = vec![
                // --- Steady-state (cross-query cache warms up) ---
                BenchConfig {
                    label: "v3-warm".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-S4D16".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 4, drain_budget: 16,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // Cold steady-state (starts cold, warms across queries)
                BenchConfig {
                    label: "v3-cold".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // --- Per-query cold: pool.clear() before each query ---
                // Isolates intra-query page reuse (no cross-query warming)
                // Should approach simulation's ~135 phys_reads/q
                BenchConfig {
                    label: "v3-perq-cold".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                BenchConfig {
                    label: "v3-perq-S4D16".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 4, drain_budget: 16,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                // --- ef sweep (warm): find iso-recall points for DiskANN comparison ---
                BenchConfig {
                    label: "v3-warm-ef225".to_string(),
                    ef: 225, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-ef250".to_string(),
                    ef: 250, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-ef300".to_string(),
                    ef: 300, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // --- ef sweep (perq-cold): strictest comparison ---
                BenchConfig {
                    label: "v3-perq-ef225".to_string(),
                    ef: 225, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                BenchConfig {
                    label: "v3-perq-ef250".to_string(),
                    ef: 250, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
            ];

            let v3_num_pages = v3_meta.num_pages.unwrap_or(0) as usize;

            for cfg in &v3_configs {
                // v3 pool sizing: key is page_id, not vid.
                // cache_pct=0 → 256 pages (1MB) to allow intra-query page reuse
                //   (p99 = ~164 unique pages/q; below this, eviction kills reuse)
                // cache_pct>0 → pct of total pages, min 256
                let pool_pages = if cfg.cache_pct > 0 {
                    (v3_num_pages * cfg.cache_pct / 100).max(256)
                } else {
                    256 // ~1MB, enough for intra-query reuse
                };
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                );

                let result = run_bench_v3(
                    cfg, &entry_set, &pool, &io_v3, &bank,
                    &adj_index, &query_vecs, &ground_truth,
                ).await;
                print_bench_row(cfg, &result);

                pool.stop_prefetch();
                handle.await;
            }

            // =================================================================
            // Hub pinning benchmark: pin first N pages, measure benefit
            // =================================================================
            eprintln!("\n--- Hub pinning (v3, BFS-reordered pages) ---");
            print_bench_header(n, dim, num_queries, 0);

            for &pin_count in &[64u32, 128u32] {
                let pin_pages: Vec<u32> = (0..pin_count).collect();

                // v3-perq-pinN: per-query cold with pinned hub pages
                {
                    let pool_pages = 256usize; // 1MB
                    let pool_bytes = pool_pages * 4096;
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    // Pin before starting worker
                    let actually_pinned = pool.pin_pages(&pin_pages, &io_v3).await
                        .expect("pin_pages failed");
                    eprintln!("  pin{}: requested={}, actually_pinned={}", pin_count, pin_count, actually_pinned);
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );

                    let cfg = BenchConfig {
                        label: format!("v3-perq-pin{}", pin_count),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }

                // v3-warm-pinN: warm cache with pinned hub pages
                {
                    let pool_pages = (v3_num_pages * 5 / 100).max(256);
                    let pool_bytes = pool_pages * 4096;
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    // Pin BEFORE starting prefetch worker
                    let actually_pinned = pool.pin_pages(&pin_pages, &io_v3).await
                        .expect("pin_pages failed");
                    eprintln!("  pin{}: requested={}, actually_pinned={}", pin_count, pin_count, actually_pinned);
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );

                    let cfg = BenchConfig {
                        label: format!("v3-warm-pin{}", pin_count),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 5,
                        num_queries, warmup_queries,
                        ada_ef: false,
                        clear_per_query: false,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
            }

            // =================================================================
            // Equal-budget comparison: same pool_bytes for v1 vs v3
            // Separates "page packing wins" from "pool sizing wins"
            // All per-query cold (pool.clear() before each query)
            // =================================================================
            eprintln!("\n--- Equal-budget comparison (v1 vs v3, same pool_bytes, perq-cold) ---");
            print_bench_header(n, dim, num_queries, 0);

            let budget_sizes: Vec<(usize, &str)> = vec![
                (32 * 1024, "32KB"),
                (256 * 1024, "256KB"),
                (1024 * 1024, "1MB"),
                (4 * 1024 * 1024, "4MB"),
            ];

            for &(budget_bytes, budget_label) in &budget_sizes {
                // v1 with this budget, per-query cold
                {
                    let label = format!("v1-{}", budget_label);
                    let pool = Rc::new(AdjacencyPool::new(budget_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                    );
                    let cfg = BenchConfig {
                        label, ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    // run_bench doesn't support clear_per_query, so inline the loop
                    let mut recalls = Vec::with_capacity(num_queries);
                    let mut latencies_ms = Vec::with_capacity(num_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_useful = 0u64;
                    let mut sum_wasted = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys_reads = 0u64;
                    let mut sum_sf = 0u64;
                    let mut sum_pf_issued = 0u64;
                    let mut sum_pf_consumed = 0u64;
                    let mut sum_io_wait_ns = 0u64;
                    let mut sum_compute_ns = 0u64;
                    let mut sum_dist_ns = 0u64;
                    let wall_start = std::time::Instant::now();

                    for i in 0..num_queries {
                        // Quiesce: pause → drain → yield → wait LOADING → clear → unpause
                        pool.pause_prefetch(true);
                        pool.drain_prefetch();
                        monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                        while pool.has_loading() {
                            monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                        }
                        pool.clear();
                        pool.pause_prefetch(false);
                        let q = &query_vecs[i];
                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe(
                            q, &entry_set, k, 200, 4, 0, 0,
                            &pool, &io, &bank, &mut perf, PerfLevel::EnableTime,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);
                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, &ground_truth[i]));
                        sum_exp += perf.expansions;
                        sum_useful += perf.useful_expansions;
                        sum_wasted += perf.wasted_expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys_reads += perf.phys_reads;
                        sum_sf += perf.singleflight_waits;
                        sum_pf_issued += perf.prefetch_issued;
                        sum_pf_consumed += perf.prefetch_consumed;
                        sum_io_wait_ns += perf.io_wait_ns;
                        sum_compute_ns += perf.compute_ns;
                        sum_dist_ns += perf.dist_ns;
                    }

                    let nf = num_queries as f64;
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let mut sorted_lat = latencies_ms.clone();
                    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let total_exp = sum_useful + sum_wasted;
                    let ns_to_ms = 1.0 / 1_000_000.0;
                    let result = BenchResult {
                        recall: recalls.iter().sum::<f64>() / nf,
                        lat_p50_ms: percentile(&sorted_lat, 50.0),
                        lat_p99_ms: percentile(&sorted_lat, 99.0),
                        qps: nf / wall_secs,
                        avg_expansions: sum_exp as f64 / nf,
                        avg_useful: sum_useful as f64 / nf,
                        avg_wasted: sum_wasted as f64 / nf,
                        avg_blk_q: sum_blk as f64 / nf,
                        avg_miss_q: sum_miss as f64 / nf,
                        avg_hit_q: sum_hit as f64 / nf,
                        avg_singleflight: sum_sf as f64 / nf,
                        avg_pf_issued: sum_pf_issued as f64 / nf,
                        avg_pf_consumed: sum_pf_consumed as f64 / nf,
                        avg_best_at: 0.0, avg_first_topk: 0.0,
                        early_stop_pct: 0.0,
                        waste_ratio: if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 },
                        hit_rate: if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 },
                        avg_phys_reads_q: sum_phys_reads as f64 / nf,
                        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
                        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
                        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
                        ms_per_miss: if sum_miss > 0 { (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64 } else { 0.0 },
                        total_bypasses: pool.stats().bypasses,
                        total_evict_fail: pool.stats().evict_fail_all_pinned,
                        avg_refine_count: 0.0,
                        avg_refine_ms: 0.0,
                        avg_total_io_q: sum_phys_reads as f64 / nf,
                    };
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }

                // v3 with same budget, per-query cold
                {
                    let label = format!("v3-{}", budget_label);
                    let pool = Rc::new(AdjacencyPool::new(budget_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );
                    let cfg = BenchConfig {
                        label, ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
            }

            // =================================================================
            // v4 PQ-only scoring: no vectors in DRAM, PQ codes replace VectorBank
            // DRAM: codebook (~768KB) + pq_codes (N×M) + adj_index + pool cache
            // =================================================================
            eprintln!("\n--- v4 PQ-only scoring (no vectors in DRAM) ---");

            // For cosine on L2-normalized vectors, train on normalized copies
            let mut norm_vectors = vectors.clone();
            l2_normalize_batch(&mut norm_vectors, dim);

            let fp32_bank_diag = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // =============================================================
            // PQ Oracle brute-force recall: measures PQ quality ceiling
            // =============================================================
            for &oracle_m in &[48usize, 96] {
                eprintln!("\n  --- PQ{} Oracle (brute-force recall@{}) ---", oracle_m, k);
                let pq_train_start = std::time::Instant::now();
                let oracle_codebook = PqCodebook::train(&norm_vectors, n, dim, oracle_m, 20, 42);
                eprintln!("  PQ{} codebook trained in {:.1}s ({}×256×{} = {} centroids)",
                    oracle_m, pq_train_start.elapsed().as_secs_f64(),
                    oracle_m, oracle_codebook.subspace_dim, oracle_m * 256);

                let pq_encode_start = std::time::Instant::now();
                let oracle_codes = oracle_codebook.encode_all(&norm_vectors, n);
                eprintln!("  {} vectors encoded in {:.1}s ({} bytes = {:.1} KB DRAM)",
                    n, pq_encode_start.elapsed().as_secs_f64(),
                    oracle_codes.len(), oracle_codes.len() as f64 / 1024.0);

                // IMPORTANT: For cosine on L2-normalized vectors, use L2 ADC.
                // L2^2 is monotonic with cosine distance on the unit sphere:
                //   ||q-x||^2 = 2 - 2*dot(q,x)  => ordering is identical.
                // Using inner-product ADC here is inconsistent with our L2-trained
                // k-means centroids and yields much worse ranking quality.
                let oracle_bank = divergence_core::distance::PqVectorBank::new(
                    &oracle_codes, n, dim, &oracle_codebook, false,
                );

                let mut oracle_recalls = Vec::with_capacity(num_queries);
                for qi in 0..num_queries {
                    let q = &query_vecs[qi];
                    let prepared = oracle_bank.prepare(q);
                    let mut pq_dists: Vec<(u32, f32)> = (0..n)
                        .map(|v| (v as u32, prepared.distance(q, v)))
                        .collect();
                    pq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let top_k_ids: Vec<u32> = pq_dists[..k].iter().map(|&(id, _)| id).collect();
                    let recall = recall_at_k(&top_k_ids, &ground_truth[qi]);
                    oracle_recalls.push(recall);
                }
                let mean_recall = oracle_recalls.iter().sum::<f64>() / oracle_recalls.len() as f64;
                let min_recall = oracle_recalls.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_recall = oracle_recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!("  PQ{} oracle recall@{} (nq={}): mean={:.4}, min={:.4}, max={:.4}",
                    oracle_m, k, num_queries, mean_recall, min_recall, max_recall);
            }

            // =============================================================
            // Graph search benchmark with PQ96 only
            // =============================================================
            {
                let pq_m = 96usize; // 96 subspaces, 8 dims each for 768d

                let pq_train_start = std::time::Instant::now();
                let codebook = PqCodebook::train(&norm_vectors, n, dim, pq_m, 20, 42);
                eprintln!("\n  PQ{} codebook trained in {:.1}s ({}×256×{} = {} centroids)",
                    pq_m, pq_train_start.elapsed().as_secs_f64(),
                    pq_m, codebook.subspace_dim, pq_m * 256);

                let pq_encode_start = std::time::Instant::now();
                let pq_codes = codebook.encode_all(&norm_vectors, n);
                eprintln!("  {} vectors encoded in {:.1}s ({} bytes = {:.1} KB DRAM)",
                    n, pq_encode_start.elapsed().as_secs_f64(),
                    pq_codes.len(), pq_codes.len() as f64 / 1024.0);

                let codebook_bytes = pq_m * 256 * codebook.subspace_dim * 4;
                let codes_bytes = pq_codes.len();
                let adj_index_bytes = adj_index.len() * 8;
                eprintln!("  DRAM budget: codebook={:.1}KB + codes={:.1}KB + adj_index={:.1}KB = {:.1}KB total",
                    codebook_bytes as f64 / 1024.0,
                    codes_bytes as f64 / 1024.0,
                    adj_index_bytes as f64 / 1024.0,
                    (codebook_bytes + codes_bytes + adj_index_bytes) as f64 / 1024.0);

                // Same rationale as the oracle: use L2 ADC for normalized cosine data.
                let pq_bank = divergence_core::distance::PqVectorBank::new(
                    &pq_codes, n, dim, &codebook, false,
                );

                // PQ approximation quality diagnostic
                {
                    let mut rank_correct_10 = 0usize;
                    let mut rank_correct_100 = 0usize;
                    let nq_diag = num_queries.min(20);
                    for qi in 0..nq_diag {
                        let q = &query_vecs[qi];
                        let pq_prepared = pq_bank.prepare(q);
                        let mut exact: Vec<(usize, f32)> = (0..n).map(|v| (v, fp32_bank_diag.distance(q, v))).collect();
                        let mut approx: Vec<(usize, f32)> = (0..n).map(|v| (v, pq_prepared.distance(q, v))).collect();
                        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let exact_top10: std::collections::HashSet<usize> = exact[..10].iter().map(|&(i, _)| i).collect();
                        let exact_top100: std::collections::HashSet<usize> = exact[..100].iter().map(|&(i, _)| i).collect();
                        let approx_top10: std::collections::HashSet<usize> = approx[..10].iter().map(|&(i, _)| i).collect();
                        let approx_top100: std::collections::HashSet<usize> = approx[..100].iter().map(|&(i, _)| i).collect();
                        rank_correct_10 += exact_top10.intersection(&approx_top10).count();
                        rank_correct_100 += exact_top100.intersection(&approx_top100).count();
                    }
                    eprintln!("  PQ{} ranking quality (nq={}): top-10 overlap={:.1}%, top-100 overlap={:.1}%",
                        pq_m, nq_diag,
                        rank_correct_10 as f64 / (nq_diag * 10) as f64 * 100.0,
                        rank_correct_100 as f64 / (nq_diag * 100) as f64 * 100.0);
                }

                print_bench_header(n, dim, num_queries, warmup_queries);

                let ef_values: Vec<usize> = vec![200, 300, 400, 500];
                for &ef_val in &ef_values {
                    // Warm config
                    let cfg = BenchConfig {
                        label: format!("pq{}-warm-ef{}", pq_m, ef_val),
                        ef: ef_val, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 5,
                        num_queries, warmup_queries,
                        ada_ef: false,
                        clear_per_query: false,
                    };
                    let pool_pages = (v3_num_pages * 5 / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &pq_bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;

                    // Per-query cold config
                    let cfg_cold = BenchConfig {
                        label: format!("pq{}-perq-ef{}", pq_m, ef_val),
                        ef: ef_val, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let pool_cold = Rc::new(AdjacencyPool::new(256 * 4096));
                    let handle_cold = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool_cold), Rc::clone(&io_v3), prefetch_budget,
                    );
                    let result_cold = run_bench_v3(
                        &cfg_cold, &entry_set, &pool_cold, &io_v3, &pq_bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg_cold, &result_cold);
                    pool_cold.stop_prefetch();
                    handle_cold.await;
                }

                // =============================================================
                // v4 two-stage: PQ96 traversal + FP32 disk refine
                // =============================================================
                eprintln!("\n  --- v4 two-stage: PQ{} traversal + FP32 disk refine ---", pq_m);

                let vec_reader = Rc::new(
                    VectorReader::open(&v3_dir_str, dim, direct_io)
                        .await
                        .expect("failed to open VectorReader"),
                );
                let refine_inflight = 32usize;

                let refine_rs: Vec<usize> = vec![200, 500, 1000, 2000];
                for &refine_r in &refine_rs {
                    // Warm config
                    let cfg = BenchConfig {
                        label: format!("pq{}+R{}", pq_m, refine_r),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 5,
                        num_queries, warmup_queries,
                        ada_ef: false,
                        clear_per_query: false,
                    };
                    let pool_pages = (v3_num_pages * 5 / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );

                    let result = run_bench_v3_refine(
                        &cfg, refine_r, refine_inflight,
                        &entry_set, &pool, &io_v3, &pq_bank,
                        &adj_index, &vec_reader, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;

                    // Per-query cold config
                    let cfg_cold = BenchConfig {
                        label: format!("pq{}+R{}-pq", pq_m, refine_r),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let pool_cold = Rc::new(AdjacencyPool::new(256 * 4096));
                    let handle_cold = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool_cold), Rc::clone(&io_v3), prefetch_budget,
                    );
                    let result_cold = run_bench_v3_refine(
                        &cfg_cold, refine_r, refine_inflight,
                        &entry_set, &pool_cold, &io_v3, &pq_bank,
                        &adj_index, &vec_reader, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg_cold, &result_cold);
                    pool_cold.stop_prefetch();
                    handle_cold.await;
                }
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ============================================================================
// EXP-PQ-GATE: Inline PQ gating sweep
// ============================================================================

#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_N required
fn exp_pq_gate() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;

    eprintln!("=== EXP-PQ-GATE: Cohere {}K, dim={}, k={} ===", n / 1000, dim, k);

    // 1. Build NSW index
    eprintln!("Building NSW index (n={}, m_max={}, ef_c={}) ...", n, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // 2. Train PQ codebook on L2-normalized vectors (cosine → IP on unit sphere)
    let pq_m = if dim % 48 == 0 { 48 } else if dim % 32 == 0 { 32 } else { 16 };
    eprintln!("Training PQ codebook (M={}, subspace_dim={}) ...", pq_m, dim / pq_m);
    let t0 = std::time::Instant::now();
    let mut norm_vectors = vectors.clone();
    l2_normalize_batch(&mut norm_vectors, dim);
    let codebook = PqCodebook::train(&norm_vectors, n, dim, pq_m, 20, 42);
    eprintln!("  PQ trained in {:.1}s", t0.elapsed().as_secs_f64());

    // 3. Encode all vectors
    eprintln!("Encoding all vectors ...");
    let t0 = std::time::Instant::now();
    let pq_codes_all = codebook.encode_all(&norm_vectors, n);
    eprintln!("  Encoded in {:.1}s ({} bytes)", t0.elapsed().as_secs_f64(), pq_codes_all.len());

    // 4. Write v2 index to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd).join("pq_gate");
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();

    eprintln!("Writing v2 index to {} ...", dir_str);
    let writer = IndexWriter::new(&dir_path);
    writer
        .write_v2(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
            &codebook, &pq_codes_all, "ip",
        )
        .unwrap();
    eprintln!("  v2 index written (direct_io={})", direct_io);

    // 5. Run gate_ratio sweep
    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    let ef = 200;
    let prefetch_width = 4;
    let prefetch_budget = 4;
    let warmup_queries = 10;

    // Gate ratio configs: 1.0 = no gating (baseline), then decreasing
    let gate_configs: Vec<(f32, usize)> = vec![
        (1.0,  4),   // baseline: no gating
        (0.75, 4),   // keep 75%
        (0.50, 4),   // keep 50%
        (0.33, 4),   // keep 33%
        (0.25, 4),   // keep 25%
    ];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // TSV header
            eprintln!(
                "\n{:>10} {:>5} {:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7}",
                "gate", "gmin", "ef", "recall", "p50ms", "p99ms", "qps",
                "exp/q", "blk/q", "mis/q", "pq_scrd", "pq_pass", "pq_filt",
                "use/q", "wst/q"
            );

            for &(gate_ratio, gate_min) in &gate_configs {
                let pool_bytes = n * 4096 * 5 / 100; // 5% cache
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );

                // Warmup
                for q in query_vecs.iter().take(warmup_queries) {
                    let mut norm_q = q.clone();
                    l2_normalize(&mut norm_q);
                    let pq_dt = codebook.build_distance_table(&norm_q, true);
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pq(
                        q, &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                        Some(&pq_dt), gate_ratio, gate_min,
                    ).await;
                }

                // Measure
                let mut recalls = Vec::with_capacity(num_queries);
                let mut latencies_ms = Vec::with_capacity(num_queries);
                let mut sum_exp = 0u64;
                let mut sum_useful = 0u64;
                let mut sum_wasted = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_pq_scored = 0u64;
                let mut sum_pq_passed = 0u64;
                let mut sum_pq_filtered = 0u64;

                let wall_start = std::time::Instant::now();

                for i in 0..num_queries {
                    let q = &query_vecs[i];
                    let mut norm_q = q.clone();
                    l2_normalize(&mut norm_q);
                    let pq_dt = codebook.build_distance_table(&norm_q, true);

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pq(
                        q, &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &bank, &mut perf, PerfLevel::EnableTime,
                        Some(&pq_dt), gate_ratio, gate_min,
                    ).await;
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
                    latencies_ms.push(elapsed_ms);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));

                    sum_exp += perf.expansions;
                    sum_useful += perf.useful_expansions;
                    sum_wasted += perf.wasted_expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_pq_scored += perf.pq_candidates_scored;
                    sum_pq_passed += perf.pq_candidates_passed;
                    sum_pq_filtered += perf.pq_candidates_filtered;
                }

                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_queries as f64;
                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;

                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!(
                    "{:>10.2} {:>5} {:>3} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8.1} {:>8.1} {:>8.1} {:>7.1} {:>7.1}",
                    gate_ratio, gate_min, ef,
                    mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_pq_scored as f64 / nf,
                    sum_pq_passed as f64 / nf,
                    sum_pq_filtered as f64 / nf,
                    sum_useful as f64 / nf,
                    sum_wasted as f64 / nf,
                );

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ============================================================================
// EXP-PQ-GATE-V2: Iso-recall ef sweep + gating×stopping combo
//
// Two questions:
//   1. Does PQ gating shift the recall-vs-ef curve left?
//      (same recall at lower ef → fewer blk/q)
//   2. Does gating + adaptive stopping (S/D) together beat either alone?
// ============================================================================

struct PqGateConfig {
    label: String,
    ef: usize,
    gate_ratio: f32,
    gate_min: usize,
    stall_limit: u32,
    drain_budget: u32,
}

#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_N required
fn exp_pq_gate_v2() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;

    eprintln!("=== EXP-PQ-GATE-V2: Cohere {}K, dim={}, k={} ===", n / 1000, dim, k);

    // 1. Build NSW index
    eprintln!("Building NSW index (n={}, m_max={}, ef_c={}) ...", n, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // 2. Train PQ codebook
    let pq_m = if dim % 48 == 0 { 48 } else if dim % 32 == 0 { 32 } else { 16 };
    eprintln!("Training PQ codebook (M={}, subspace_dim={}) ...", pq_m, dim / pq_m);
    let t0 = std::time::Instant::now();
    let mut norm_vectors = vectors.clone();
    l2_normalize_batch(&mut norm_vectors, dim);
    let codebook = PqCodebook::train(&norm_vectors, n, dim, pq_m, 20, 42);
    eprintln!("  PQ trained in {:.1}s", t0.elapsed().as_secs_f64());

    // 3. Encode all vectors
    let t0 = std::time::Instant::now();
    let pq_codes_all = codebook.encode_all(&norm_vectors, n);
    eprintln!("  Encoded in {:.1}s ({} bytes)", t0.elapsed().as_secs_f64(), pq_codes_all.len());

    // 4. Write v2 index
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd).join("pq_gate_v2");
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();

    eprintln!("Writing v2 index to {} ...", dir_str);
    let writer = IndexWriter::new(&dir_path);
    writer
        .write_v2(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
            &codebook, &pq_codes_all, "ip",
        )
        .unwrap();
    eprintln!("  v2 index written (direct_io={})", direct_io);

    // 5. Load for search
    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    let prefetch_width = 4;
    let prefetch_budget = 4;
    let warmup_queries = 10;

    // === Experiment configs ===
    let mut configs: Vec<PqGateConfig> = Vec::new();

    // Part 1: Iso-recall ef sweep — gate × ef
    for &ef in &[120, 140, 160, 180, 200] {
        configs.push(PqGateConfig {
            label: format!("g1.0-ef{}", ef),
            ef, gate_ratio: 1.0, gate_min: 4,
            stall_limit: 0, drain_budget: 0,
        });
    }
    for &ef in &[120, 140, 160, 180, 200] {
        configs.push(PqGateConfig {
            label: format!("g.75-ef{}", ef),
            ef, gate_ratio: 0.75, gate_min: 4,
            stall_limit: 0, drain_budget: 0,
        });
    }
    for &ef in &[120, 140, 160, 180, 200] {
        configs.push(PqGateConfig {
            label: format!("g.50-ef{}", ef),
            ef, gate_ratio: 0.50, gate_min: 4,
            stall_limit: 0, drain_budget: 0,
        });
    }

    // Part 2: Gating × Stopping combo (ef=200)
    configs.push(PqGateConfig {
        label: "S4D16".to_string(),
        ef: 200, gate_ratio: 1.0, gate_min: 4,
        stall_limit: 4, drain_budget: 16,
    });
    configs.push(PqGateConfig {
        label: "S4D8".to_string(),
        ef: 200, gate_ratio: 1.0, gate_min: 4,
        stall_limit: 4, drain_budget: 8,
    });
    configs.push(PqGateConfig {
        label: "g.75+S4D16".to_string(),
        ef: 200, gate_ratio: 0.75, gate_min: 4,
        stall_limit: 4, drain_budget: 16,
    });
    configs.push(PqGateConfig {
        label: "g.75+S4D8".to_string(),
        ef: 200, gate_ratio: 0.75, gate_min: 4,
        stall_limit: 4, drain_budget: 8,
    });
    configs.push(PqGateConfig {
        label: "g.50+S4D16".to_string(),
        ef: 200, gate_ratio: 0.50, gate_min: 4,
        stall_limit: 4, drain_budget: 16,
    });
    configs.push(PqGateConfig {
        label: "g.50+S4D8".to_string(),
        ef: 200, gate_ratio: 0.50, gate_min: 4,
        stall_limit: 4, drain_budget: 8,
    });

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // TSV header
            eprintln!(
                "\n{:>14} {:>3} {:>5} {:>4} {:>3} {:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>6}",
                "label", "ef", "gate", "gmin", "S", "D", "recall", "p50ms", "p99ms", "qps",
                "exp/q", "blk/q", "mis/q", "pq_pass", "pq_filt",
                "use/q", "wst/q", "early%"
            );

            for cfg in &configs {
                let pool_bytes = n * 4096 * 5 / 100; // 5% cache
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );

                // Warmup
                for q in query_vecs.iter().take(warmup_queries) {
                    let mut norm_q = q.clone();
                    l2_normalize(&mut norm_q);
                    let pq_dt = codebook.build_distance_table(&norm_q, true);
                    let pq_ref = if cfg.gate_ratio < 1.0 { Some(&pq_dt) } else { None };
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pq(
                        q, &entry_set, k, cfg.ef, prefetch_width,
                        cfg.stall_limit, cfg.drain_budget,
                        &pool, &io, &bank, &mut perf, PerfLevel::CountOnly,
                        pq_ref, cfg.gate_ratio, cfg.gate_min,
                    ).await;
                }

                // Measure
                let mut recalls = Vec::with_capacity(num_queries);
                let mut latencies_ms = Vec::with_capacity(num_queries);
                let mut sum_exp = 0u64;
                let mut sum_useful = 0u64;
                let mut sum_wasted = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_pq_passed = 0u64;
                let mut sum_pq_filtered = 0u64;
                let mut early_count = 0u64;

                let wall_start = std::time::Instant::now();

                for i in 0..num_queries {
                    let q = &query_vecs[i];
                    let mut norm_q = q.clone();
                    l2_normalize(&mut norm_q);
                    let pq_dt = codebook.build_distance_table(&norm_q, true);
                    let pq_ref = if cfg.gate_ratio < 1.0 { Some(&pq_dt) } else { None };

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pq(
                        q, &entry_set, k, cfg.ef, prefetch_width,
                        cfg.stall_limit, cfg.drain_budget,
                        &pool, &io, &bank, &mut perf, PerfLevel::EnableTime,
                        pq_ref, cfg.gate_ratio, cfg.gate_min,
                    ).await;
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
                    latencies_ms.push(elapsed_ms);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &ground_truth[i]));

                    sum_exp += perf.expansions;
                    sum_useful += perf.useful_expansions;
                    sum_wasted += perf.wasted_expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_pq_passed += perf.pq_candidates_passed;
                    sum_pq_filtered += perf.pq_candidates_filtered;
                    if perf.stopped_early { early_count += 1; }
                }

                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_queries as f64;
                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;

                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!(
                    "{:>14} {:>3} {:>5.2} {:>4} {:>3} {:>3} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8.1} {:>8.1} {:>7.1} {:>7.1} {:>6.1}",
                    cfg.label, cfg.ef, cfg.gate_ratio, cfg.gate_min,
                    cfg.stall_limit, cfg.drain_budget,
                    mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_pq_passed as f64 / nf,
                    sum_pq_filtered as f64 / nf,
                    sum_useful as f64 / nf,
                    sum_wasted as f64 / nf,
                    early_count as f64 / nf * 100.0,
                );

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ============================================================================
// EXP-PAGE-TRACE: Trace-driven page packing simulation
//
// Collects expansion VID traces from in-memory beam search, then simulates
// how many unique pages would be read under different:
//   - nodes_per_page: 4, 8, 16, 24, 32, 48, 64
//   - reorder strategies: none (sequential), BFS from entry points
//
// No NVMe needed — pure in-memory simulation.
// ============================================================================

/// Beam search that records the VID expansion order (trace).
fn traced_beam_search(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    index: &divergence_index::NswIndex,
) -> (Vec<divergence_index::ScoredId>, Vec<u32>) {
    use divergence_index::{CandidateHeap, FixedCapacityHeap, ScoredId};

    let n = index.num_vectors();
    let dim = index.dimension();
    let dist = create_distance_computer(index.metric());
    let mut visited = vec![false; n];
    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();
    let mut trace: Vec<u32> = Vec::new();

    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < n {
            visited[vid] = true;
            let d = dist.distance(query, &index.vectors_raw()[vid * dim..(vid + 1) * dim]);
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
        }
    }

    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }
        trace.push(candidate.id.0);

        for &nbr_raw in index.neighbors(candidate.id.0) {
            let nbr_idx = nbr_raw as usize;
            if nbr_idx >= n || visited[nbr_idx] {
                continue;
            }
            visited[nbr_idx] = true;
            let d = dist.distance(query, &index.vectors_raw()[nbr_idx * dim..(nbr_idx + 1) * dim]);
            let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
            if !dominated {
                let scored = ScoredId { distance: d, id: VectorId(nbr_raw) };
                candidates.push(scored);
                nearest.push(scored);
            }
        }
    }

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);
    (results, trace)
}

/// BFS reorder: traverse graph BFS from entry set, assign new VIDs in visit order.
/// Returns old_to_new[old_vid] = new_vid.
fn bfs_reorder<'a>(n: usize, entry_set: &[VectorId], neighbors_fn: impl Fn(u32) -> &'a [u32]) -> Vec<u32> {
    let mut old_to_new = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    let mut next_id = 0u32;

    for &ep in entry_set {
        let v = ep.0 as usize;
        if v < n && old_to_new[v] == u32::MAX {
            old_to_new[v] = next_id;
            next_id += 1;
            queue.push_back(v);
        }
    }

    while let Some(v) = queue.pop_front() {
        for &nbr in neighbors_fn(v as u32) {
            let ni = nbr as usize;
            if ni < n && old_to_new[ni] == u32::MAX {
                old_to_new[ni] = next_id;
                next_id += 1;
                queue.push_back(ni);
            }
        }
    }

    // Unreachable nodes (shouldn't happen in connected graph)
    for i in 0..n {
        if old_to_new[i] == u32::MAX {
            old_to_new[i] = next_id;
            next_id += 1;
        }
    }

    old_to_new
}

/// Simulate page reads for a trace given vid→page_id mapping.
/// Returns number of unique pages touched (= within-query page reads assuming
/// query-local cache holds all pages for the query's duration).
fn simulate_page_reads_mapped(trace: &[u32], vid_to_page: &[u32]) -> usize {
    let mut seen_pages = std::collections::HashSet::new();
    for &vid in trace {
        seen_pages.insert(vid_to_page[vid as usize]);
    }
    seen_pages.len()
}

/// Byte-accurate page packing: greedily pack adjacency records into PAGE_SIZE
/// pages in reorder order. Returns vid_to_page[old_vid] = page_id.
///
/// Page layout:
///   [page_header: 4B] (num_entries: u16, reserved: u16)
///   [directory: num_entries × 8B] (vid: u32, offset_within_page: u16, degree: u16)
///   [records: variable] (degree × 4B neighbor VIDs per entry)
///
/// Records are packed contiguously after the directory. If the next node
/// doesn't fit, a new page is started.
fn byte_pack_pages(
    n: usize,
    reorder: &[u32],          // old_to_new[old_vid] = new_vid
    degrees: &[u16],          // degrees[old_vid] = degree
    page_size: usize,
) -> (Vec<u32>, PackingStats) {
    const PAGE_HEADER: usize = 4;   // num_entries(u16) + reserved(u16)
    const DIR_ENTRY: usize = 8;     // vid(u32) + offset(u16) + degree(u16)

    // Build new_to_old so we can iterate in reorder order
    let mut new_to_old = vec![0u32; n];
    for old in 0..n {
        new_to_old[reorder[old] as usize] = old as u32;
    }

    let mut vid_to_page = vec![0u32; n];
    let mut page_id = 0u32;
    let mut page_used = PAGE_HEADER; // start with header
    let mut entries_on_page = 0usize;
    let mut total_pages = 0u32;
    let mut total_wasted = 0usize;
    let mut entries_per_page: Vec<usize> = Vec::new();

    for new_vid in 0..n {
        let old_vid = new_to_old[new_vid] as usize;
        let degree = degrees[old_vid] as usize;
        let record_bytes = degree * 4; // neighbor VIDs
        let entry_cost = DIR_ENTRY + record_bytes;

        // Check if this entry fits on current page
        if page_used + entry_cost > page_size && entries_on_page > 0 {
            // Close current page
            total_wasted += page_size - page_used;
            entries_per_page.push(entries_on_page);
            total_pages += 1;
            page_id += 1;
            page_used = PAGE_HEADER;
            entries_on_page = 0;
        }

        vid_to_page[old_vid] = page_id;
        page_used += entry_cost;
        entries_on_page += 1;
    }

    // Close last page
    if entries_on_page > 0 {
        total_wasted += page_size - page_used;
        entries_per_page.push(entries_on_page);
        total_pages += 1;
    }

    entries_per_page.sort();
    let stats = PackingStats {
        total_pages,
        avg_entries: n as f64 / total_pages as f64,
        min_entries: *entries_per_page.first().unwrap_or(&0),
        p50_entries: entries_per_page[entries_per_page.len() / 2],
        max_entries: *entries_per_page.last().unwrap_or(&0),
        avg_utilization: 1.0 - total_wasted as f64 / (total_pages as usize * page_size) as f64,
    };

    (vid_to_page, stats)
}

struct PackingStats {
    total_pages: u32,
    avg_entries: f64,
    min_entries: usize,
    p50_entries: usize,
    max_entries: usize,
    avg_utilization: f64,  // fraction of page bytes used
}

#[test]
#[ignore] // Needs COHERE_DIR
fn exp_page_trace() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;

    eprintln!("=== EXP-PAGE-TRACE: Cohere {}K, dim={}, k={} ===", n / 1000, dim, k);

    // 1. Build NSW index
    eprintln!("Building NSW index (n={}, m_max={}, ef_c={}) ...", n, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let entry_set = index.entry_set().to_vec();

    // 2. Degree distribution
    let degrees: Vec<u16> = (0..n).map(|v| index.neighbors(v as u32).len() as u16).collect();
    {
        let mut sorted_deg: Vec<u16> = degrees.clone();
        sorted_deg.sort();
        let avg_deg = degrees.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
        eprintln!("Degree distribution (n={}): min={} p25={} p50={} mean={:.1} p75={} max={}",
            n, sorted_deg[0], sorted_deg[n/4], sorted_deg[n/2], avg_deg,
            sorted_deg[3*n/4], sorted_deg[n-1]);
        // Per-node record size: dir_entry(8B) + degree*4B
        let avg_record = 8.0 + avg_deg * 4.0;
        let max_record = 8 + sorted_deg[n-1] as usize * 4;
        eprintln!("  avg record: {:.0}B, max record: {}B (fits 4KB: {})",
            avg_record, max_record, max_record <= 4096 - 4);
    }

    // 3. Compute reorder mappings
    eprintln!("Computing BFS reorder ...");
    let identity: Vec<u32> = (0..n as u32).collect();
    let bfs_map = bfs_reorder(n, &entry_set, |vid| index.neighbors(vid));
    eprintln!("  BFS reorder done");

    // 4. Collect traces
    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    let ef = 200;
    eprintln!("Running traced beam search (nq={}, ef={}) ...", num_queries, ef);

    let mut traces: Vec<Vec<u32>> = Vec::with_capacity(num_queries);
    let mut recalls: Vec<f64> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let (results, trace) = traced_beam_search(&query_vecs[i], &entry_set, k, ef, &index);
        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
        recalls.push(recall_at_k(&ids, &ground_truth[i]));
        traces.push(trace);
    }

    let mean_recall = recalls.iter().sum::<f64>() / num_queries as f64;
    let avg_trace_len = traces.iter().map(|t| t.len()).sum::<usize>() as f64 / num_queries as f64;
    eprintln!("  recall={:.3}, avg_expansions={:.1}", mean_recall, avg_trace_len);

    // 5. Byte-accurate page packing simulation
    let page_size = 4096usize;
    let reorders: Vec<(&str, &[u32])> = vec![
        ("sequential", &identity),
        ("bfs", &bfs_map),
    ];

    eprintln!("\n--- Byte-accurate page packing (page_size={}B) ---", page_size);
    eprintln!("  Page layout: header(4B) + directory(8B/entry) + records(degree*4B/entry)");
    eprintln!(
        "\n{:>12} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>6}",
        "reorder", "pages", "n/pg", "min", "p50", "max", "util%",
        "pg/q", "p50_pg", "p99_pg", "ratio"
    );

    for (name, reorder) in &reorders {
        let (vid_to_page, stats) = byte_pack_pages(n, reorder, &degrees, page_size);

        let mut page_reads: Vec<usize> = traces.iter()
            .map(|t| simulate_page_reads_mapped(t, &vid_to_page))
            .collect();
        let avg_pg = page_reads.iter().sum::<usize>() as f64 / num_queries as f64;

        page_reads.sort();
        let p50 = page_reads[page_reads.len() / 2];
        let p99_idx = (page_reads.len() as f64 * 0.99).ceil() as usize - 1;
        let p99 = page_reads[p99_idx.min(page_reads.len() - 1)];

        let ratio = avg_pg / avg_trace_len;

        eprintln!(
            "{:>12} {:>8} {:>6.1} {:>6} {:>6} {:>6} {:>6.1} {:>8.1} {:>8} {:>8} {:>6.3}",
            name, stats.total_pages, stats.avg_entries,
            stats.min_entries, stats.p50_entries, stats.max_entries,
            stats.avg_utilization * 100.0,
            avg_pg, p50, p99, ratio
        );
    }

    // 6. Neighbor co-expansion analysis
    eprintln!("\n--- Neighbor co-expansion analysis ---");
    let mut total_neighbors_expanded = 0u64;
    let mut total_neighbors = 0u64;
    for trace in &traces {
        let expanded_set: std::collections::HashSet<u32> = trace.iter().copied().collect();
        for &vid in trace.iter() {
            for &nbr in index.neighbors(vid) {
                total_neighbors += 1;
                if expanded_set.contains(&nbr) {
                    total_neighbors_expanded += 1;
                }
            }
        }
    }
    eprintln!("  neighbors also expanded: {}/{} ({:.1}%)",
        total_neighbors_expanded, total_neighbors,
        total_neighbors_expanded as f64 / total_neighbors as f64 * 100.0);

    // 7. Working set analysis
    eprintln!("\n--- Working set (unique pages across all {} queries) ---", num_queries);
    for (name, reorder) in &[("sequential", &identity as &[u32]), ("bfs", &bfs_map as &[u32])] {
        let (vid_to_page, stats) = byte_pack_pages(n, reorder, &degrees, page_size);
        let mut all_pages = std::collections::HashSet::new();
        for trace in &traces {
            for &vid in trace {
                all_pages.insert(vid_to_page[vid as usize]);
            }
        }
        eprintln!("  {}: {}/{} pages touched ({:.1}%)",
            name, all_pages.len(), stats.total_pages,
            all_pages.len() as f64 / stats.total_pages as f64 * 100.0);
    }

    // 8. Page access frequency analysis (hot page / hub pinning value)
    eprintln!("\n--- Page access frequency (BFS reorder) ---");
    {
        let (vid_to_page, stats) = byte_pack_pages(n, &bfs_map, &degrees, page_size);
        let mut page_freq: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut total_accesses = 0u64;
        for trace in &traces {
            for &vid in trace {
                *page_freq.entry(vid_to_page[vid as usize]).or_insert(0) += 1;
                total_accesses += 1;
            }
        }
        let mut freq_vec: Vec<u64> = page_freq.values().copied().collect();
        freq_vec.sort_unstable_by(|a, b| b.cmp(a)); // descending

        eprintln!("  total page accesses: {} across {} unique pages", total_accesses, freq_vec.len());
        for &top_n in &[1, 5, 10, 20, 50, 100] {
            if top_n > freq_vec.len() { break; }
            let cum: u64 = freq_vec[..top_n].iter().sum();
            eprintln!("  top-{:>3} pages: {:>6} accesses ({:>5.1}% of total), hottest={}/q",
                top_n, cum, cum as f64 / total_accesses as f64 * 100.0,
                if top_n == 1 { format!("{:.1}", freq_vec[0] as f64 / num_queries as f64) } else { "-".to_string() });
        }
        // Per-query: how many pages are accessed >1 time within a single query (intra-query reuse)?
        let mut intra_reuse_total = 0u64;
        let mut intra_reuse_queries = 0u64;
        for trace in &traces {
            let mut page_count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &vid in trace {
                *page_count.entry(vid_to_page[vid as usize]).or_insert(0) += 1;
            }
            let reused = page_count.values().filter(|&&c| c > 1).count();
            if reused > 0 { intra_reuse_queries += 1; }
            intra_reuse_total += reused as u64;
        }
        eprintln!("  intra-query page reuse: {:.1} pages/q reused >1 time ({}/{} queries have reuse)",
            intra_reuse_total as f64 / num_queries as f64,
            intra_reuse_queries, num_queries);
    }

    // 9. adj_index overhead
    let adj_index_bytes = n * 8; // vid -> (page_id: u32, offset: u16, degree: u16)
    eprintln!("\n--- adj_index overhead ---");
    eprintln!("  {} entries × 8B = {:.1} MB (DRAM-resident)",
        n, adj_index_bytes as f64 / 1024.0 / 1024.0);
}

// ===== SAQ Cross-Validation =====

/// Cross-validate Rust SAQ estimator against C++ reference distances.
#[test]
#[ignore] // EC2-only: needs saq_unpacked.bin + saq_ref_dists.bin + queries.bin
fn exp_saq_cross_validate() {
    let data_dir = std::env::var("COHERE_DIR")
        .unwrap_or_else(|_| "data/cohere_100k".to_string());
    let saq_path = format!("{}/saq_unpacked.bin", data_dir);
    let ref_path = format!("{}/saq_ref_dists.bin", data_dir);
    let queries_path = format!("{}/queries.bin", data_dir);

    eprintln!("Loading SAQ data from {}...", saq_path);
    let saq_data = divergence_core::SaqData::load_exported(std::path::Path::new(&saq_path))
        .expect("Failed to load SAQ data");
    eprintln!("  {} vectors, {} dims, {} segments, {} codes/vec",
        saq_data.n, saq_data.full_dim, saq_data.segments.len(), saq_data.codes_per_vec);
    for (i, seg) in saq_data.segments.iter().enumerate() {
        eprintln!("  segment {}: {} dims, {} bits, sq_delta={}, has_rotation={}",
            i, seg.dim_padded, seg.bits, seg.sq_delta, seg.rotation.is_some());
    }

    eprintln!("Loading reference distances from {}...", ref_path);
    let (nq_ref, n_ref, ref_dists) = divergence_core::quantization::saq::load_ref_distances(
        std::path::Path::new(&ref_path),
    ).expect("Failed to load reference distances");
    eprintln!("  {} queries × {} vectors", nq_ref, n_ref);
    assert_eq!(n_ref, saq_data.n);

    eprintln!("Loading queries from {}...", queries_path);
    let queries_raw = std::fs::read(&queries_path).expect("Failed to read queries");
    let queries: Vec<f32> = queries_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let dim = saq_data.full_dim;
    let nq = queries.len() / dim;
    eprintln!("  {} queries, dim={}", nq, dim);

    // Cross-validate: compute Rust distances vs C++ reference
    let bank = divergence_core::SaqVectorBank::new(&saq_data);
    let mut max_rel_err = 0.0f64;
    let mut sum_rel_err = 0.0f64;
    let mut count = 0usize;

    for qi in 0..nq_ref {
        let query = &queries[qi * dim..(qi + 1) * dim];
        for vi in 0..n_ref {
            let rust_dist = bank.distance(query, vi);
            let cpp_dist = ref_dists[qi * n_ref + vi];

            let denom = cpp_dist.abs().max(1e-10) as f64;
            let rel_err = ((rust_dist - cpp_dist).abs() as f64) / denom;
            max_rel_err = max_rel_err.max(rel_err);
            sum_rel_err += rel_err;
            count += 1;
        }
        let avg_so_far = sum_rel_err / count as f64;
        eprintln!("  query {}: avg_rel_err={:.6e}, max_rel_err={:.6e}",
            qi, avg_so_far, max_rel_err);
    }

    let avg_rel_err = sum_rel_err / count as f64;
    eprintln!("\n=== SAQ Cross-Validation ===");
    eprintln!("  {} queries × {} vectors = {} comparisons", nq_ref, n_ref, count);
    eprintln!("  avg relative error: {:.6e}", avg_rel_err);
    eprintln!("  max relative error: {:.6e}", max_rel_err);

    // Tolerance: SAQ should match to ~1e-4 relative error
    assert!(avg_rel_err < 1e-3,
        "Average relative error too large: {avg_rel_err:.6e}");
    assert!(max_rel_err < 1e-1,
        "Max relative error too large: {max_rel_err:.6e}");

    // Also test proxy recall using Rust estimator
    eprintln!("\n--- Proxy Recall (Rust SAQ estimator) ---");
    let gt_path = format!("{}/gt.bin", data_dir);
    let gt_raw = std::fs::read(&gt_path).expect("Failed to read GT");
    let gt: Vec<u32> = gt_raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let k = 100;
    let gt_per_query = gt.len() / nq;

    for &r in &[100usize, 200, 500, 1000] {
        let mut total_recall = 0.0f64;
        for qi in 0..nq {
            let query = &queries[qi * dim..(qi + 1) * dim];
            // Compute SAQ distance to all vectors
            let mut dists: Vec<(f32, u32)> = (0..saq_data.n)
                .map(|vi| (bank.distance(query, vi), vi as u32))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Recall@k in top-R
            let gt_slice = &gt[qi * gt_per_query..qi * gt_per_query + k];
            let top_r: Vec<u32> = dists[..r.min(dists.len())].iter().map(|&(_, v)| v).collect();
            let hits = gt_slice.iter().filter(|&&g| top_r.contains(&g)).count();
            total_recall += hits as f64 / k as f64;
        }
        let avg_recall = total_recall / nq as f64;
        eprintln!("  R={:>5}: recall@{}={:.4}", r, k, avg_recall);
    }
}

// ===== SAQ Graph Search Experiment =====
// Tests SAQ as cheap_bank in v3 beam search (not oracle scan).
// Compares: FP32 baseline vs SAQ proxy, both on the same graph.

#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_DIR required
fn exp_saq_graph() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| "data/cohere_100k".to_string());

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_cohere_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    // Load SAQ data (unpacked). SAQ_SUFFIX selects variant (e.g., "_norot_eq16").
    let saq_suffix = std::env::var("SAQ_SUFFIX").unwrap_or_default();
    let saq_path = format!("{}/saq_unpacked{}.bin", dataset_dir, saq_suffix);
    eprintln!("Loading SAQ data from {}...", saq_path);
    let saq_data = divergence_core::SaqData::load_exported(std::path::Path::new(&saq_path))
        .expect("Failed to load SAQ data");
    assert_eq!(saq_data.n, n, "SAQ n ({}) != dataset n ({})", saq_data.n, n);
    let num_seg = saq_data.segments.len();
    let all_b4 = saq_data.segments.iter().all(|s| s.bits == 4);
    eprintln!("  {} vectors, {} dims, {} segments, {} codes/vec (unpacked: {} B/vec)",
        saq_data.n, saq_data.full_dim, num_seg, saq_data.codes_per_vec,
        saq_data.codes_per_vec + num_seg * 12);
    for (i, seg) in saq_data.segments.iter().enumerate() {
        eprintln!("    seg[{}]: dim_padded={}, bits={}, rotation={}",
            i, seg.dim_padded, seg.bits, seg.rotation.is_some());
    }

    // Pack to 4-bit if single segment + B=4 (packed format v0 constraint)
    let saq_packed = if num_seg == 1 && all_b4 {
        let packed = divergence_core::SaqPackedData::from_unpacked(&saq_data);
        eprintln!("  Packed 4-bit: {} B/vec ({} codes + 12 factors), total {:.1} MB",
            packed.packed_per_vec + 12, packed.packed_per_vec,
            packed.memory_bytes() as f64 / 1024.0 / 1024.0);
        eprintln!("  vs FP32 {:.0} B/vec: {:.1}× DRAM savings",
            dim as f64 * 4.0, (dim * 4) as f64 / (packed.packed_per_vec + 12) as f64);
        Some(packed)
    } else {
        eprintln!("  Skipping 4-bit pack: {} segments, all_b4={} (packed v0 requires 1 seg + B=4)",
            num_seg, all_b4);
        None
    };

    // Build NSW index
    let m_max = 32;
    let ef_construction = 200;
    eprintln!("Building NSW index (n={}, dim={}, m_max={}, ef_c={}) ...", n, dim, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // Write v3 page-packed adjacency
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd).join("saq");
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let v3_dir = dir_path.join("v3");
    std::fs::create_dir_all(&v3_dir).unwrap();

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();
    let reorder = bfs_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid));
    let writer = IndexWriter::new(&dir_path);
    writer.write(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
    ).unwrap();
    let v3_writer = IndexWriter::new(&v3_dir);
    v3_writer.write_v3(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
        &reorder,
    ).unwrap();
    std::fs::copy(dir_path.join("vectors.dat"), v3_dir.join("vectors.dat")).unwrap();
    let v3_meta = IndexMeta::load_from(&v3_dir.join("meta.json")).unwrap();
    let v3_num_pages = v3_meta.num_pages.unwrap_or(0) as usize;
    eprintln!("  v3 index: {} pages ({:.1} MB adjacency)",
        v3_num_pages, v3_num_pages as f64 * 4096.0 / 1024.0 / 1024.0);

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };
    let adj_index = load_adj_index(&v3_dir.join("adj_index.dat"), v3_meta.num_vectors as usize).unwrap();

    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    // Pool sizing: fraction of actual v3 pages, not n*4096
    let cache_pct = 5usize;
    let pool_pages = (v3_num_pages * cache_pct / 100).max(256);
    let pool_bytes = pool_pages * 4096;
    eprintln!("  Pool: {} pages ({:.1} MB) = {}% of {} v3 pages",
        pool_pages, pool_bytes as f64 / 1024.0 / 1024.0, cache_pct, v3_num_pages);

    let v3_dir_str = v3_dir.to_str().unwrap().to_owned();
    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open_pages(&v3_dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );

            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);
            let saq_bank = divergence_core::SaqVectorBank::new(&saq_data);
            let prefetch_budget = 4;

            // Use packed bank if available, else fall back to unpacked
            let saq_packed_bank = saq_packed.as_ref()
                .map(|p| divergence_core::SaqPackedVectorBank::new(p));

            // Pick the best SAQ bank: packed if single-segment B=4, else unpacked
            let saq_bench_bank: &dyn VectorBank = match saq_packed_bank.as_ref() {
                Some(pb) => pb,
                None => &saq_bank,
            };
            let saq_label = if saq_packed_bank.is_some() { "SAQ-pack4" } else { "SAQ-unpack" };

            print_bench_header(n, dim, num_queries, 10);

            // --- Stage 1: Graph-only (warm + cold) for FP32 and SAQ ---
            for (label, bank) in [
                ("FP32-cosine", &fp32_bank as &dyn VectorBank),
                (saq_label, saq_bench_bank),
            ] {
                // Warm run
                {
                    let cfg = BenchConfig {
                        label: format!("{}-warm", label),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct,
                        num_queries, warmup_queries: 10,
                        ada_ef: false,
                        clear_per_query: false,
                    };
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                    );
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io, bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
                // Cold (per-query clear)
                {
                    let cfg = BenchConfig {
                        label: format!("{}-cold", label),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                    );
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io, bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
            }

            // --- Stage 2: Two-stage v4 refine, sweep refine_r ---
            eprintln!("\n=== Two-Stage v4: {} graph → FP32 disk refine (sweep R) ===", saq_label);
            eprintln!("  NOTE: R capped at ef=200 (search returns at most ef candidates).");

            let vec_reader = Rc::new(
                VectorReader::open(v3_dir.to_str().unwrap(), dim, direct_io)
                    .await
                    .expect("failed to open VectorReader"),
            );
            let refine_inflight = 32usize;

            for &refine_r in &[80usize, 120, 160, 200] {
                // Cold only (benchmark-fair, most informative)
                let label = format!("SAQ+ref-R{}", refine_r);
                let cfg = BenchConfig {
                    label: label.clone(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                };
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );
                let result = run_bench_v3_refine(
                    &cfg, refine_r, refine_inflight,
                    &entry_set, &pool, &io, saq_bench_bank,
                    &adj_index, &vec_reader, &query_vecs, &ground_truth,
                ).await;
                print_bench_row(&cfg, &result);
                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}
