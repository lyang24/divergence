//! Integration test: build NSW → write to disk → async search → verify results match.
//!
//! These tests require io_uring support (Linux 5.1+, not inside unprivileged containers).
//! They are automatically skipped if io_uring is unavailable.

use std::path::Path;

use divergence_core::distance::{
    create_distance_computer, fp32_to_fp16, FP16VectorBank, FP32VectorBank, VectorBank,
};
use divergence_core::{MetricType, VectorId};
use divergence_engine::{
    disk_graph_search, disk_graph_search_refine, AdjacencyPool, IoDriver, PerfLevel,
    QueryRecorder, SearchGuard, SearchPerfContext,
};
use divergence_index::{NswBuilder, NswConfig};
use divergence_storage::{load_vectors, IndexMeta, IndexWriter};

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
    let distance = create_distance_computer(MetricType::L2);

    // 5. Run async disk search inside monoio runtime
    if !with_runtime(|rt| {
        let disk_results = rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(64 * 1024);
            let bank = FP32VectorBank::new(&disk_vectors, dim, &*distance);
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
    let dist = create_distance_computer(MetricType::L2);

    // 4. Generate query batch
    let queries: Vec<Vec<f32>> = generate_vectors(num_queries, dim, 999);

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(256 * 1024);
            let bank = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
    let dist = create_distance_computer(MetricType::L2);
    let queries = generate_vectors(num_queries, dim, 999);

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(n * 4096);
            let bank = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
    let dist = create_distance_computer(MetricType::L2);
    let queries = generate_vectors(num_queries, dim, 999);

    let cache_sizes_kb: Vec<usize> = vec![256, 1024, 4096, 8192];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let bank = FP32VectorBank::new(&disk_vectors, dim, &*dist);

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
        let dist = create_distance_computer(MetricType::L2);
        let queries = generate_vectors(num_queries, dim, 999);

        if !with_runtime(|rt| {
            rt.block_on(async {
                let io = IoDriver::open(&dir_str, dim, 64, false)
                    .await
                    .expect("failed to open IO driver");

                let pool = AdjacencyPool::new(256 * 1024);
                let bank = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
// Experiment 4: FP32 vs FP16 distance comparison
// ---------------------------------------------------------------------------

/// A/B comparison: FP32 vs FP16 distance at dim=512 and dim=768.
/// Measures: dist_ns reduction, total latency reduction, distance accuracy.
#[test]
fn exp_fp32_vs_fp16() {
    let n = 2000;
    let k = 10;
    let ef = 64;
    let num_queries = 100;
    let dims = [512, 768];

    eprintln!("\n========== EXP4: FP32 vs FP16 ==========");
    eprintln!(
        "{:<8} {:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "dim", "mode", "total_us", "dist_us", "dist%", "dist/call", "speedup"
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

                // --- FP32 pass ---
                let pool_fp32 = AdjacencyPool::new(256 * 1024);
                let bank_fp32 = FP32VectorBank::new(&disk_vectors, dim, &*dist);
                let mut fp32_total = 0u64;
                let mut fp32_dist = 0u64;
                let mut fp32_dist_calls = 0u64;

                for q in &queries {
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    disk_graph_search(
                        q, &entry_set, k, ef, &pool_fp32, &io, &bank_fp32,
                        &mut perf, PerfLevel::EnableTime,
                    )
                    .await;
                    fp32_total += t.elapsed().as_nanos() as u64;
                    fp32_dist += perf.dist_ns;
                    fp32_dist_calls += perf.distance_computes;
                }

                // --- FP16 pass ---
                let pool_fp16 = AdjacencyPool::new(256 * 1024);
                let bank_fp16 = FP16VectorBank::new(&vectors_fp16, dim, MetricType::L2);
                let mut fp16_total = 0u64;
                let mut fp16_dist = 0u64;
                let mut fp16_dist_calls = 0u64;

                for q in &queries {
                    let mut perf = SearchPerfContext::default();
                    let t = std::time::Instant::now();
                    disk_graph_search(
                        q, &entry_set, k, ef, &pool_fp16, &io, &bank_fp16,
                        &mut perf, PerfLevel::EnableTime,
                    )
                    .await;
                    fp16_total += t.elapsed().as_nanos() as u64;
                    fp16_dist += perf.dist_ns;
                    fp16_dist_calls += perf.distance_computes;
                }

                let nq = num_queries as f64;
                let fp32_mean = fp32_total as f64 / nq / 1000.0;
                let fp16_mean = fp16_total as f64 / nq / 1000.0;
                let fp32_dist_mean = fp32_dist as f64 / nq / 1000.0;
                let fp16_dist_mean = fp16_dist as f64 / nq / 1000.0;
                let fp32_dist_pct = fp32_dist as f64 / fp32_total as f64 * 100.0;
                let fp16_dist_pct = fp16_dist as f64 / fp16_total as f64 * 100.0;
                let fp32_per_call = if fp32_dist_calls > 0 {
                    fp32_dist as f64 / fp32_dist_calls as f64
                } else {
                    0.0
                };
                let fp16_per_call = if fp16_dist_calls > 0 {
                    fp16_dist as f64 / fp16_dist_calls as f64
                } else {
                    0.0
                };

                let dist_speedup = if fp16_per_call > 0.0 {
                    fp32_per_call / fp16_per_call
                } else {
                    0.0
                };
                let total_speedup = if fp16_mean > 0.0 {
                    fp32_mean / fp16_mean
                } else {
                    0.0
                };

                eprintln!(
                    "{:<8} {:<6} {:>10.0} {:>10.0} {:>10.1} {:>10.0} {:>10}",
                    dim, "fp32", fp32_mean, fp32_dist_mean, fp32_dist_pct, fp32_per_call, "-"
                );
                eprintln!(
                    "{:<8} {:<6} {:>10.0} {:>10.0} {:>10.1} {:>10.0} {:>10.2}x",
                    dim, "fp16", fp16_mean, fp16_dist_mean, fp16_dist_pct, fp16_per_call,
                    dist_speedup
                );
                eprintln!(
                    "  dim={} total speedup: {:.2}x ({:.0}us → {:.0}us)",
                    dim, total_speedup, fp32_mean, fp16_mean
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
        let dist = create_distance_computer(MetricType::L2);
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
                let bank_fp32 = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
    let dist = create_distance_computer(MetricType::L2);
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
            let bank_fp32 = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
    let dist = create_distance_computer(MetricType::L2);
    let query = generate_vectors(1, dim, 999)[0].clone();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");
            let pool = AdjacencyPool::new(64 * 1024);
            let bank = FP32VectorBank::new(&disk_vectors, dim, &*dist);
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
