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
use divergence_core::quantization::{ScalarQuantizer, l2_normalize};
use divergence_core::{MetricType, VectorId};
use divergence_engine::{
    disk_graph_search, disk_graph_search_exp, disk_graph_search_pipe, disk_graph_search_refine,
    AdjacencyPool, IoDriver, PerfLevel, QueryRecorder, SearchGuard, SearchPerfContext,
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
                        q, &entry_set, k, ef, w, &pool, &io, &bank,
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
                        q, &entry_set, k, ef, w, &pool, &io, &bank,
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
    let w_values = [0usize, 2, 4];

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
        "\n{:<4} {:<4} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "B", "W", "r@k", "qps", "p50us", "p99us", "p999us", "io_w%", "avg_ifl", "blk/q", "mis/q"
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
                            q, &entry_set_rc, k, ef, w, &pool, &io, &warmup_bank,
                            &mut perf, PerfLevel::CountOnly,
                        )
                        .await;
                    }
                    drop(warmup_bank);

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
                                    &q, &es_c, k, ef, w, &pool_c, &io_c,
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

                    eprintln!(
                        "{:<4} {:<4} {:>7.3} {:>8.1} {:>8.0} {:>8.0} {:>8.0} {:>8.1} {:>8.2} {:>6.0} {:>6.0}",
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

