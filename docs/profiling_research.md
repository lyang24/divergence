# Profile Harness Research Distillation

Sources: RocksDB PerfContext, ScyllaDB latency measurement, zns-tools eBPF,
NSDI'22 end-host diagnosis (Haecki et al.)

---

## 1. RocksDB PerfContext & IOStatsContext

### What They Built

Thread-local per-operation counter system with ~90 fields (PerfContext) and ~12
fields (IOStatsContext). Two separate structs because IO syscall stats and
internal operation stats serve different audiences.

**Key files**: `include/rocksdb/perf_context.h`, `iostats_context.h`,
`monitoring/perf_step_timer.h`, `monitoring/histogram.h`

### Design We Should Adopt

**1. Thread-local storage with PerfLevel gating**

```cpp
enum PerfLevel {
    kDisable = 1,           // zero overhead
    kEnableCount = 2,       // counters only (no timing)
    kEnableWait = 3,        // + IO wait timing
    kEnableTime = 6,        // + all timing including mutex
};
extern thread_local PerfLevel perf_level;
extern thread_local PerfContext perf_context;
```

For Divergence: We're single-threaded per core (monoio). Cell<u64> counters are
cheaper than thread_local atomics. PerfLevel is valuable — start with
`EnableCount` always on, `EnableTime` on demand.

**2. Reset → operate → snapshot pattern**

```cpp
get_perf_context()->Reset();    // zero all counters
db->Get(key, &value);           // operation runs, counters accumulate
auto snapshot = *get_perf_context();  // copy out
```

For Divergence: Reset before each query, snapshot after. Allows per-query stats
without allocation (the struct is pre-allocated thread-local).

**3. PerfStepTimer — conditional zero-cost timing**

```cpp
class PerfStepTimer {
    bool enabled;  // checked once at construction from perf_level
    uint64_t start;
    uint64_t* metric;  // pointer to counter field

    void Start() { if (enabled) start = clock_gettime(MONOTONIC); }
    void Stop()  { if (enabled) *metric += now() - start; }
};
```

Key: the branch is perfectly predictable (always same for entire query).
`clock_gettime(CLOCK_MONOTONIC)` costs ~20ns on modern Linux — negligible for
operations >10μs, but adds up if called per distance compute. Solution: only
time coarse phases (IO wait, total), derive compute as residual.

**4. Fixed-size logarithmic histogram**

109 buckets with 1.5× multiplier. Bucket lookup via binary search (O(log 109) ≈
7 comparisons). All atomic — but for single-threaded, Cell<u64> suffices.

For Divergence: adopt bucket structure. 109 buckets covers 1μs to 100s which
is more than enough for query latencies.

### What We Skip

- ~90 fields is excessive. We need ~15 fields for search.
- CPU time via `CLOCK_THREAD_CPUTIME_ID` — we're single-threaded, wall time
  minus IO wait = compute time (no scheduling noise on pinned cores).
- Per-level context (RocksDB tracks per-SST-level). We don't have levels.
- Compile-time `#define NPERF_CONTEXT` — use Rust `cfg` feature flag instead.

---

## 2. ScyllaDB Latency Measurement

### What They Built

Dual-tier histogram system for a Seastar (thread-per-core) database.
Per-shard stats with probabilistic sampling, aggregated on-demand.

**Key files**: `utils/estimated_histogram.hh`, `utils/histogram.hh`,
`utils/latency.hh`, `test/perf/perf.hh`

### Design We Should Adopt

**1. Probabilistic sampling for timing (1/128)**

```cpp
struct ihistogram {
    uint64_t sample_mask = 0x80;  // 1 in 128
    void mark(latency_counter& lc) {
        _count++;
        if (lc.is_start()) {       // only sampled ops have start time
            _total++;
            record(lc.stop());      // record actual latency
        }
    }
};
```

Not all queries are timed — only 1/128 get `clock_gettime()` calls. Counter
`_count` is always incremented (cheap), but timing overhead is amortized.

For Divergence: at MVP, always time (N is small). For production with 100K+ QPS,
adopt 1/128 sampling to keep timing overhead < 0.1%.

**2. time_estimated_histogram — compact, mergeable**

Log-linear buckets (4 sub-buckets per exponential range). Range: 512μs–33.5s.
Merge = element-wise addition. Quantile = binary search.

For Divergence: adopt for per-core histograms. Merge across cores on demand
(same pattern as our future multi-core aggregation).

**3. Scheduling latency measurer (reactor stall detection)**

```cpp
class scheduling_latency_measurer {
    time_point _last = now();
    estimated_histogram _hist;

    void tick() {
        auto stall = now() - _last;
        _last = now();
        _hist.add(stall.count());
    }
};
```

Schedule periodic task every 1ms. Deviation from 1ms = event loop stall.
Records in histogram — p99 stall duration reveals if monoio is stalling.

For Divergence: critical. If p99 is high but IO wait is low, stalls in the
monoio event loop are the likely cause. This detector disambiguates.

**4. Rolling window (10s) for steady-state reporting**

`summary_calculator` tracks delta between current and previous histogram.
Reports p50/p95/p99 over last 10 seconds, not since boot.

For Divergence: adopt for benchmark harness. First N queries = warmup, report
only steady-state metrics.

### What We Skip

- Shard-0-only registration (we don't have a metrics registry yet).
- Welford's online variance (we need percentiles, not stddev).

---

## 3. zns-tools (eBPF Cross-Layer Profiling)

### What They Built

Multi-layer bpftrace probes: NVMe → block layer → F2FS → VFS → RocksDB.
Correlates via request tags and inode numbers. Outputs Chrome Trace Format
timelines for cross-layer visualization.

**Key files**: `zns-tools.app/zns-probes.bt`, `tracegen.py`

### Design We Should Adopt

**1. Request-tag matching for IO latency breakdown**

```c
// On NVMe command submission
k:nvme_setup_cmd {
    @submit_time[$request_tag] = nsecs;
}
// On NVMe command completion
k:nvme_complete_rq {
    @device_lat = nsecs - @submit_time[$request_tag];
}
```

For Divergence: we can tag each io_uring SQE and measure:
- t_submit: app calls io_uring_enter
- t_complete: CQE arrives
- device_latency = t_complete - t_submit (includes kernel + device)

**2. Block layer probes catch ALL IO paths (including io_uring)**

`nvme_setup_cmd` / `nvme_complete_rq` fire regardless of submission path.
No special io_uring probes needed for device-level visibility.

For Divergence: when we need kernel-side confirmation of IO latency, attach
kprobes to `nvme_setup_cmd` / `nvme_complete_rq`. Always-on overhead: ~1-2%.

**3. Chrome Trace Format for timeline visualization**

```json
{"traceEvents": [
  {"name": "nvme_read", "ph": "B", "ts": 1000, "args": {"LBA": "0x3000"}},
  {"name": "nvme_read", "ph": "E", "ts": 2500}
]}
```

Open in Perfetto or chrome://tracing. Shows cross-layer timing at a glance.

For Divergence: output per-query trace events in CTF format. Enables visual
debugging of "where did this p99 query spend its time?"

### What We Skip

- F2FS / VFS probes (we use O_DIRECT, bypassing filesystem cache).
- Zone-specific tracking (we're on conventional NVMe, not ZNS).
- bpftrace as primary tool (we instrument in-process first, eBPF for validation).

---

## 4. NSDI'22: End-Host Latency Diagnosis (NSight)

### Key Findings

**Overhead hierarchy of instrumentation:**

| Method | CPU Overhead |
|--------|-------------|
| NIC/NVMe hardware timestamps | ~0% |
| App-level soft timestamps (batched) | ~0.5% |
| kprobes on hot paths | ~1-2% |
| eBPF (per-function) | May be significant (config-dependent) |
| Ftrace (all functions) | Very high (>100%) |

**Conclusion**: Application-side timestamps + hardware timestamps = < 3.1% total
overhead. eBPF and Ftrace overhead varies widely by attach point, output mode,
and event frequency — treat the paper's numbers as relative ordering, not
absolutes. Rule: instrument phase boundaries in-process (always on), use
kprobes/eBPF only short-term when app-side numbers look suspicious.

### Design We Should Adopt

**1. Shim-layer timestamps at phase boundaries**

Instrument 3-5 key points, not every function:
- Query start
- IO submission (before get_or_load await)
- IO completion (after get_or_load returns)
- Distance compute phase start/end
- Query complete

This gives the breakdown without per-function overhead.

**2. Head-of-line blocking dominates p99 in async IO**

The paper found that p99 in memcached (async, multi-core) was dominated by
**HOL blocking** — one request waiting while the core processes another.
NOT device latency (which is p50-stable).

For Divergence (monoio): each core processes queries cooperatively. If one
query's `get_or_load` suspends and another query runs compute, the first
query's latency includes the second's compute time. This is the primary p99
risk in our architecture.

**Proving HOL requires two timers**: `io_wait_ns` and `compute_ns`. If
`compute_ns` is high AND `sched_lag` spikes correlate with p99 spikes,
then HOL is confirmed. Response:
- Reduce ef / expansion budget (cap compute per query)
- Move refinement off the traversal critical path
- Batch distance compute with SIMD (reduce per-expansion compute time)
Without these two timers, HOL is speculation, not diagnosis.

**3. IO pressure indicators (not a single "80%" threshold)**

Track io_uring submission queue depth before each submit. High SQ depth =
requests queuing in kernel = latency spike incoming.

For Divergence, four precise metrics (not a single utilization %):
- `in_flight_adj_reads / max_in_flight` — semaphore utilization
- `avg_loading_await_ns` — mean time waiting on LOADING dedup
- `evict_fail_all_pinned` — cache thrashing signal (from pool stats)
- NVMe queue depth (system-side, via eBPF when investigating)

Use these as leading indicators, not thresholds. Correlate spikes in
these metrics with p99 latency spikes to identify root cause.

**4. Anomaly disambiguation: nested vs. parent**

Don't blame a parent function if its child explains >80% of the latency.
Only report leaf-level anomalies as root causes.

For Divergence: in the profile report, break down: if total_ns = 500μs and
io_wait_ns = 450μs, report "IO-bound (90%)". If io_wait_ns = 50μs, report
"compute-bound (90%)". Don't double-count.

### What We Skip

- NIC hardware timestamps (we're not doing network profiling).
- Clock domain reconciliation (single host, CLOCK_MONOTONIC suffices).
- Full CPU profiling with perf (defer to when we need function-level drill-down).

---

## Synthesis: What Divergence Needs

### 6 Patterns to Adopt

1. **Thread-local SearchPerfContext** (RocksDB pattern)
   - Cell<u64> counters, no atomics (monoio is single-threaded)
   - Reset before query, snapshot after
   - Fields: blocks_read, distance_computes, expansions, cache_hits,
     cache_misses, cache_bypasses, io_wait_ns, total_ns

2. **PerfLevel gating** (RocksDB pattern)
   - Level 0: counters only (blocks_read, distance_computes) — always on
   - Level 1: + coarse timing (io_wait_ns, total_ns) — ~20ns/block overhead
   - Level 2: + per-phase timing, CTF trace output — benchmark only

3. **Fixed-bucket logarithmic histogram** (RocksDB + ScyllaDB)
   - ~100 buckets, 1.5× multiplier, covers 1μs–100s
   - Per-core instance, merge on demand
   - Record: total_ns, io_wait_ns, compute_ns (derived)

4. **Scheduling stall detector** (ScyllaDB pattern)
   - Periodic 1ms tick in monoio event loop
   - Deviation histogram reveals event loop stalls
   - Disambiguates "IO slow" vs "event loop blocked"

5. **Phase-boundary timestamps** (NSDI'22 pattern)
   - 3-5 instrumentation points, not per-function
   - Query start → IO phase → compute phase → query end
   - Derive breakdown: io_wait% vs compute%

6. **HOL blocking awareness** (NSDI'22 finding)
   - Track "own time" vs "wall time" per query
   - If wall_time >> own_time, another coroutine ate our time slice
   - Expose io_uring SQ utilization as leading indicator

### SearchPerfContext Fields (MVP)

```rust
pub struct SearchPerfContext {
    // Counters (always on, ~zero overhead)
    pub blocks_read: u64,          // get_or_load calls
    pub cache_hits: u64,           // pool stats delta
    pub cache_misses: u64,         // pool stats delta
    pub cache_bypasses: u64,       // pool stats delta
    pub distance_computes: u64,    // distance() calls
    pub expansions: u64,           // candidates expanded

    // Timing (Level 1+, ~20ns per block read)
    pub io_wait_ns: u64,           // wall time in get_or_load
    pub total_ns: u64,             // wall time for entire search
    // compute_ns = total_ns - io_wait_ns (derived, not measured)
}
```

### Profile Harness Requirements

1. Run N queries, collect per-query SearchPerfContext
2. Compute p50/p99 of: total_ns, io_wait_ns, compute_ns, blocks_read
3. Cold pass (empty cache) vs warm pass (same queries repeated)
4. Print: `blocks/query`, `distance_computes/query`, `io_wait%`, `cache_hit_rate%`
5. Answer: "Is p99 IO-bound or compute-bound?"
6. Answer: "Cache hit rate 0%→X% reduced blocks/query by Y%"

### Deferred (Post-MVP)

- Probabilistic 1/128 sampling (ScyllaDB) — when QPS > 10K
- CTF trace output (zns-tools) — for per-query visual debugging
- kprobe on nvme_setup_cmd/complete_rq (zns-tools) — kernel-side IO validation
- Event loop stall detector (ScyllaDB) — when running multi-query concurrent
- HOL blocking tracker — when running multi-query concurrent
