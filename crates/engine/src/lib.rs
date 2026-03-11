pub mod ada_ef;
pub mod aligned;
pub mod cache;
pub mod calibrate;
pub mod engine;
pub mod io;
pub mod perf;
pub mod runtime;
pub mod search;

pub use ada_ef::{AdaEfParams, AdaEfStats, AdaEfTable, estimate_ada_ef};
pub use aligned::AlignedBuf;
pub use cache::{AdjacencyPool, CacheGuard, CacheStatsSnapshot};
pub use calibrate::{calibrate_device, CalibrationResult, QdMeasurement};
pub use engine::{
    CoreSetup, Engine, EngineConfig, EngineHealth, EngineStats, GlobalQueryLimiter,
    HealthChecker, QueryPermit,
};
pub use io::{Fp16VectorReader, GlobalIoBudget, Int8VectorReader, IoDriver, VectorReader, default_global_qd};
pub use perf::{
    CoreSnapshot, Histogram, PerfLevel, QueryRecorder, SchedLagTracker, SearchGuard,
    SearchPerfContext,
};
pub use runtime::{spawn_worker, WorkerConfig};
pub use search::{
    disk_graph_search, disk_graph_search_exp, disk_graph_search_pipe, disk_graph_search_pipe_v3,
    disk_graph_search_pipe_v3_refine, disk_graph_search_pipe_v3_refine_3stage,
    disk_graph_search_pipe_v3_refine_fp16, disk_graph_search_pipe_v3_refine_int8,
    disk_graph_search_pipe_v3_pagesched, disk_graph_search_pipe_v3_traced,
    disk_graph_search_pq, disk_graph_search_refine,
    TraceRecorder,
};
