//! IO driver with two-level inflight budgeting.
//!
//! Two-level IO budget:
//!   1. GlobalIoBudget (atomic, cross-core) — caps total device queue depth
//!   2. LocalSemaphore (per-core, RefCell) — caps per-core burst, provides async yield
//!
//! Every IO must acquire both: global token first, then local permit.
//! This ensures the NVMe device queue stays within the sweet-spot QD
//! regardless of how many cores are active.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use std::os::unix::fs::OpenOptionsExt as _;

use monoio::fs::File;

use crate::aligned::AlignedBuf;
use divergence_storage::BLOCK_SIZE;

// ---------------------------------------------------------------------------
// LocalSemaphore — single-threaded, no Send
// ---------------------------------------------------------------------------

struct SemState {
    permits: usize,
    waiters: VecDeque<Waker>,
}

/// Single-threaded async semaphore for bounding inflight IO.
pub struct LocalSemaphore {
    state: RefCell<SemState>,
}

impl LocalSemaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            state: RefCell::new(SemState {
                permits,
                waiters: VecDeque::new(),
            }),
        }
    }

    /// Acquire a permit. Suspends if none available.
    pub fn acquire(&self) -> SemAcquire<'_> {
        SemAcquire { sem: self }
    }

    fn try_acquire(&self) -> bool {
        let mut state = self.state.borrow_mut();
        if state.permits > 0 {
            state.permits -= 1;
            true
        } else {
            false
        }
    }

    /// Number of permits currently available (not acquired).
    pub fn available(&self) -> usize {
        self.state.borrow().permits
    }

    fn release(&self) {
        let mut state = self.state.borrow_mut();
        state.permits += 1;
        if let Some(waker) = state.waiters.pop_front() {
            waker.wake();
        }
    }

    fn register_waker(&self, waker: Waker) {
        self.state.borrow_mut().waiters.push_back(waker);
    }
}

pub struct SemAcquire<'a> {
    sem: &'a LocalSemaphore,
}

impl<'a> Future for SemAcquire<'a> {
    type Output = SemPermit<'a>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.sem.try_acquire() {
            Poll::Ready(SemPermit { sem: this.sem })
        } else {
            // Must re-register on every poll. If a previous wakeup occurred but
            // another task stole the permit before we were polled (lost wakeup),
            // we need a fresh waker in the queue or we'll never be woken again.
            this.sem.register_waker(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// RAII permit — releases on drop.
pub struct SemPermit<'a> {
    sem: &'a LocalSemaphore,
}

impl<'a> Drop for SemPermit<'a> {
    fn drop(&mut self) {
        self.sem.release();
    }
}

// ---------------------------------------------------------------------------
// GlobalIoBudget — cross-core device queue depth cap
// ---------------------------------------------------------------------------

/// Global IO budget shared across all monoio cores. Caps total inflight IO
/// to the NVMe device's sweet-spot queue depth (typically 12-16 for 4KB
/// random reads on a single NVMe SSD).
///
/// Uses atomic CAS — no locks, no wakers, no cross-core coordination.
/// Per-core waiters poll via yield when the global budget is exhausted.
///
/// Sizing guideline: set capacity to the QD where fio shows IOPS near-peak
/// with p99 still acceptable. For i4i.2xlarge NVMe: QD=16.
pub struct GlobalIoBudget {
    available: AtomicUsize,
    capacity: usize,
}

// Safety: AtomicUsize is Send+Sync.
unsafe impl Send for GlobalIoBudget {}
unsafe impl Sync for GlobalIoBudget {}

impl GlobalIoBudget {
    pub fn new(capacity: usize) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
            capacity,
        }
    }

    /// Try to acquire one token. Returns true on success.
    pub fn try_acquire(&self) -> bool {
        let mut current = self.available.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return false;
            }
            match self.available.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(new) => current = new,
            }
        }
    }

    /// Release one token back to the pool.
    pub fn release(&self) {
        self.available.fetch_add(1, Ordering::Release);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available(&self) -> usize {
        self.available.load(Ordering::Relaxed)
    }
}

/// RAII global IO token — releases on drop.
pub struct GlobalIoPermit<'a> {
    budget: &'a GlobalIoBudget,
}

impl Drop for GlobalIoPermit<'_> {
    fn drop(&mut self) {
        self.budget.release();
    }
}

/// Acquire a global IO token, yielding to the monoio event loop if none
/// available. This lets io_uring CQE processing run (completing IOs and
/// freeing tokens) between attempts.
async fn acquire_global(budget: &GlobalIoBudget) -> GlobalIoPermit<'_> {
    loop {
        if budget.try_acquire() {
            return GlobalIoPermit { budget };
        }
        // Yield to event loop — CQE completions release tokens.
        monoio::time::sleep(std::time::Duration::from_micros(5)).await;
    }
}

/// Default global queue depth for production use.
///
/// Rule from AD-6: `max(16, 4 × cores)`. Below the knee, sem_wait% spikes
/// and p99 explodes. Over-provisioning is harmless.
pub fn default_global_qd(num_cores: usize) -> usize {
    16usize.max(4 * num_cores)
}

// ---------------------------------------------------------------------------
// IoDriver
// ---------------------------------------------------------------------------

/// Async IO driver for reading adjacency blocks from disk.
///
/// Two-level inflight budget:
///   1. `global_budget` (optional, Arc<GlobalIoBudget>) — device-wide QD cap
///   2. `adj_sem` (per-core LocalSemaphore) — per-core burst cap
///
/// If global_budget is None, only the local adj_sem controls inflight depth.
/// This is the single-core / test mode.
pub struct IoDriver {
    adj_file: File,
    adj_sem: LocalSemaphore,
    adj_capacity: usize,
    dimension: usize,
    /// Cross-core global IO budget. None = single-core / test mode.
    global_budget: Option<Arc<GlobalIoBudget>>,
    /// Health flag shared with HealthChecker. None = test mode.
    #[allow(dead_code)]
    health: Option<Arc<AtomicU8>>,
    /// Cumulative nanoseconds spent waiting for global + local permits.
    sem_wait_ns: Cell<u64>,
    /// Cumulative nanoseconds spent in actual NVMe reads (device time).
    device_ns: Cell<u64>,
    /// Number of IO operations completed (for averaging).
    io_count: Cell<u64>,
}

impl IoDriver {
    /// Open index files for async reading.
    ///
    /// `direct_io`: set to false for tmpfs/tests (O_DIRECT doesn't work on tmpfs).
    /// `global_budget`: shared device-level QD cap. None for single-core / tests.
    pub async fn open(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
    ) -> io::Result<Self> {
        Self::open_with_budget(index_dir, dimension, adj_inflight, direct_io, None).await
    }

    /// Open v3 page-packed adjacency file (adjacency_pages.dat) for async reading.
    ///
    /// This is a convenience wrapper so v3 experiments can reuse the existing
    /// 4KB-block cache and IO budget plumbing unchanged.
    pub async fn open_pages(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
    ) -> io::Result<Self> {
        Self::open_with_budget_file(
            index_dir,
            dimension,
            adj_inflight,
            direct_io,
            None,
            "adjacency_pages.dat",
        )
        .await
    }

    /// Open with explicit global IO budget.
    pub async fn open_with_budget(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Option<Arc<GlobalIoBudget>>,
    ) -> io::Result<Self> {
        Self::open_with_budget_file(
            index_dir,
            dimension,
            adj_inflight,
            direct_io,
            global_budget,
            "adjacency.dat",
        )
        .await
    }

    async fn open_with_budget_file(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Option<Arc<GlobalIoBudget>>,
        filename: &str,
    ) -> io::Result<Self> {
        let path = format!("{}/{}", index_dir, filename);

        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }

        let adj_file = opts.open(&path).await?;

        Ok(Self {
            adj_file,
            adj_sem: LocalSemaphore::new(adj_inflight),
            adj_capacity: adj_inflight,
            dimension,
            global_budget,
            health: None,
            sem_wait_ns: Cell::new(0),
            device_ns: Cell::new(0),
            io_count: Cell::new(0),
        })
    }

    /// Production mode. Global budget required, health flag required.
    ///
    /// The health flag is stored for potential future use by the IO path
    /// (e.g., adaptive backoff). Currently only read by HealthChecker.
    pub async fn open_production(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Arc<GlobalIoBudget>,
        health: Arc<AtomicU8>,
    ) -> io::Result<Self> {
        Self::open_production_file(
            index_dir,
            dimension,
            adj_inflight,
            direct_io,
            global_budget,
            health,
            "adjacency.dat",
        )
        .await
    }

    /// Production-mode open for v3 page-packed adjacency file (adjacency_pages.dat).
    pub async fn open_pages_production(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Arc<GlobalIoBudget>,
        health: Arc<AtomicU8>,
    ) -> io::Result<Self> {
        Self::open_production_file(
            index_dir,
            dimension,
            adj_inflight,
            direct_io,
            global_budget,
            health,
            "adjacency_pages.dat",
        )
        .await
    }

    async fn open_production_file(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Arc<GlobalIoBudget>,
        health: Arc<AtomicU8>,
        filename: &str,
    ) -> io::Result<Self> {
        let path = format!("{}/{}", index_dir, filename);

        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }

        let adj_file = opts.open(&path).await?;

        Ok(Self {
            adj_file,
            adj_sem: LocalSemaphore::new(adj_inflight),
            adj_capacity: adj_inflight,
            dimension,
            global_budget: Some(global_budget),
            health: Some(health),
            sem_wait_ns: Cell::new(0),
            device_ns: Cell::new(0),
            io_count: Cell::new(0),
        })
    }

    /// Read one 4KB adjacency block for the given vector ID.
    /// Acquires global budget token (if configured) + local semaphore permit.
    pub async fn read_adj_block(&self, vid: u32) -> io::Result<AlignedBuf> {
        let t0 = Instant::now();
        // Two-level acquire: global device QD cap first, then per-core limit
        let _global = match &self.global_budget {
            Some(gb) => Some(acquire_global(gb).await),
            None => None,
        };
        let _permit = self.adj_sem.acquire().await;
        let t1 = Instant::now();

        let buf = AlignedBuf::new(BLOCK_SIZE);
        let offset = vid as u64 * BLOCK_SIZE as u64;

        let (result, buf) = self.adj_file.read_at(buf, offset).await;
        let t2 = Instant::now();

        self.sem_wait_ns.set(self.sem_wait_ns.get() + (t1 - t0).as_nanos() as u64);
        self.device_ns.set(self.device_ns.get() + (t2 - t1).as_nanos() as u64);
        self.io_count.set(self.io_count.get() + 1);

        let n = result?;
        if n != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: {} bytes (expected {})", n, BLOCK_SIZE),
            ));
        }
        Ok(buf)
        // _permit and _global drop here → release both levels
    }

    /// Read one 4KB adjacency block directly into a SlotPtr buffer.
    ///
    /// SlotPtr is only created by SlotStore, guaranteeing 4KB alignment and
    /// sufficient capacity. The slot is in LOADING state with pin_count=0,
    /// so no other code accesses it concurrently.
    pub(crate) async fn read_adj_block_direct(
        &self,
        vid: u32,
        dst: crate::cache::SlotPtr,
    ) -> io::Result<()> {
        let t0 = Instant::now();
        // Two-level acquire: global device QD cap first, then per-core limit
        let _global = match &self.global_budget {
            Some(gb) => Some(acquire_global(gb).await),
            None => None,
        };
        let _permit = self.adj_sem.acquire().await;
        let t1 = Instant::now();

        // Safety: SlotPtr guarantees 4KB-aligned, BLOCK_SIZE-capacity memory
        // exclusively owned by the LOADING entry.
        let buf = unsafe { SlotBuf::from_raw(dst.as_mut_ptr(), BLOCK_SIZE) };
        let offset = vid as u64 * BLOCK_SIZE as u64;

        let (result, _buf) = self.adj_file.read_at(buf, offset).await;
        let t2 = Instant::now();

        self.sem_wait_ns.set(self.sem_wait_ns.get() + (t1 - t0).as_nanos() as u64);
        self.device_ns.set(self.device_ns.get() + (t2 - t1).as_nanos() as u64);
        self.io_count.set(self.io_count.get() + 1);

        let n = result?;
        if n != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: {} bytes (expected {})", n, BLOCK_SIZE),
            ));
        }
        Ok(())
        // _permit and _global drop here → release both levels
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Total adjacency IO permits (inflight budget configured at open time).
    pub fn adj_capacity(&self) -> usize {
        self.adj_capacity
    }

    /// Number of adjacency IO permits currently available.
    pub fn available_adj_permits(&self) -> usize {
        self.adj_sem.available()
    }

    /// Current global inflight depth: `capacity - available`.
    /// Returns `None` if no global budget is configured (single-core / test mode).
    pub fn global_inflight(&self) -> Option<usize> {
        self.global_budget.as_ref().map(|gb| gb.capacity() - gb.available())
    }

    /// Snapshot and reset IO timing counters. Returns (sem_wait_ns, device_ns, io_count).
    pub fn take_io_timing(&self) -> (u64, u64, u64) {
        let s = self.sem_wait_ns.replace(0);
        let d = self.device_ns.replace(0);
        let c = self.io_count.replace(0);
        (s, d, c)
    }
}

// ---------------------------------------------------------------------------
// SlotBuf — wraps a raw *mut u8 as an IoBuf for monoio
// ---------------------------------------------------------------------------

/// Thin wrapper around a raw pointer for use as a monoio IO buffer.
/// Does NOT own the memory — caller is responsible for lifetime.
struct SlotBuf {
    ptr: *mut u8,
    capacity: usize,
    len: usize,
}

impl SlotBuf {
    /// # Safety
    /// `ptr` must be valid, 4KB-aligned, and point to at least `capacity` bytes.
    /// The memory must remain valid for the duration of the IO operation.
    unsafe fn from_raw(ptr: *mut u8, capacity: usize) -> Self {
        Self {
            ptr,
            capacity,
            len: 0,
        }
    }
}

// Safety: pointer is stable (it's a SlotStore slot), buffer won't be moved.
unsafe impl monoio::buf::IoBuf for SlotBuf {
    fn read_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn bytes_init(&self) -> usize {
        self.len
    }
}

unsafe impl monoio::buf::IoBufMut for SlotBuf {
    fn write_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn bytes_total(&mut self) -> usize {
        self.capacity
    }

    unsafe fn set_init(&mut self, pos: usize) {
        self.len = pos;
    }
}

// ---------------------------------------------------------------------------
// VectorReader — async reader for FP32 vectors from vectors.dat
// ---------------------------------------------------------------------------

/// Async reader for FP32 vectors stored on disk (vectors.dat).
///
/// Used by v4 two-stage search: graph traversal uses PQ proxy distances (DRAM),
/// then refine reads FP32 vectors from disk for exact re-scoring.
///
/// Each vector is `dim × 4` bytes at offset `vid × dim × 4`.
/// O_DIRECT requires 512-byte aligned reads, so we round up the read size.
pub struct VectorReader {
    file: File,
    dim: usize,
    /// Bytes per vector (dim * 4).
    vec_bytes: usize,
}

impl VectorReader {
    pub async fn open(index_dir: &str, dim: usize, direct_io: bool) -> io::Result<Self> {
        let path = format!("{}/vectors.dat", index_dir);
        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(&path).await?;
        let vec_bytes = dim * 4;
        Ok(Self { file, dim, vec_bytes })
    }

    /// Read one FP32 vector from disk by VID. Returns `dim` floats.
    pub async fn read_vector(&self, vid: u32) -> io::Result<Vec<f32>> {
        // O_DIRECT: offset must be 512-aligned. vid * vec_bytes may not be.
        // Compute the aligned offset and the intra-block offset.
        let raw_offset = vid as u64 * self.vec_bytes as u64;
        let align_offset = raw_offset & !511;
        let intra_offset = (raw_offset - align_offset) as usize;
        let total_read = ((intra_offset + self.vec_bytes) + 511) & !511;

        let buf = AlignedBuf::new(total_read);
        let (result, buf) = self.file.read_at(buf, align_offset).await;
        let n = result?;
        if n < intra_offset + self.vec_bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short vector read: {} bytes (need {})", n, intra_offset + self.vec_bytes),
            ));
        }

        // Extract the f32 vector from the aligned buffer
        let bytes = buf.as_slice();
        let vec_bytes = &bytes[intra_offset..intra_offset + self.vec_bytes];
        let f32_slice = unsafe {
            std::slice::from_raw_parts(vec_bytes.as_ptr() as *const f32, self.dim)
        };
        Ok(f32_slice.to_vec())
    }

    /// Read one FP32 vector from disk and compute cosine distance against `query`
    /// directly on the aligned IO buffer — avoids allocating a `Vec<f32>` per read.
    ///
    /// Returns the cosine distance (1 - cos_sim). `query_norm_sq` = Σ(q_i²).
    pub async fn read_cosine_distance(
        &self,
        vid: u32,
        query: &[f32],
        query_norm_sq: f32,
    ) -> io::Result<f32> {
        debug_assert_eq!(
            query.len(),
            self.dim,
            "query dim mismatch: query.len={} dim={}",
            query.len(),
            self.dim
        );

        let raw_offset = vid as u64 * self.vec_bytes as u64;
        let align_offset = raw_offset & !511;
        let intra_offset = (raw_offset - align_offset) as usize;
        let total_read = ((intra_offset + self.vec_bytes) + 511) & !511;

        let buf = AlignedBuf::new(total_read);
        let (result, buf) = self.file.read_at(buf, align_offset).await;
        let n = result?;
        if n < intra_offset + self.vec_bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short vector read: {} bytes (need {})", n, intra_offset + self.vec_bytes),
            ));
        }

        let bytes = buf.as_slice();
        let vec_bytes = &bytes[intra_offset..intra_offset + self.vec_bytes];
        debug_assert_eq!(intra_offset & 3, 0, "vector intra_offset must be 4B aligned");
        debug_assert_eq!(self.vec_bytes & 3, 0, "vec_bytes must be multiple of 4");
        debug_assert_eq!(
            (vec_bytes.as_ptr() as usize) & 3,
            0,
            "vector pointer must be 4B aligned"
        );
        let v = unsafe {
            std::slice::from_raw_parts(vec_bytes.as_ptr() as *const f32, self.dim)
        };

        // Compute cosine distance in-place on the IO buffer
        let mut dot = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..self.dim {
            dot += query[i] * v[i];
            norm_b += v[i] * v[i];
        }
        let denom = (query_norm_sq * norm_b).sqrt();
        let dist = if denom == 0.0 { 1.0 } else { 1.0 - dot / denom };
        Ok(dist)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Int8VectorReader — async reader for int8-quantized vectors
// ---------------------------------------------------------------------------

/// Async reader for int8-quantized vectors stored on disk (vectors_int8.dat).
///
/// Each vector is `dim` bytes (1 byte per dimension) at offset `vid × dim`.
/// Encoding: round(f32_val * 127.0), clamped to [-127, 127].
/// For dim=768: 768 bytes/vec vs FP32's 3072 bytes/vec = 4× smaller.
///
/// O_DIRECT requires 512-byte aligned reads, handled automatically.
pub struct Int8VectorReader {
    file: File,
    dim: usize,
    /// Bytes per vector (dim * 1).
    vec_bytes: usize,
}

impl Int8VectorReader {
    pub async fn open(dir: &str, dim: usize, direct_io: bool) -> io::Result<Self> {
        let path = format!("{}/vectors_int8.dat", dir);
        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(&path).await?;
        Ok(Self { file, dim, vec_bytes: dim })
    }

    /// Read one int8 vector from disk and compute cosine distance against `query`.
    ///
    /// Computes directly on the IO buffer — no heap allocation.
    /// The 1/127 scale factor cancels in cosine: cos(q, c/127) = dot(q,c) / (||q||·||c||).
    /// So we work entirely in the integer code domain (no division).
    ///
    /// Returns `(cosine_distance, bytes_read)` where bytes_read is the actual
    /// IO size submitted to the kernel (for honest IO accounting).
    pub async fn read_cosine_distance(
        &self,
        vid: u32,
        query: &[f32],
        query_norm_sq: f32,
    ) -> io::Result<(f32, usize)> {
        debug_assert_eq!(query.len(), self.dim);

        let raw_offset = vid as u64 * self.vec_bytes as u64;
        let align_offset = raw_offset & !511;
        let intra_offset = (raw_offset - align_offset) as usize;
        let total_read = ((intra_offset + self.vec_bytes) + 511) & !511;

        let buf = AlignedBuf::new(total_read);
        let (result, buf) = self.file.read_at(buf, align_offset).await;
        let n = result?;
        if n < intra_offset + self.vec_bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short int8 vector read: {} bytes (need {})", n, intra_offset + self.vec_bytes),
            ));
        }

        let bytes = buf.as_slice();
        let codes = &bytes[intra_offset..intra_offset + self.vec_bytes];

        // Cosine distance using integer codes directly.
        // cos(q, code/127) = dot(q, code) / (||q|| * ||code||)
        // The 1/127 cancels: dot(q, code/127) / ||code/127|| = dot(q, code) / ||code||.
        // query_norm_sq is precomputed as Σ(q_i²) in f32 domain.
        let mut dot_qc = 0.0f32;
        let mut norm_c_sq = 0.0f32;
        for i in 0..self.dim {
            let ci = codes[i] as i8 as f32;
            dot_qc += query[i] * ci;
            norm_c_sq += ci * ci;
        }
        let denom = (query_norm_sq * norm_c_sq).sqrt();
        let dist = if denom == 0.0 { 1.0 } else { 1.0 - dot_qc / denom };
        Ok((dist, total_read))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes per vector on disk (= dim for int8).
    pub fn vec_bytes(&self) -> usize {
        self.vec_bytes
    }
}

// ---------------------------------------------------------------------------
// FP16 cosine distance kernel (F16C + AVX2 / scalar fallback)
// ---------------------------------------------------------------------------

/// Compute dot(query, fp16_vec) and norm_sq(fp16_vec) in a single pass.
/// `fp16_bytes`: raw LE bytes of the f16 vector (dim * 2 bytes).
/// `query`: f32 query vector (dim elements).
/// Returns (dot_product, norm_b_sq).
#[inline]
fn cosine_dot_fp16(fp16_bytes: &[u8], query: &[f32], dim: usize) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_dot_fp16_f16c(fp16_bytes, query, dim) };
        }
    }
    cosine_dot_fp16_scalar(fp16_bytes, query, dim)
}

fn cosine_dot_fp16_scalar(fp16_bytes: &[u8], query: &[f32], dim: usize) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..dim {
        let lo = fp16_bytes[i * 2] as u16;
        let hi = fp16_bytes[i * 2 + 1] as u16;
        let half = lo | (hi << 8);
        let v = divergence_storage::f16_to_f32(half);
        dot += query[i] * v;
        norm_b += v * v;
    }
    (dot, norm_b)
}

/// F16C + FMA kernel: VCVTPH2PS converts 8 f16→f32, then FMA for dot+norm.
/// Processes 8 elements per iteration (256-bit SIMD).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2,fma")]
unsafe fn cosine_dot_fp16_f16c(fp16_bytes: &[u8], query: &[f32], dim: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_acc = _mm256_setzero_ps();

    let chunks = dim / 8;
    let fp16_ptr = fp16_bytes.as_ptr() as *const __m128i;
    let query_ptr = query.as_ptr();

    for i in 0..chunks {
        // Load 8 × f16 (128 bits) and convert to 8 × f32 (256 bits)
        let half8 = _mm_loadu_si128(fp16_ptr.add(i));
        let v8 = _mm256_cvtph_ps(half8);

        // Load 8 × f32 query values
        let q8 = _mm256_loadu_ps(query_ptr.add(i * 8));

        // dot += q * v, norm += v * v
        dot_acc = _mm256_fmadd_ps(q8, v8, dot_acc);
        norm_acc = _mm256_fmadd_ps(v8, v8, norm_acc);
    }

    // Horizontal sum of 8-wide accumulators
    let dot_sum = hsum256_ps(dot_acc);
    let norm_sum = hsum256_ps(norm_acc);

    // Handle remainder (dim % 8)
    let tail_start = chunks * 8;
    let mut dot_tail = 0.0f32;
    let mut norm_tail = 0.0f32;
    for i in tail_start..dim {
        let lo = fp16_bytes[i * 2] as u16;
        let hi = fp16_bytes[i * 2 + 1] as u16;
        let half = lo | (hi << 8);
        let v = divergence_storage::f16_to_f32(half);
        dot_tail += query[i] * v;
        norm_tail += v * v;
    }

    (dot_sum + dot_tail, norm_sum + norm_tail)
}

/// Horizontal sum of __m256 (8 floats → 1 float).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
    let hi128 = _mm256_extractf128_ps(v, 1); // [a4 a5 a6 a7]
    let lo128 = _mm256_castps256_ps128(v);    // [a0 a1 a2 a3]
    let sum128 = _mm_add_ps(lo128, hi128);    // [a0+a4, a1+a5, a2+a6, a3+a7]
    let shuf = _mm_movehdup_ps(sum128);       // [a1+a5, a1+a5, a3+a7, a3+a7]
    let sums = _mm_add_ps(sum128, shuf);      // [s01, _, s23, _]
    let shuf2 = _mm_movehl_ps(sums, sums);   // [s23, _, ...]
    let final_sum = _mm_add_ss(sums, shuf2);  // [s01+s23]
    _mm_cvtss_f32(final_sum)
}

// ---------------------------------------------------------------------------
// Fp16VectorReader — async reader for half-precision vectors
// ---------------------------------------------------------------------------

/// Async reader for FP16 vectors stored on disk (vectors_fp16.dat).
///
/// Each vector is `dim × 2` bytes at offset `vid × dim × 2`.
/// FP16 gives ~3 decimal digits of precision — much better than int8 (~2.1).
/// For dim=768: 1536 bytes/vec vs FP32's 3072 bytes/vec = 2× smaller IO.
///
/// O_DIRECT requires 512-byte aligned reads, handled automatically.
pub struct Fp16VectorReader {
    file: File,
    dim: usize,
    /// Bytes per vector (dim * 2).
    vec_bytes: usize,
}

impl Fp16VectorReader {
    pub async fn open(dir: &str, dim: usize, direct_io: bool) -> io::Result<Self> {
        let path = format!("{}/vectors_fp16.dat", dir);
        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(&path).await?;
        Ok(Self { file, dim, vec_bytes: dim * 2 })
    }

    /// Read one FP16 vector from disk and compute cosine distance against `query`.
    ///
    /// Converts f16 → f32 on-the-fly and computes cosine distance in-place
    /// on the IO buffer — no heap allocation per read.
    ///
    /// Returns `(cosine_distance, bytes_read)` where bytes_read is the actual
    /// IO size submitted to the kernel (for honest IO accounting).
    pub async fn read_cosine_distance(
        &self,
        vid: u32,
        query: &[f32],
        query_norm_sq: f32,
    ) -> io::Result<(f32, usize)> {
        debug_assert_eq!(query.len(), self.dim);

        let raw_offset = vid as u64 * self.vec_bytes as u64;
        let align_offset = raw_offset & !511;
        let intra_offset = (raw_offset - align_offset) as usize;
        let total_read = ((intra_offset + self.vec_bytes) + 511) & !511;

        let buf = AlignedBuf::new(total_read);
        let (result, buf) = self.file.read_at(buf, align_offset).await;
        let n = result?;
        if n < intra_offset + self.vec_bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short fp16 vector read: {} bytes (need {})", n, intra_offset + self.vec_bytes),
            ));
        }

        let bytes = buf.as_slice();
        let fp16_bytes = &bytes[intra_offset..intra_offset + self.vec_bytes];
        debug_assert_eq!(intra_offset & 1, 0, "fp16 intra_offset must be 2B aligned");

        // Compute cosine distance: convert f16→f32 and fused dot+norm
        let (dot, norm_b) = cosine_dot_fp16(fp16_bytes, query, self.dim);
        let denom = (query_norm_sq * norm_b).sqrt();
        let dist = if denom == 0.0 { 1.0 } else { 1.0 - dot / denom };
        Ok((dist, total_read))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vec_bytes(&self) -> usize {
        self.vec_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semaphore_basic() {
        let sem = LocalSemaphore::new(2);

        // Can acquire twice
        assert!(sem.try_acquire());
        assert!(sem.try_acquire());
        // Third fails
        assert!(!sem.try_acquire());

        // Release one
        sem.release();
        assert!(sem.try_acquire());
    }
}
