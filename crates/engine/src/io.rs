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
use std::sync::atomic::{AtomicUsize, Ordering};
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

    /// Open with explicit global IO budget.
    pub async fn open_with_budget(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
        global_budget: Option<Arc<GlobalIoBudget>>,
    ) -> io::Result<Self> {
        let adj_path = format!("{}/adjacency.dat", index_dir);

        let mut opts = monoio::fs::OpenOptions::new();
        opts.read(true);
        if direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }

        let adj_file = opts.open(&adj_path).await?;

        Ok(Self {
            adj_file,
            adj_sem: LocalSemaphore::new(adj_inflight),
            adj_capacity: adj_inflight,
            dimension,
            global_budget,
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
