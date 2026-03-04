//! IO driver with separate adj/vec inflight budgets.
//!
//! IoDriver opens adjacency.dat (and optionally vectors.dat) with O_DIRECT
//! and provides async read methods gated by per-category semaphores.
//!
//! LocalSemaphore is a single-threaded async semaphore (no Send needed since
//! monoio is thread-per-core).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

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
// IoDriver
// ---------------------------------------------------------------------------

/// Async IO driver for reading adjacency blocks from disk.
///
/// Uses O_DIRECT for zero-copy NVMe reads. Inflight reads are bounded
/// by the adjacency semaphore.
pub struct IoDriver {
    adj_file: File,
    adj_sem: LocalSemaphore,
    adj_capacity: usize,
    dimension: usize,
}

impl IoDriver {
    /// Open index files for async reading.
    ///
    /// `direct_io`: set to false for tmpfs/tests (O_DIRECT doesn't work on tmpfs).
    pub async fn open(
        index_dir: &str,
        dimension: usize,
        adj_inflight: usize,
        direct_io: bool,
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
        })
    }

    /// Read one 4KB adjacency block for the given vector ID.
    /// Acquires a semaphore permit, reads, releases on return.
    pub async fn read_adj_block(&self, vid: u32) -> io::Result<AlignedBuf> {
        let _permit = self.adj_sem.acquire().await;

        let buf = AlignedBuf::new(BLOCK_SIZE);
        let offset = vid as u64 * BLOCK_SIZE as u64;

        let (result, buf) = self.adj_file.read_at(buf, offset).await;
        let n = result?;
        if n != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: {} bytes (expected {})", n, BLOCK_SIZE),
            ));
        }
        Ok(buf)
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
        let _permit = self.adj_sem.acquire().await;

        // Safety: SlotPtr guarantees 4KB-aligned, BLOCK_SIZE-capacity memory
        // exclusively owned by the LOADING entry.
        let buf = unsafe { SlotBuf::from_raw(dst.as_mut_ptr(), BLOCK_SIZE) };
        let offset = vid as u64 * BLOCK_SIZE as u64;

        let (result, _buf) = self.adj_file.read_at(buf, offset).await;
        let n = result?;
        if n != BLOCK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: {} bytes (expected {})", n, BLOCK_SIZE),
            ));
        }
        Ok(())
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
