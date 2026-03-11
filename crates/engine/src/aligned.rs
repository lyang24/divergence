//! 4KB-aligned heap buffer implementing monoio's IoBuf/IoBufMut.
//!
//! Required for O_DIRECT IO — buffer pointer, file offset, and read length
//! must meet the kernel/filesystem alignment rules (typically 512B on NVMe).
//! Standard `Vec<u8>` does not guarantee suitable alignment.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

// 4KB alignment is always safe and also works for 512B-aligned direct IO.
const ALIGNMENT: usize = 4096;
// Read length/offset alignment for O_DIRECT (common case). We still allocate
// with 4KB alignment, but we only round the *size* up to 512 so vector reads
// can be 1KB/3KB instead of always 4KB.
const IO_SIZE_ALIGNMENT: usize = 512;

/// 4KB-aligned buffer for O_DIRECT IO.
///
/// - Allocates via `alloc_zeroed` with 4096 alignment
/// - Pointer is stable (no realloc) — satisfies IoBuf contract
/// - `Unpin + 'static` as required by monoio
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,      // initialized bytes
    capacity: usize,  // allocated (rounded up to 4096)
}

impl AlignedBuf {
    /// Allocate a new buffer with at least `size` bytes, rounded up to 512.
    pub fn new(size: usize) -> Self {
        let capacity = round_up(size);
        let layout = Layout::from_size_align(capacity, ALIGNMENT).expect("invalid layout");

        // Safety: layout has non-zero size (round_up guarantees >= 512)
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");

        Self {
            ptr,
            len: 0,
            capacity,
        }
    }

    /// Create from an existing byte slice (copies data).
    pub fn from_slice(data: &[u8]) -> Self {
        let mut buf = Self::new(data.len());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr.as_ptr(), data.len());
        }
        buf.len = data.len();
        buf
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, ALIGNMENT).unwrap();
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// Safety: pointer is stable (no realloc), Unpin + 'static.
unsafe impl monoio::buf::IoBuf for AlignedBuf {
    fn read_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len
    }
}

// Safety: pointer is stable, bytes_total returns full capacity for kernel writes.
unsafe impl monoio::buf::IoBufMut for AlignedBuf {
    fn write_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn bytes_total(&mut self) -> usize {
        self.capacity
    }

    unsafe fn set_init(&mut self, pos: usize) {
        self.len = pos;
    }
}

fn round_up(size: usize) -> usize {
    let s = if size == 0 { IO_SIZE_ALIGNMENT } else { size };
    (s + IO_SIZE_ALIGNMENT - 1) & !(IO_SIZE_ALIGNMENT - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_check() {
        let buf = AlignedBuf::new(4096);
        assert_eq!(buf.ptr.as_ptr() as usize % 4096, 0);
        assert_eq!(buf.capacity(), 4096);
    }

    #[test]
    fn capacity_rounds_up() {
        let buf = AlignedBuf::new(1);
        assert_eq!(buf.capacity(), 512);

        let buf = AlignedBuf::new(513);
        assert_eq!(buf.capacity(), 1024);

        let buf = AlignedBuf::new(0);
        assert_eq!(buf.capacity(), 512);
    }

    #[test]
    fn from_slice_roundtrip() {
        let data = b"hello world";
        let buf = AlignedBuf::from_slice(data);
        assert_eq!(buf.len(), data.len());
        assert_eq!(buf.as_slice(), data);
        assert_eq!(buf.ptr.as_ptr() as usize % 4096, 0);
    }
}
