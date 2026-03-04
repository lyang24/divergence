use std::cmp::Ordering;

use divergence_core::VectorId;

/// A scored vector ID. Lower distance = better.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScoredId {
    pub distance: f32,
    pub id: VectorId,
}

impl PartialEq for ScoredId {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for ScoredId {}

impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.0.cmp(&other.id.0))
    }
}

/// Fixed-capacity max-heap for the result set (W in the HNSW paper).
/// Keeps the best `capacity` items by distance. Worst item is at the top.
pub struct FixedCapacityHeap {
    data: Vec<ScoredId>,
    capacity: usize,
}

impl FixedCapacityHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn clear(&mut self, new_capacity: usize) {
        self.data.clear();
        self.capacity = new_capacity;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Peek at the worst (furthest) element. O(1).
    pub fn furthest(&self) -> Option<&ScoredId> {
        self.data.first()
    }

    /// Push an item. If at capacity and item is worse than the worst, rejected.
    /// If at capacity and item is better, replaces the worst. O(log n).
    pub fn push(&mut self, item: ScoredId) {
        if self.data.len() < self.capacity {
            self.data.push(item);
            self.sift_up(self.data.len() - 1);
        } else if !self.data.is_empty() && item.distance < self.data[0].distance {
            self.data[0] = item;
            self.sift_down(0);
        }
    }

    /// Drain all items sorted by distance ascending (best first).
    pub fn into_sorted_vec(mut self) -> Vec<ScoredId> {
        let mut result = Vec::with_capacity(self.data.len());
        while !self.data.is_empty() {
            let last = self.data.len() - 1;
            self.data.swap(0, last);
            result.push(self.data.pop().unwrap());
            if !self.data.is_empty() {
                self.sift_down(0);
            }
        }
        result.reverse();
        result
    }

    /// Drain sorted, reusable (doesn't consume self, just clears).
    pub fn drain_sorted(&mut self) -> Vec<ScoredId> {
        let mut result = Vec::with_capacity(self.data.len());
        while !self.data.is_empty() {
            let last = self.data.len() - 1;
            self.data.swap(0, last);
            result.push(self.data.pop().unwrap());
            if !self.data.is_empty() {
                self.sift_down(0);
            }
        }
        result.reverse();
        result
    }

    // Max-heap: parent >= children (by distance)
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx].distance > self.data[parent].distance {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;

            if left < len && self.data[left].distance > self.data[largest].distance {
                largest = left;
            }
            if right < len && self.data[right].distance > self.data[largest].distance {
                largest = right;
            }
            if largest != idx {
                self.data.swap(idx, largest);
                idx = largest;
            } else {
                break;
            }
        }
    }
}

/// Min-heap for the candidate set (C in the HNSW paper).
/// Nearest candidate is at the top.
pub struct CandidateHeap {
    data: Vec<ScoredId>,
}

impl CandidateHeap {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(64),
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Push a candidate. O(log n).
    pub fn push(&mut self, item: ScoredId) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    /// Pop up to `buf.len()` nearest candidates into `buf`, then push them all back.
    /// Returns the number of items written. O(n log N) — negligible at n<=8 in IO-bound loops.
    /// Unlike raw heap-array slicing, this returns the *actual* nearest candidates.
    pub fn peek_nearest(&mut self, buf: &mut [ScoredId]) -> usize {
        let n = buf.len().min(self.data.len());
        for i in 0..n {
            buf[i] = self.pop().unwrap();
        }
        for i in 0..n {
            self.push(buf[i]);
        }
        n
    }

    /// Pop the nearest candidate. O(log n).
    pub fn pop(&mut self) -> Option<ScoredId> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let item = self.data.pop().unwrap();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        Some(item)
    }

    // Min-heap: parent <= children (by distance)
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx].distance < self.data[parent].distance {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < len && self.data[left].distance < self.data[smallest].distance {
                smallest = left;
            }
            if right < len && self.data[right].distance < self.data[smallest].distance {
                smallest = right;
            }
            if smallest != idx {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(dist: f32, id: u32) -> ScoredId {
        ScoredId {
            distance: dist,
            id: VectorId(id),
        }
    }

    #[test]
    fn fixed_heap_keeps_nearest() {
        let mut heap = FixedCapacityHeap::new(3);
        heap.push(sid(5.0, 0));
        heap.push(sid(3.0, 1));
        heap.push(sid(7.0, 2));
        // Full. Now push something better than worst (7.0).
        heap.push(sid(1.0, 3));
        assert_eq!(heap.len(), 3);
        // Worst should now be 5.0
        assert_eq!(heap.furthest().unwrap().distance, 5.0);
    }

    #[test]
    fn fixed_heap_rejects_worse() {
        let mut heap = FixedCapacityHeap::new(2);
        heap.push(sid(1.0, 0));
        heap.push(sid(2.0, 1));
        heap.push(sid(3.0, 2)); // worse than worst (2.0), rejected
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.furthest().unwrap().distance, 2.0);
    }

    #[test]
    fn fixed_heap_sorted_drain() {
        let mut heap = FixedCapacityHeap::new(5);
        heap.push(sid(5.0, 0));
        heap.push(sid(1.0, 1));
        heap.push(sid(3.0, 2));
        heap.push(sid(2.0, 3));
        heap.push(sid(4.0, 4));
        let sorted = heap.into_sorted_vec();
        let dists: Vec<f32> = sorted.iter().map(|s| s.distance).collect();
        assert_eq!(dists, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn candidate_heap_pops_nearest() {
        let mut heap = CandidateHeap::new();
        heap.push(sid(5.0, 0));
        heap.push(sid(1.0, 1));
        heap.push(sid(3.0, 2));
        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
        assert_eq!(heap.pop().unwrap().distance, 5.0);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn candidate_heap_peek_nearest() {
        let mut heap = CandidateHeap::new();
        heap.push(sid(5.0, 0));
        heap.push(sid(1.0, 1));
        heap.push(sid(3.0, 2));
        heap.push(sid(2.0, 3));

        // Peek top 2
        let mut buf = [ScoredId::default(); 4];
        let n = heap.peek_nearest(&mut buf[..2]);
        assert_eq!(n, 2);
        assert_eq!(buf[0].distance, 1.0);
        assert_eq!(buf[1].distance, 2.0);

        // Heap should still have all 4 items
        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
        assert_eq!(heap.pop().unwrap().distance, 5.0);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn candidate_heap_peek_nearest_larger_than_heap() {
        let mut heap = CandidateHeap::new();
        heap.push(sid(3.0, 0));
        heap.push(sid(1.0, 1));

        let mut buf = [ScoredId::default(); 8];
        let n = heap.peek_nearest(&mut buf);
        assert_eq!(n, 2);
        assert_eq!(buf[0].distance, 1.0);
        assert_eq!(buf[1].distance, 3.0);

        // Heap intact
        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
    }
}
