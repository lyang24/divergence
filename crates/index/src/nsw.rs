//! Flat NSW index — FlatNav ("Down with the Hierarchy").
//!
//! Single-layer navigable small world graph. No hierarchy.
//! Hub nodes naturally form routing highways in high-dimensional spaces.
//!
//! NswBuilder: parallel construction with per-node RwLock.
//! NswIndex: immutable reader with flattened graph + hub entry set.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};

use divergence_core::distance::{create_distance_computer, DistanceComputer};
use divergence_core::{MetricType, VectorId};

use crate::heap::{CandidateHeap, FixedCapacityHeap, ScoredId};
use crate::visited::VisitedPool;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NswConfig {
    pub m_max: usize,            // max neighbors per node (e.g., 32)
    pub ef_construction: usize,  // beam width during build (e.g., 200)
    pub num_entry_points: usize, // hub entry set size (default 64)
}

impl NswConfig {
    pub fn new(m_max: usize, ef_construction: usize) -> Self {
        Self {
            m_max,
            ef_construction,
            num_entry_points: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguous slot layout — uses u32 backing for alignment safety
// ---------------------------------------------------------------------------

/// Fixed-size slot for flat graph. Backed by Vec<u32> for guaranteed alignment.
///
/// Logical layout per slot (in u32 units):
///   [0]: neighbor_count
///   [1..1+m_max]: neighbor VIDs
///
/// slot_size_u32 = 1 + m_max
struct SlotLayout {
    slot_size_u32: usize,
    m_max: usize,
}

impl SlotLayout {
    fn new(m_max: usize) -> Self {
        Self {
            slot_size_u32: 1 + m_max,
            m_max,
        }
    }

    #[inline]
    fn offset(&self, vid: u32) -> usize {
        vid as usize * self.slot_size_u32
    }

    fn get_neighbor_count(&self, data: &[u32], vid: u32) -> usize {
        data[self.offset(vid)] as usize
    }

    fn set_neighbor_count(&self, data: &mut [u32], vid: u32, count: usize) {
        data[self.offset(vid)] = count as u32;
    }

    fn get_neighbors<'a>(&self, data: &'a [u32], vid: u32) -> &'a [u32] {
        let off = self.offset(vid) + 1;
        let count = self.get_neighbor_count(data, vid);
        &data[off..off + count]
    }

    fn set_neighbors(&self, data: &mut [u32], vid: u32, neighbors: &[u32]) {
        debug_assert!(neighbors.len() <= self.m_max);
        self.set_neighbor_count(data, vid, neighbors.len());
        let off = self.offset(vid) + 1;
        data[off..off + neighbors.len()].copy_from_slice(neighbors);
    }
}

// ---------------------------------------------------------------------------
// Aligned interior-mutable buffers (UnsafeCell, not raw pointer casts)
// ---------------------------------------------------------------------------

/// UnsafeCell wrapper for Vec<u32>. Write-guarded by external RwLock.
struct SyncU32Vec(UnsafeCell<Vec<u32>>);

// Safety: access is synchronized by per-vertex RwLock in NswBuilder.
unsafe impl Sync for SyncU32Vec {}

impl SyncU32Vec {
    fn new(data: Vec<u32>) -> Self {
        Self(UnsafeCell::new(data))
    }
    /// Safety: caller must hold appropriate lock.
    #[inline]
    unsafe fn as_slice(&self) -> &[u32] {
        unsafe { &*self.0.get() }
    }
    /// Safety: caller must hold write lock for the accessed region.
    #[inline]
    unsafe fn as_mut_slice(&self) -> &mut [u32] {
        unsafe { &mut *self.0.get() }
    }
}

/// UnsafeCell wrapper for Vec<f32>. Each VID region is write-once.
struct SyncF32Vec(UnsafeCell<Vec<f32>>);

// Safety: each VID's region is written exactly once before any reads.
// Reads to a VID only happen after that VID's insert has stored its vector.
// Concurrent writes to distinct VIDs touch disjoint memory.
unsafe impl Sync for SyncF32Vec {}

impl SyncF32Vec {
    fn new(data: Vec<f32>) -> Self {
        Self(UnsafeCell::new(data))
    }
    #[inline]
    unsafe fn as_slice(&self) -> &[f32] {
        unsafe { &*self.0.get() }
    }
    /// Safety: caller must ensure exclusive access to the written region.
    #[inline]
    unsafe fn write_region(&self, offset: usize, src: &[f32]) {
        unsafe {
            let ptr = self.0.get();
            let slice = std::slice::from_raw_parts_mut(
                (*ptr).as_mut_ptr().add(offset),
                src.len(),
            );
            slice.copy_from_slice(src);
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Mutable flat NSW graph for parallel construction.
pub struct NswBuilder {
    config: NswConfig,
    dimension: usize,
    metric: MetricType,
    distance: Box<dyn DistanceComputer>,

    // Vector storage: flat, pre-allocated. Write-once per VID.
    vectors: SyncF32Vec,
    num_inserted: AtomicU32,
    capacity: usize,

    // Graph: contiguous u32 array with per-vertex RwLock.
    graph_data: SyncU32Vec,
    locks: Vec<RwLock<()>>,
    slot_layout: SlotLayout,

    // Global state: entry point for incremental construction
    entry_point: Mutex<Option<VectorId>>,

    /// Bitset: one bit per VID. Atomic for lock-free concurrent insert-once check.
    /// Uses AtomicU64 for fewer atomic objects and better cacheline alignment.
    inserted: Vec<AtomicU64>,

    // Thread-local resources
    visited_pool: VisitedPool,
}

impl NswBuilder {
    pub fn new(config: NswConfig, dimension: usize, metric: MetricType, capacity: usize) -> Self {
        let slot_layout = SlotLayout::new(config.m_max);
        let graph_size = capacity * slot_layout.slot_size_u32;

        Self {
            distance: create_distance_computer(metric),
            config,
            dimension,
            metric,
            vectors: SyncF32Vec::new(vec![0.0f32; capacity * dimension]),
            num_inserted: AtomicU32::new(0),
            capacity,
            graph_data: SyncU32Vec::new(vec![0u32; graph_size]),
            locks: (0..capacity).map(|_| RwLock::new(())).collect(),
            slot_layout,
            entry_point: Mutex::new(None),
            inserted: (0..((capacity + 63) / 64)).map(|_| AtomicU64::new(0)).collect(),
            visited_pool: VisitedPool::new(capacity),
        }
    }

    /// Insert a vector. Thread-safe — can be called from multiple threads
    /// concurrently, provided each VID is inserted exactly once.
    ///
    /// # Panics
    /// Panics if `vid >= capacity` or `vector.len() != dimension`.
    pub fn insert(&self, id: VectorId, vector: &[f32]) {
        assert_eq!(vector.len(), self.dimension, "vector dimension mismatch");
        let vid = id.0;
        assert!(
            (vid as usize) < self.capacity,
            "VID {} out of bounds (capacity={})",
            vid,
            self.capacity
        );

        // Insert-once contract: duplicate VID = data race on SyncF32Vec (UB).
        let word_idx = vid as usize / 64;
        let bit_mask = 1u64 << (vid as usize % 64);
        let prev = self.inserted[word_idx].fetch_or(bit_mask, Ordering::AcqRel);
        assert!(
            prev & bit_mask == 0,
            "VID {} inserted twice — build-once immutable index requires each VID inserted exactly once",
            vid
        );

        // Store vector (write-once per VID, disjoint regions)
        let voff = vid as usize * self.dimension;
        unsafe { self.vectors.write_region(voff, vector) };

        self.num_inserted.fetch_add(1, Ordering::Relaxed);

        // Read current entry point
        let current_ep = {
            let ep_guard = self.entry_point.lock();
            *ep_guard
        };

        // First vector — set as entry point and return
        if current_ep.is_none() {
            let mut ep_guard = self.entry_point.lock();
            if ep_guard.is_none() {
                *ep_guard = Some(id);
                return;
            }
            // Another thread beat us — fall through to normal insert
            drop(ep_guard);
            let current_ep = {
                let ep_guard = self.entry_point.lock();
                ep_guard.unwrap()
            };
            self.connect(vid, vector, current_ep);
            return;
        }

        self.connect(vid, vector, current_ep.unwrap());
    }

    /// Core connection logic: single beam search + bidirectional linking.
    fn connect(&self, vid: u32, vector: &[f32], ep: VectorId) {
        let mut visited = self.visited_pool.get();
        let mut nearest = FixedCapacityHeap::new(self.config.ef_construction);
        let mut candidates = CandidateHeap::new();
        let mut neighbor_buf: Vec<u32> = Vec::new();

        self.search_layer(
            vector,
            &[ep],
            self.config.ef_construction,
            &mut visited,
            &mut nearest,
            &mut candidates,
            &mut neighbor_buf,
        );

        let scored = nearest.drain_sorted();
        let neighbors = self.select_neighbors_heuristic(&scored, self.config.m_max);

        self.set_neighbors(vid, &neighbors);
        for &nbr in &neighbors {
            self.add_bidirectional_link(nbr, vid);
        }
    }

    /// Beam search on the flat graph.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[VectorId],
        ef: usize,
        visited: &mut crate::visited::VisitedListHandle<'_>,
        nearest: &mut FixedCapacityHeap,
        candidates: &mut CandidateHeap,
        neighbor_buf: &mut Vec<u32>,
    ) {
        nearest.clear(ef);
        candidates.clear();

        for &ep in entry_points {
            let d = self.distance.distance(query, self.get_vector(ep));
            let scored = ScoredId {
                distance: d,
                id: ep,
            };
            nearest.push(scored);
            candidates.push(scored);
            visited.check_and_mark(ep.0);
        }

        while let Some(candidate) = candidates.pop() {
            if let Some(furthest) = nearest.furthest() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            self.read_neighbors_into(candidate.id, neighbor_buf);
            for &nbr_raw in neighbor_buf.iter() {
                if visited.check_and_mark(nbr_raw) {
                    continue;
                }
                let nbr = VectorId(nbr_raw);
                let d = self.distance.distance(query, self.get_vector(nbr));
                let dominated =
                    nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId {
                        distance: d,
                        id: nbr,
                    };
                    candidates.push(scored);
                    nearest.push(scored);
                }
            }
        }
    }

    /// Heuristic neighbor selection (Algorithm 4 from HNSW paper).
    fn select_neighbors_heuristic(&self, candidates: &[ScoredId], m: usize) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);

        for &ScoredId {
            id: cand_id,
            distance: cand_dist,
        } in candidates
        {
            if selected.len() >= m {
                break;
            }
            let cand_vec = self.get_vector(cand_id);
            let mut good = true;
            for &existing in &selected {
                let existing_vec = self.get_vector(VectorId(existing));
                let dist_to_existing = self.distance.distance(cand_vec, existing_vec);
                if dist_to_existing < cand_dist {
                    good = false;
                    break;
                }
            }
            if good {
                selected.push(cand_id.0);
            }
        }
        selected
    }

    /// Get vector for a VectorId.
    #[inline]
    fn get_vector(&self, id: VectorId) -> &[f32] {
        let off = id.0 as usize * self.dimension;
        unsafe { &self.vectors.as_slice()[off..off + self.dimension] }
    }

    /// Read neighbors into a reusable buffer (no allocation on hot path).
    fn read_neighbors_into(&self, id: VectorId, buf: &mut Vec<u32>) {
        buf.clear();
        let _lock = self.locks[id.0 as usize].read();
        let data = unsafe { self.graph_data.as_slice() };
        buf.extend_from_slice(self.slot_layout.get_neighbors(data, id.0));
    }

    /// Set neighbors for a vertex.
    fn set_neighbors(&self, vid: u32, neighbors: &[u32]) {
        let _lock = self.locks[vid as usize].write();
        let data = unsafe { self.graph_data.as_mut_slice() };
        self.slot_layout.set_neighbors(data, vid, neighbors);
    }

    /// Add a bidirectional link: nbr -> vid. If nbr exceeds m_max, prune.
    fn add_bidirectional_link(&self, nbr: u32, vid: u32) {
        let _lock = self.locks[nbr as usize].write();
        let data = unsafe { self.graph_data.as_mut_slice() };
        let existing = self.slot_layout.get_neighbors(data, nbr);
        // Skip if already connected
        if existing.contains(&vid) {
            return;
        }
        let mut neighbors: Vec<u32> = existing.to_vec();
        neighbors.push(vid);
        if neighbors.len() > self.config.m_max {
            let nbr_vec = self.get_vector(VectorId(nbr));
            let mut scored: Vec<ScoredId> = neighbors
                .iter()
                .map(|&n| ScoredId {
                    id: VectorId(n),
                    distance: self.distance.distance(nbr_vec, self.get_vector(VectorId(n))),
                })
                .collect();
            scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            let pruned = self.select_neighbors_heuristic(&scored, self.config.m_max);
            self.slot_layout.set_neighbors(data, nbr, &pruned);
        } else {
            self.slot_layout.set_neighbors(data, nbr, &neighbors);
        }
    }

    /// Freeze the builder into an immutable NswIndex.
    ///
    /// # Panics
    /// Panics if no vectors were inserted, or if insert count != capacity
    /// (dense IDs 0..capacity-1 are required).
    pub fn build(self) -> NswIndex {
        let num_inserted = self.num_inserted.load(Ordering::Relaxed) as usize;
        assert!(num_inserted > 0, "cannot build index with 0 vectors");
        assert_eq!(
            num_inserted, self.capacity,
            "expected {} vectors but only {} were inserted (dense IDs 0..N-1 required)",
            self.capacity, num_inserted
        );

        // Defense-in-depth (debug only): verify every VID 0..capacity-1 was inserted.
        #[cfg(debug_assertions)]
        for vid in 0..self.capacity {
            let word_idx = vid / 64;
            let bit_mask = 1u64 << (vid % 64);
            debug_assert!(
                self.inserted[word_idx].load(Ordering::Relaxed) & bit_mask != 0,
                "VID {} was never inserted (dense IDs 0..{} required for build-once immutable index)",
                vid, self.capacity - 1
            );
        }

        // Extract inner data from UnsafeCells (self is consumed, no more aliases)
        let vectors = self.vectors.0.into_inner();
        let graph_data = self.graph_data.0.into_inner();

        // Select hub entry set by highest degree
        let entry_set = select_entry_set(
            &graph_data,
            &self.slot_layout,
            num_inserted,
            self.config.num_entry_points,
        );

        NswIndex {
            config: self.config,
            dimension: self.dimension,
            metric: self.metric,
            distance: create_distance_computer(self.metric),
            vectors,
            num_vectors: num_inserted,
            graph_data,
            slot_layout: self.slot_layout,
            entry_set,
            visited_pool: VisitedPool::new(self.capacity),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry set selection
// ---------------------------------------------------------------------------

/// Select hub nodes by highest degree.
fn select_entry_set(
    graph_data: &[u32],
    layout: &SlotLayout,
    n: usize,
    count: usize,
) -> Vec<VectorId> {
    let count = count.min(n);
    let mut degrees: Vec<(u32, usize)> = (0..n as u32)
        .map(|vid| (vid, layout.get_neighbor_count(graph_data, vid)))
        .collect();
    degrees.sort_by(|a, b| b.1.cmp(&a.1)); // highest degree first
    degrees.truncate(count);
    degrees.into_iter().map(|(vid, _)| VectorId(vid)).collect()
}

// ---------------------------------------------------------------------------
// Reader (immutable, lock-free)
// ---------------------------------------------------------------------------

/// Immutable flat NSW index for concurrent search. No locks on the read path.
pub struct NswIndex {
    config: NswConfig,
    dimension: usize,
    metric: MetricType,
    distance: Box<dyn DistanceComputer>,
    vectors: Vec<f32>,
    num_vectors: usize,
    graph_data: Vec<u32>,
    slot_layout: SlotLayout,
    entry_set: Vec<VectorId>,
    visited_pool: VisitedPool,
}

impl NswIndex {
    /// KNN search: beam search from hub entry set.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<ScoredId> {
        debug_assert!(ef >= k);

        let mut visited = self.visited_pool.get();
        let mut nearest = FixedCapacityHeap::new(ef);
        let mut candidates = CandidateHeap::new();
        self.search_layer(query, &self.entry_set, ef, &mut visited, &mut nearest, &mut candidates);

        let mut results = nearest.into_sorted_vec();
        results.truncate(k);
        results
    }

    /// Beam search on the flat graph.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[VectorId],
        ef: usize,
        visited: &mut crate::visited::VisitedListHandle<'_>,
        nearest: &mut FixedCapacityHeap,
        candidates: &mut CandidateHeap,
    ) {
        nearest.clear(ef);
        candidates.clear();

        for &ep in entry_points {
            let d = self.distance.distance(query, self.get_vector(ep));
            let scored = ScoredId {
                distance: d,
                id: ep,
            };
            nearest.push(scored);
            candidates.push(scored);
            visited.check_and_mark(ep.0);
        }

        while let Some(candidate) = candidates.pop() {
            if let Some(furthest) = nearest.furthest() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for &nbr_raw in self.get_neighbors(candidate.id) {
                if visited.check_and_mark(nbr_raw) {
                    continue;
                }
                let nbr = VectorId(nbr_raw);
                let d = self.distance.distance(query, self.get_vector(nbr));
                let dominated =
                    nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
                if !dominated {
                    let scored = ScoredId {
                        distance: d,
                        id: nbr,
                    };
                    candidates.push(scored);
                    nearest.push(scored);
                }
            }
        }
    }

    #[inline]
    fn get_vector(&self, id: VectorId) -> &[f32] {
        let off = id.0 as usize * self.dimension;
        &self.vectors[off..off + self.dimension]
    }

    #[inline]
    fn get_neighbors(&self, id: VectorId) -> &[u32] {
        self.slot_layout.get_neighbors(&self.graph_data, id.0)
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn entry_set(&self) -> &[VectorId] {
        &self.entry_set
    }

    pub fn config(&self) -> &NswConfig {
        &self.config
    }

    pub fn metric(&self) -> MetricType {
        self.metric
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    // -- Serialization accessors (used by storage crate) --

    /// Raw flat vector data: N * dimension f32 values.
    pub fn vectors_raw(&self) -> &[f32] {
        &self.vectors
    }

    /// Neighbors of a given vector ID (public version).
    pub fn neighbors(&self, vid: u32) -> &[u32] {
        self.slot_layout.get_neighbors(&self.graph_data, vid)
    }

    /// Max neighbors per node.
    pub fn max_degree(&self) -> usize {
        self.config.m_max
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use rand::Rng;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
            .collect()
    }

    fn build_index(
        vectors: &[Vec<f32>],
        dim: usize,
        m_max: usize,
        ef_construction: usize,
        metric: MetricType,
    ) -> NswIndex {
        let config = NswConfig::new(m_max, ef_construction);
        let builder = NswBuilder::new(config, dim, metric, vectors.len());
        for (i, v) in vectors.iter().enumerate() {
            builder.insert(VectorId(i as u32), v);
        }
        builder.build()
    }

    // ------------------------------------------------------------------
    // Graph invariant tests
    // ------------------------------------------------------------------

    #[test]
    fn no_duplicate_neighbors() {
        let n = 500;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 55);
        let index = build_index(&vectors, dim, 32, 200, MetricType::L2);

        for vid in 0..n as u32 {
            let nbrs = index.get_neighbors(VectorId(vid));
            let mut seen = std::collections::HashSet::new();
            for &nbr in nbrs {
                assert!(
                    seen.insert(nbr),
                    "VID {} has duplicate neighbor {}",
                    vid,
                    nbr
                );
                assert_ne!(nbr, vid, "VID {} has self-loop", vid);
            }
        }
    }

    #[test]
    fn neighbor_count_within_bounds() {
        let n = 500;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 77);
        let config = NswConfig::new(32, 200);
        let builder = NswBuilder::new(config.clone(), dim, MetricType::L2, n);

        for (i, v) in vectors.iter().enumerate() {
            builder.insert(VectorId(i as u32), v);
        }

        let index = builder.build();

        for vid in 0..n as u32 {
            let count = index.get_neighbors(VectorId(vid)).len();
            assert!(
                count <= index.config.m_max,
                "vid {} has {} neighbors, max is {}",
                vid,
                count,
                index.config.m_max
            );
        }
    }

    // ------------------------------------------------------------------
    // Bounds / contract tests
    // ------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "VID 100 out of bounds")]
    fn insert_oob_panics() {
        let config = NswConfig::new(32, 200);
        let builder = NswBuilder::new(config, 4, MetricType::L2, 100);
        builder.insert(VectorId(100), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn insert_wrong_dim_panics() {
        let config = NswConfig::new(32, 200);
        let builder = NswBuilder::new(config, 4, MetricType::L2, 10);
        builder.insert(VectorId(0), &[1.0, 2.0, 3.0]); // dim 3, expected 4
    }

    #[test]
    #[should_panic(expected = "expected 10 vectors but only 5 were inserted")]
    fn build_incomplete_panics() {
        let config = NswConfig::new(32, 200);
        let builder = NswBuilder::new(config, 4, MetricType::L2, 10);
        for i in 0..5 {
            builder.insert(VectorId(i), &[1.0, 2.0, 3.0, 4.0]);
        }
        let _index = builder.build();
    }

    // ------------------------------------------------------------------
    // Entry set tests
    // ------------------------------------------------------------------

    #[test]
    fn entry_set_size() {
        let n = 200;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 42);
        let mut config = NswConfig::new(32, 200);
        config.num_entry_points = 64;
        let builder = NswBuilder::new(config, dim, MetricType::L2, n);
        for (i, v) in vectors.iter().enumerate() {
            builder.insert(VectorId(i as u32), v);
        }
        let index = builder.build();
        assert_eq!(index.entry_set().len(), 64);

        // When n < num_entry_points, entry set size = n
        let n_small = 10;
        let vectors_small = generate_vectors(n_small, dim, 43);
        let mut config2 = NswConfig::new(32, 200);
        config2.num_entry_points = 64;
        let builder2 = NswBuilder::new(config2, dim, MetricType::L2, n_small);
        for (i, v) in vectors_small.iter().enumerate() {
            builder2.insert(VectorId(i as u32), v);
        }
        let index2 = builder2.build();
        assert_eq!(index2.entry_set().len(), n_small);
    }

    #[test]
    fn entry_set_are_hubs() {
        let n = 500;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 99);
        let mut config = NswConfig::new(32, 200);
        config.num_entry_points = 32;
        let builder = NswBuilder::new(config, dim, MetricType::L2, n);
        for (i, v) in vectors.iter().enumerate() {
            builder.insert(VectorId(i as u32), v);
        }
        let index = builder.build();

        // Compute degrees for all nodes
        let mut degrees: Vec<(u32, usize)> = (0..n as u32)
            .map(|vid| (vid, index.get_neighbors(VectorId(vid)).len()))
            .collect();
        degrees.sort_by(|a, b| b.1.cmp(&a.1));
        let top_k: std::collections::HashSet<u32> = degrees[..32].iter().map(|(vid, _)| *vid).collect();

        // Every entry set node should be in the top-K by degree
        for ep in index.entry_set() {
            assert!(
                top_k.contains(&ep.0),
                "entry point VID {} is not in top-{} by degree",
                ep.0,
                32
            );
        }
    }

    // ------------------------------------------------------------------
    // Concurrent construction
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_insert_4_threads() {
        let n = 4000;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 900);
        let config = NswConfig::new(32, 200);
        let builder = Arc::new(NswBuilder::new(config, dim, MetricType::L2, n));

        let num_threads = 4;
        let chunk_size = n / num_threads;

        std::thread::scope(|s| {
            for t in 0..num_threads {
                let builder = Arc::clone(&builder);
                let vectors = &vectors;
                s.spawn(move || {
                    let start = t * chunk_size;
                    let end = if t == num_threads - 1 { n } else { start + chunk_size };
                    for i in start..end {
                        builder.insert(VectorId(i as u32), &vectors[i]);
                    }
                });
            }
        });

        let builder = Arc::try_unwrap(builder)
            .ok()
            .expect("all threads joined, Arc should be unique");
        assert_eq!(builder.num_inserted.load(Ordering::Relaxed) as usize, n);

        let index = builder.build();
        // Every node should have at least one neighbor
        for vid in 1..n as u32 {
            let count = index.get_neighbors(VectorId(vid)).len();
            assert!(
                count > 0,
                "vid {} has 0 neighbors after concurrent build",
                vid
            );
        }
    }

    // ------------------------------------------------------------------
    // Insert-once contract tests
    // ------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "inserted twice")]
    fn duplicate_insert_panics() {
        let n = 100;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 42);
        let config = NswConfig::new(16, 50);
        let builder = NswBuilder::new(config, dim, MetricType::L2, n);
        for i in 0..n {
            builder.insert(VectorId(i as u32), &vectors[i]);
        }
        builder.insert(VectorId(0), &vectors[0]); // must panic
    }

    #[test]
    #[should_panic(expected = "expected 100 vectors but only 99")]
    fn build_catches_incomplete_insert() {
        let n = 100;
        let dim = 32;
        let vectors = generate_vectors(n, dim, 42);
        let config = NswConfig::new(16, 50);
        let builder = NswBuilder::new(config, dim, MetricType::L2, n);
        for i in 0..n {
            if i == 50 { continue; }
            builder.insert(VectorId(i as u32), &vectors[i]);
        }
        builder.build(); // panics: 99 != 100
    }

}
