//! Fixed-size adjacency block layout (v1 + v2).
//!
//! Each vector gets one 4096-byte block in adjacency.dat.
//! Block offset = vid * BLOCK_SIZE.
//!
//! ## v1 layout (legacy, no inline codes):
//!   [degree: u16][padding: 6 bytes][neighbor_vids: u32 × degree][zero-pad to 4096]
//!
//! ## v2 layout (inline PQ codes):
//!   [degree: u16][version: u8 = 0x01][code_type: u8 = 0x01][num_subquantizers: u16]
//!   [reserved: 2 bytes][neighbor_vids: u32 × degree][neighbor_codes: u8[M] × degree]
//!   [zero-pad to 4096]
//!
//! v1 blocks have version byte (offset 2) = 0x00 (was padding).
//! v2 reader transparently handles v1 blocks (no codes available).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub const BLOCK_SIZE: usize = 4096;
const HEADER_SIZE: usize = 8;
const MAX_NEIGHBORS_V1: usize = (BLOCK_SIZE - HEADER_SIZE) / 4;

/// Block layout version.
const LAYOUT_V2: u8 = 0x01;
/// Code type: Product Quantization.
const CODE_TYPE_PQ: u8 = 0x01;

// ─── v1 encode/decode (legacy, kept for backward compat) ───────────────────

/// Encode a neighbor list into a v1 4096-byte block (no inline codes).
pub fn encode_adj_block(neighbors: &[u32], buf: &mut [u8; BLOCK_SIZE]) {
    assert!(
        neighbors.len() <= MAX_NEIGHBORS_V1,
        "too many neighbors: {} > {}",
        neighbors.len(),
        MAX_NEIGHBORS_V1
    );

    buf.fill(0);
    let count = neighbors.len() as u16;
    buf[0..2].copy_from_slice(&count.to_le_bytes());
    // version byte at offset 2 stays 0x00 (v1)

    for (i, &nbr) in neighbors.iter().enumerate() {
        let off = HEADER_SIZE + i * 4;
        buf[off..off + 4].copy_from_slice(&nbr.to_le_bytes());
    }
}

/// Decode a neighbor list from a v1 block (allocating). Legacy interface.
pub fn decode_adj_block(buf: &[u8]) -> Vec<u32> {
    assert!(buf.len() >= HEADER_SIZE);
    let count = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    assert!(count <= MAX_NEIGHBORS_V1);

    let mut neighbors = Vec::with_capacity(count);
    for i in 0..count {
        let off = HEADER_SIZE + i * 4;
        let nbr = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        neighbors.push(nbr);
    }
    neighbors
}

// ─── v2 zero-copy decode ───────────────────────────────────────────────────

/// Zero-copy view into a decoded adjacency block.
/// Borrows directly from the 4KB buffer — no heap allocation on the hot path.
#[derive(Debug)]
pub struct AdjBlockView<'a> {
    /// Degree (number of neighbors).
    pub degree: usize,
    /// Raw bytes for neighbor VIDs: `degree * 4` bytes starting at offset 8.
    /// Use `neighbor_vid(i)` for safe access.
    vid_bytes: &'a [u8],
    /// PQ codes for neighbors, or empty if v1 block.
    /// Layout: `degree * M` bytes, row-major (neighbor i's code at `[i*M .. (i+1)*M]`).
    code_bytes: &'a [u8],
    /// Number of PQ subquantizers (M). 0 if v1 block (no codes).
    pub num_subquantizers: usize,
}

impl<'a> AdjBlockView<'a> {
    /// Read the i-th neighbor VID (little-endian u32).
    #[inline]
    pub fn neighbor_vid(&self, i: usize) -> u32 {
        debug_assert!(i < self.degree);
        let off = i * 4;
        u32::from_le_bytes([
            self.vid_bytes[off],
            self.vid_bytes[off + 1],
            self.vid_bytes[off + 2],
            self.vid_bytes[off + 3],
        ])
    }

    /// Get the PQ code slice for the i-th neighbor.
    /// Returns empty slice if this is a v1 block (no codes).
    #[inline]
    pub fn neighbor_code(&self, i: usize) -> &'a [u8] {
        if self.num_subquantizers == 0 {
            return &[];
        }
        debug_assert!(i < self.degree);
        let m = self.num_subquantizers;
        &self.code_bytes[i * m..(i + 1) * m]
    }

    /// Whether this block has inline PQ codes.
    #[inline]
    pub fn has_codes(&self) -> bool {
        self.num_subquantizers > 0
    }
}

/// Zero-copy decode of an adjacency block (v1 or v2).
/// Returns a view that borrows from `buf`. No heap allocation.
///
/// Detects v1 vs v2 via the version byte at offset 2.
pub fn decode_adj_block_view(buf: &[u8]) -> AdjBlockView<'_> {
    assert!(buf.len() >= HEADER_SIZE);

    let degree = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    let version = buf[2];

    let vids_start = HEADER_SIZE;
    let vids_end = vids_start + degree * 4;
    assert!(vids_end <= buf.len(), "block too small for {degree} neighbor VIDs");

    if version == LAYOUT_V2 {
        // v2: has inline PQ codes
        let code_type = buf[3];
        assert_eq!(code_type, CODE_TYPE_PQ, "unsupported code_type {code_type}");

        let m = u16::from_le_bytes([buf[4], buf[5]]) as usize;
        assert!(m > 0, "v2 block with M=0");

        let codes_start = vids_end;
        let codes_end = codes_start + degree * m;
        assert!(
            codes_end <= buf.len(),
            "block too small for {degree} neighbors × {m} PQ codes: need {codes_end}, have {}",
            buf.len()
        );

        AdjBlockView {
            degree,
            vid_bytes: &buf[vids_start..vids_end],
            code_bytes: &buf[codes_start..codes_end],
            num_subquantizers: m,
        }
    } else {
        // v1 (or v0): no codes
        AdjBlockView {
            degree,
            vid_bytes: &buf[vids_start..vids_end],
            code_bytes: &[],
            num_subquantizers: 0,
        }
    }
}

// ─── v2 encode ─────────────────────────────────────────────────────────────

/// Compute the maximum degree that fits in a v2 block with M subquantizers.
pub fn max_degree_v2(m: usize) -> usize {
    // header(8) + degree * (4 bytes VID + M bytes code) <= BLOCK_SIZE
    (BLOCK_SIZE - HEADER_SIZE) / (4 + m)
}

/// Encode a v2 adjacency block with inline PQ codes.
///
/// `neighbors`: neighbor VIDs
/// `codes`: flat PQ codes, `neighbors.len() * m` bytes, row-major.
///   codes[i*m .. (i+1)*m] is the PQ code for neighbors[i].
/// `m`: number of PQ subquantizers.
pub fn encode_adj_block_v2(
    neighbors: &[u32],
    codes: &[u8],
    m: usize,
    buf: &mut [u8; BLOCK_SIZE],
) {
    let degree = neighbors.len();
    let max_deg = max_degree_v2(m);
    assert!(
        degree <= max_deg,
        "too many neighbors for v2 block: {degree} > {max_deg} (M={m})"
    );
    assert_eq!(
        codes.len(),
        degree * m,
        "codes length mismatch: expected {} ({}×{}), got {}",
        degree * m,
        degree,
        m,
        codes.len()
    );

    buf.fill(0);

    // Header
    buf[0..2].copy_from_slice(&(degree as u16).to_le_bytes());
    buf[2] = LAYOUT_V2;
    buf[3] = CODE_TYPE_PQ;
    buf[4..6].copy_from_slice(&(m as u16).to_le_bytes());
    // buf[6..8] reserved (zeros)

    // Neighbor VIDs
    let vids_start = HEADER_SIZE;
    for (i, &nbr) in neighbors.iter().enumerate() {
        let off = vids_start + i * 4;
        buf[off..off + 4].copy_from_slice(&nbr.to_le_bytes());
    }

    // PQ codes (immediately after VIDs)
    let codes_start = vids_start + degree * 4;
    buf[codes_start..codes_start + codes.len()].copy_from_slice(codes);
}

// ─── v2 file writer ────────────────────────────────────────────────────────

/// Write adjacency.dat with v2 blocks (inline PQ codes).
///
/// `pq_codes_all`: flat array of all PQ codes, shape N × M, row-major.
///   `pq_codes_all[vid * m .. (vid+1) * m]` is the PQ code for vector `vid`.
/// `neighbors_fn`: returns neighbor VIDs for a given vid.
/// `m`: number of PQ subquantizers.
pub fn write_adjacency_file_v2<'a>(
    path: &Path,
    num_vectors: u32,
    neighbors_fn: impl Fn(u32) -> &'a [u32],
    pq_codes_all: &[u8],
    m: usize,
) -> io::Result<()> {
    assert_eq!(
        pq_codes_all.len(),
        num_vectors as usize * m,
        "pq_codes_all length mismatch"
    );

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut block = [0u8; BLOCK_SIZE];
    let mut code_buf = Vec::new();

    for vid in 0..num_vectors {
        let nbrs = neighbors_fn(vid);

        // Gather PQ codes for this node's neighbors
        code_buf.clear();
        code_buf.reserve(nbrs.len() * m);
        for &nbr in nbrs {
            let start = nbr as usize * m;
            code_buf.extend_from_slice(&pq_codes_all[start..start + m]);
        }

        encode_adj_block_v2(nbrs, &code_buf, m, &mut block);
        writer.write_all(&block)?;
    }

    writer.flush()
}

/// Write adjacency.dat with v1 blocks (no codes). Legacy interface.
pub fn write_adjacency_file<'a>(
    path: &Path,
    num_vectors: u32,
    neighbors_fn: impl Fn(u32) -> &'a [u32],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut block = [0u8; BLOCK_SIZE];

    for vid in 0..num_vectors {
        let nbrs = neighbors_fn(vid);
        encode_adj_block(nbrs, &mut block);
        writer.write_all(&block)?;
    }

    writer.flush()
}

// ─── v3: page-based packed adjacency ──────────────────────────────────────
//
// Multiple adjacency records packed into 4KB pages. Records are variable-length:
//   [degree: u16][neighbor_vids: u32 × degree]
// = 2 + degree*4 bytes per record.
//
// A separate DRAM-resident adj_index maps vid → (page_id, offset, degree).
// Pages have no internal directory — the adj_index provides all addressing.
//
// BFS reorder determines physical write order: nodes close in the graph
// get packed into the same/nearby pages, so beam search expansions share
// pages within a query.

/// Entry in the adjacency index (adj_index.dat). 8 bytes per VID.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct AdjIndexEntry {
    /// Page number in adjacency_pages.dat.
    pub page_id: u32,
    /// Byte offset of this VID's record within the page.
    pub offset: u16,
    /// Degree (number of neighbors). Redundant with in-page data but avoids
    /// needing to read the page just to know the degree.
    pub degree: u16,
}

impl AdjIndexEntry {
    pub const SIZE: usize = 8;

    pub fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&self.page_id.to_le_bytes());
        buf[4..6].copy_from_slice(&self.offset.to_le_bytes());
        buf[6..8].copy_from_slice(&self.degree.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 8]) -> Self {
        Self {
            page_id: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            offset: u16::from_le_bytes([buf[4], buf[5]]),
            degree: u16::from_le_bytes([buf[6], buf[7]]),
        }
    }
}

/// Load adj_index.dat into memory. Returns one AdjIndexEntry per VID.
pub fn load_adj_index(path: &Path, num_vectors: usize) -> io::Result<Vec<AdjIndexEntry>> {
    let data = std::fs::read(path)?;
    if data.len() != num_vectors * AdjIndexEntry::SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "adj_index size mismatch: expected {} bytes ({} × {}), got {}",
                num_vectors * AdjIndexEntry::SIZE,
                num_vectors,
                AdjIndexEntry::SIZE,
                data.len()
            ),
        ));
    }
    let entries: Vec<AdjIndexEntry> = data
        .chunks_exact(AdjIndexEntry::SIZE)
        .map(|chunk| {
            let buf: [u8; 8] = chunk.try_into().unwrap();
            AdjIndexEntry::from_bytes(&buf)
        })
        .collect();
    Ok(entries)
}

/// Read the i-th neighbor VID from a packed page record.
#[inline]
pub fn page_record_vid(page_buf: &[u8], entry: &AdjIndexEntry, i: usize) -> u32 {
    debug_assert!(i < entry.degree as usize);
    let off = entry.offset as usize + 2 + i * 4; // skip degree(2) + i*4
    u32::from_le_bytes([
        page_buf[off],
        page_buf[off + 1],
        page_buf[off + 2],
        page_buf[off + 3],
    ])
}

/// BFS reorder: traverse graph from entry points, return old_to_new VID mapping.
/// Nodes visited early get low new VIDs → packed into early pages → better locality.
pub fn bfs_reorder_graph<'a>(
    n: usize,
    entry_set: &[u32],
    neighbors_fn: impl Fn(u32) -> &'a [u32],
) -> Vec<u32> {
    let mut old_to_new = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    let mut next_id = 0u32;

    for &ep in entry_set {
        let v = ep as usize;
        if v < n && old_to_new[v] == u32::MAX {
            old_to_new[v] = next_id;
            next_id += 1;
            queue.push_back(v);
        }
    }

    while let Some(v) = queue.pop_front() {
        for &nbr in neighbors_fn(v as u32) {
            let ni = nbr as usize;
            if ni < n && old_to_new[ni] == u32::MAX {
                old_to_new[ni] = next_id;
                next_id += 1;
                queue.push_back(ni);
            }
        }
    }

    // Unreachable nodes
    for i in 0..n {
        if old_to_new[i] == u32::MAX {
            old_to_new[i] = next_id;
            next_id += 1;
        }
    }

    old_to_new
}

/// Write v3 page-packed adjacency files.
///
/// Produces:
///   - `adjacency_pages.dat`: packed 4KB pages with adjacency records
///   - `adj_index.dat`: vid → (page_id, offset, degree) mapping
///
/// `reorder`: old_to_new VID mapping (from `bfs_reorder_graph`).
///   Determines physical packing order. Logical VIDs are unchanged.
/// `neighbors_fn`: returns neighbor VIDs for a given logical vid.
///
/// Returns the number of pages written.
pub fn write_packed_adjacency<'a>(
    pages_path: &Path,
    index_path: &Path,
    num_vectors: u32,
    neighbors_fn: impl Fn(u32) -> &'a [u32],
    reorder: &[u32],
) -> io::Result<u32> {
    let n = num_vectors as usize;
    assert_eq!(reorder.len(), n);
    if n == 0 {
        // Empty index: write empty files for consistency.
        std::fs::write(pages_path, &[] as &[u8])?;
        std::fs::write(index_path, &[] as &[u8])?;
        return Ok(0);
    }

    // Validate that reorder is a permutation of 0..n-1.
    // Use a compact bitset (n/8 bytes) rather than Vec<bool>.
    let mut seen = vec![0u8; (n + 7) / 8];
    for &new in reorder {
        let new = new as usize;
        assert!(new < n, "reorder out of range: new_vid={new} n={n}");
        let byte = new / 8;
        let mask = 1u8 << (new % 8);
        assert!(
            (seen[byte] & mask) == 0,
            "reorder is not a permutation: duplicate new_vid={new}"
        );
        seen[byte] |= mask;
    }

    // Build new_to_old so we iterate in reorder (packing) order
    let mut new_to_old = vec![0u32; n];
    for old in 0..n {
        new_to_old[reorder[old] as usize] = old as u32;
    }

    // First pass: compute record sizes and assign to pages
    let mut adj_index = vec![AdjIndexEntry::default(); n];
    let mut page_id = 0u32;
    let mut page_used = 0usize;

    for new_vid in 0..n {
        let old_vid = new_to_old[new_vid] as usize;
        let nbrs = neighbors_fn(old_vid as u32);
        let degree = nbrs.len();
        let record_bytes = 2 + degree * 4; // degree(u16) + VIDs
        assert!(
            record_bytes <= BLOCK_SIZE,
            "adj record too large for one page: vid={} degree={} bytes={} (page={})",
            old_vid,
            degree,
            record_bytes,
            BLOCK_SIZE
        );

        // Start new page if record doesn't fit
        if page_used + record_bytes > BLOCK_SIZE && page_used > 0 {
            page_id += 1;
            page_used = 0;
        }

        assert!(
            page_used <= u16::MAX as usize,
            "page offset overflow: page_used={}",
            page_used
        );
        adj_index[old_vid] = AdjIndexEntry {
            page_id,
            offset: page_used as u16,
            degree: degree as u16,
        };
        page_used += record_bytes;
    }
    let total_pages = page_id + 1;

    // Second pass: write pages
    let pages_file = File::create(pages_path)?;
    let mut pages_writer = BufWriter::new(pages_file);
    let mut page_buf = [0u8; BLOCK_SIZE];
    let mut current_page = 0u32;
    page_buf.fill(0);

    for new_vid in 0..n {
        let old_vid = new_to_old[new_vid] as usize;
        let entry = adj_index[old_vid];

        // Flush page if we've moved to a new one
        while current_page < entry.page_id {
            pages_writer.write_all(&page_buf)?;
            page_buf.fill(0);
            current_page += 1;
        }

        // Write record into page buffer
        let off = entry.offset as usize;
        let degree = entry.degree;
        let nbrs = neighbors_fn(old_vid as u32);
        debug_assert_eq!(nbrs.len(), degree as usize);

        page_buf[off..off + 2].copy_from_slice(&degree.to_le_bytes());
        for (i, &nbr) in nbrs.iter().enumerate() {
            let vid_off = off + 2 + i * 4;
            page_buf[vid_off..vid_off + 4].copy_from_slice(&nbr.to_le_bytes());
        }
    }
    // Flush last page
    pages_writer.write_all(&page_buf)?;
    pages_writer.flush()?;

    // Write adj_index
    let index_file = File::create(index_path)?;
    let mut index_writer = BufWriter::new(index_file);
    for entry in &adj_index {
        index_writer.write_all(&entry.to_bytes())?;
    }
    index_writer.flush()?;

    Ok(total_pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── v1 tests (unchanged) ──────────────────────────────────────────

    #[test]
    fn roundtrip_empty() {
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&[], &mut buf);
        let decoded = decode_adj_block(&buf);
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_small() {
        let neighbors = vec![10, 20, 30, 42];
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&neighbors, &mut buf);
        let decoded = decode_adj_block(&buf);
        assert_eq!(decoded, neighbors);
    }

    #[test]
    fn roundtrip_max() {
        let neighbors: Vec<u32> = (0..MAX_NEIGHBORS_V1 as u32).collect();
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&neighbors, &mut buf);
        let decoded = decode_adj_block(&buf);
        assert_eq!(decoded, neighbors);
    }

    #[test]
    fn file_offsets_correct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adjacency.dat");

        let data: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![]];
        write_adjacency_file(&path, 3, |vid| &data[vid as usize]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * BLOCK_SIZE);

        for (vid, expected) in data.iter().enumerate() {
            let offset = vid * BLOCK_SIZE;
            let decoded = decode_adj_block(&bytes[offset..offset + BLOCK_SIZE]);
            assert_eq!(&decoded, expected, "mismatch at vid {}", vid);
        }
    }

    // ─── v2 tests ──────────────────────────────────────────────────────

    #[test]
    fn v2_roundtrip() {
        let neighbors = vec![10u32, 20, 30];
        let m = 4; // 4 subquantizers for test
        let codes: Vec<u8> = vec![
            1, 2, 3, 4,   // codes for neighbor 10
            5, 6, 7, 8,   // codes for neighbor 20
            9, 10, 11, 12, // codes for neighbor 30
        ];

        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block_v2(&neighbors, &codes, m, &mut buf);

        let view = decode_adj_block_view(&buf);
        assert_eq!(view.degree, 3);
        assert!(view.has_codes());
        assert_eq!(view.num_subquantizers, 4);

        assert_eq!(view.neighbor_vid(0), 10);
        assert_eq!(view.neighbor_vid(1), 20);
        assert_eq!(view.neighbor_vid(2), 30);

        assert_eq!(view.neighbor_code(0), &[1, 2, 3, 4]);
        assert_eq!(view.neighbor_code(1), &[5, 6, 7, 8]);
        assert_eq!(view.neighbor_code(2), &[9, 10, 11, 12]);
    }

    #[test]
    fn v2_empty() {
        let m = 16;
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block_v2(&[], &[], m, &mut buf);

        let view = decode_adj_block_view(&buf);
        assert_eq!(view.degree, 0);
        assert!(view.has_codes());
        assert_eq!(view.num_subquantizers, 16);
    }

    #[test]
    fn v1_compat_via_view() {
        // v1-encoded block should decode via view with no codes
        let neighbors = vec![10u32, 20, 30];
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&neighbors, &mut buf);

        let view = decode_adj_block_view(&buf);
        assert_eq!(view.degree, 3);
        assert!(!view.has_codes());
        assert_eq!(view.num_subquantizers, 0);

        assert_eq!(view.neighbor_vid(0), 10);
        assert_eq!(view.neighbor_vid(1), 20);
        assert_eq!(view.neighbor_vid(2), 30);
        assert!(view.neighbor_code(0).is_empty());
    }

    #[test]
    fn max_degree_v2_capacity() {
        // PQ48: (4096 - 8) / (4 + 48) = 78
        assert_eq!(max_degree_v2(48), 78);
        // PQ32: (4096 - 8) / (4 + 32) = 113
        assert_eq!(max_degree_v2(32), 113);
        // PQ16: (4096 - 8) / (4 + 16) = 204
        assert_eq!(max_degree_v2(16), 204);
    }

    #[test]
    fn v2_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adjacency.dat");

        let m = 4;
        let n = 3u32;
        // neighbors: node 0 → [1,2], node 1 → [0,2], node 2 → [0]
        let nbr_data: Vec<Vec<u32>> = vec![vec![1, 2], vec![0, 2], vec![0]];
        // PQ codes for each vector (n=3, m=4)
        let pq_codes_all: Vec<u8> = vec![
            10, 11, 12, 13, // vector 0
            20, 21, 22, 23, // vector 1
            30, 31, 32, 33, // vector 2
        ];

        write_adjacency_file_v2(
            &path,
            n,
            |vid| &nbr_data[vid as usize],
            &pq_codes_all,
            m,
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * BLOCK_SIZE);

        // Node 0: neighbors [1, 2], codes should be [20,21,22,23] and [30,31,32,33]
        let view0 = decode_adj_block_view(&bytes[0..BLOCK_SIZE]);
        assert_eq!(view0.degree, 2);
        assert_eq!(view0.neighbor_vid(0), 1);
        assert_eq!(view0.neighbor_vid(1), 2);
        assert_eq!(view0.neighbor_code(0), &[20, 21, 22, 23]);
        assert_eq!(view0.neighbor_code(1), &[30, 31, 32, 33]);

        // Node 2: neighbor [0], code should be [10,11,12,13]
        let view2 = decode_adj_block_view(&bytes[2 * BLOCK_SIZE..3 * BLOCK_SIZE]);
        assert_eq!(view2.degree, 1);
        assert_eq!(view2.neighbor_vid(0), 0);
        assert_eq!(view2.neighbor_code(0), &[10, 11, 12, 13]);
    }

    // ─── v3 (page-packed) tests ──────────────────────────────────────

    #[test]
    fn v3_packed_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let pages_path = dir.path().join("adjacency_pages.dat");
        let index_path = dir.path().join("adj_index.dat");

        let n = 5u32;
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],
            vec![0, 2, 3],
            vec![0, 1],
            vec![1, 4],
            vec![3],
        ];
        // Identity reorder (no reorder)
        let reorder: Vec<u32> = (0..n).collect();

        let total_pages = write_packed_adjacency(
            &pages_path,
            &index_path,
            n,
            |vid| &adj[vid as usize],
            &reorder,
        )
        .unwrap();

        // Load index
        let index = load_adj_index(&index_path, n as usize).unwrap();
        assert_eq!(index.len(), 5);

        // All should fit on 1 page: total = 5 records,
        // sizes = (2+8)+(2+12)+(2+8)+(2+8)+(2+4) = 48 bytes << 4096
        assert_eq!(total_pages, 1);
        for entry in &index {
            assert_eq!(entry.page_id, 0);
        }

        // Read page and verify neighbor VIDs
        let page_data = std::fs::read(&pages_path).unwrap();
        assert_eq!(page_data.len(), BLOCK_SIZE); // 1 page

        for vid in 0..n as usize {
            let entry = &index[vid];
            assert_eq!(entry.degree as usize, adj[vid].len());
            for (i, &expected_nbr) in adj[vid].iter().enumerate() {
                let actual = page_record_vid(&page_data, entry, i);
                assert_eq!(actual, expected_nbr, "vid={} neighbor {}", vid, i);
            }
        }
    }

    #[test]
    fn v3_packed_with_bfs_reorder() {
        let dir = tempfile::tempdir().unwrap();
        let pages_path = dir.path().join("adjacency_pages.dat");
        let index_path = dir.path().join("adj_index.dat");

        let n = 5u32;
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],
            vec![0, 2, 3],
            vec![0, 1],
            vec![1, 4],
            vec![3],
        ];
        // BFS reorder from entry point [1]
        let reorder = bfs_reorder_graph(n as usize, &[1], |vid| &adj[vid as usize]);

        let total_pages = write_packed_adjacency(
            &pages_path,
            &index_path,
            n,
            |vid| &adj[vid as usize],
            &reorder,
        )
        .unwrap();

        // Load and verify
        let index = load_adj_index(&index_path, n as usize).unwrap();
        let page_data = std::fs::read(&pages_path).unwrap();

        // Verify every adjacency record is correct (reorder doesn't change neighbor VIDs)
        for vid in 0..n as usize {
            let entry = &index[vid];
            assert_eq!(entry.degree as usize, adj[vid].len());
            for (i, &expected_nbr) in adj[vid].iter().enumerate() {
                let actual = page_record_vid(&page_data, entry, i);
                assert_eq!(actual, expected_nbr, "vid={} neighbor {}", vid, i);
            }
        }

        // VID 1 (entry point) should have new_vid=0 → earliest in packing
        assert_eq!(reorder[1], 0);
        assert_eq!(total_pages, 1); // small graph, all fits in 1 page
    }

    #[test]
    fn v3_packed_page_boundary() {
        // Force a page split by creating nodes with large enough records
        let dir = tempfile::tempdir().unwrap();
        let pages_path = dir.path().join("adjacency_pages.dat");
        let index_path = dir.path().join("adj_index.dat");

        // Each node has 100 neighbors → record = 2 + 400 = 402 bytes
        // 4096 / 402 ≈ 10 nodes per page
        let n = 25u32;
        let adj: Vec<Vec<u32>> = (0..n as usize)
            .map(|vid| {
                (0..100)
                    .map(|i| ((vid + i + 1) % n as usize) as u32)
                    .collect()
            })
            .collect();
        let reorder: Vec<u32> = (0..n).collect();

        let total_pages = write_packed_adjacency(
            &pages_path,
            &index_path,
            n,
            |vid| &adj[vid as usize],
            &reorder,
        )
        .unwrap();

        assert!(total_pages > 1, "should span multiple pages, got {}", total_pages);

        let index = load_adj_index(&index_path, n as usize).unwrap();
        let page_data = std::fs::read(&pages_path).unwrap();
        assert_eq!(page_data.len(), total_pages as usize * BLOCK_SIZE);

        // Verify all records
        for vid in 0..n as usize {
            let entry = &index[vid];
            assert_eq!(entry.degree, 100);
            let page_start = entry.page_id as usize * BLOCK_SIZE;
            let page_buf = &page_data[page_start..page_start + BLOCK_SIZE];
            for i in 0..100 {
                let actual = page_record_vid(page_buf, entry, i);
                assert_eq!(actual, adj[vid][i], "vid={} neighbor {}", vid, i);
            }
        }
    }
}
