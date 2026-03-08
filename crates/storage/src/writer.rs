//! IndexWriter — serializes an in-memory graph to disk files.
//!
//! Produces three or four files in the output directory:
//!   - adjacency.dat: one 4096-byte block per vector (v1 or v2)
//!   - vectors.dat: contiguous f32 array
//!   - meta.json: index metadata + entry set
//!   - pq_codebook.bin: PQ codebook (only for v2 / inline PQ)

use std::io;
use std::path::PathBuf;

use divergence_core::quantization::pq::PqCodebook;

use crate::adjacency::{self, BLOCK_SIZE};
use crate::meta::{IndexMeta, PqMeta};
use crate::vectors;

pub struct IndexWriter {
    dir: PathBuf,
}

impl IndexWriter {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Write v1 index files (no inline PQ codes). Legacy interface.
    pub fn write<'a>(
        &self,
        num_vectors: u32,
        dimension: usize,
        metric: &str,
        max_degree: usize,
        ef_construction: usize,
        entry_set: &[u32],
        vectors_data: &[f32],
        neighbors_fn: impl Fn(u32) -> &'a [u32],
    ) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;

        adjacency::write_adjacency_file(
            &self.adj_path(),
            num_vectors,
            &neighbors_fn,
        )?;

        vectors::write_vectors_file(&self.vec_path(), vectors_data)?;

        let meta = IndexMeta {
            dimension,
            metric: metric.to_string(),
            num_vectors,
            max_degree,
            ef_construction,
            adj_block_size: BLOCK_SIZE,
            entry_set: entry_set.iter().map(|&v| v).collect(),
            adj_layout_version: 1,
            pq: None,
            num_pages: None,
        };
        meta.write_to(&self.meta_path())?;

        Ok(())
    }

    /// Write v2 index files with inline PQ codes in adjacency blocks.
    ///
    /// `codebook`: trained PQ codebook.
    /// `pq_codes_all`: flat encoded PQ codes, shape (num_vectors, codebook.m).
    ///   Must be pre-computed via `codebook.encode_all()`.
    /// `pq_metric`: metric used for PQ training ("l2" or "ip").
    pub fn write_v2<'a>(
        &self,
        num_vectors: u32,
        dimension: usize,
        metric: &str,
        max_degree: usize,
        ef_construction: usize,
        entry_set: &[u32],
        vectors_data: &[f32],
        neighbors_fn: impl Fn(u32) -> &'a [u32],
        codebook: &PqCodebook,
        pq_codes_all: &[u8],
        pq_metric: &str,
    ) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;

        // Write v2 adjacency blocks with inline PQ codes
        adjacency::write_adjacency_file_v2(
            &self.adj_path(),
            num_vectors,
            &neighbors_fn,
            pq_codes_all,
            codebook.m,
        )?;

        vectors::write_vectors_file(&self.vec_path(), vectors_data)?;

        // Write PQ codebook
        let codebook_filename = "pq_codebook.bin";
        codebook.save(&self.dir.join(codebook_filename))?;

        let meta = IndexMeta {
            dimension,
            metric: metric.to_string(),
            num_vectors,
            max_degree,
            ef_construction,
            adj_block_size: BLOCK_SIZE,
            entry_set: entry_set.iter().map(|&v| v).collect(),
            adj_layout_version: 2,
            pq: Some(PqMeta {
                num_subquantizers: codebook.m,
                num_centroids: codebook.k,
                subspace_dim: codebook.subspace_dim,
                metric: pq_metric.to_string(),
                codebook_file: codebook_filename.to_string(),
            }),
            num_pages: None,
        };
        meta.write_to(&self.meta_path())?;

        Ok(())
    }

    /// Write v3 index files with page-packed adjacency + BFS reorder.
    ///
    /// Produces:
    ///   - adjacency_pages.dat: packed 4KB pages
    ///   - adj_index.dat: vid → (page_id, offset, degree), DRAM-resident
    ///   - vectors.dat: contiguous f32 array
    ///   - meta.json: index metadata (adj_layout_version=3)
    ///
    /// `reorder`: old_to_new VID mapping (from `bfs_reorder_graph`).
    pub fn write_v3<'a>(
        &self,
        num_vectors: u32,
        dimension: usize,
        metric: &str,
        max_degree: usize,
        ef_construction: usize,
        entry_set: &[u32],
        vectors_data: &[f32],
        neighbors_fn: impl Fn(u32) -> &'a [u32],
        reorder: &[u32],
    ) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;

        let num_pages = adjacency::write_packed_adjacency(
            &self.pages_path(),
            &self.adj_index_path(),
            num_vectors,
            &neighbors_fn,
            reorder,
        )?;

        vectors::write_vectors_file(&self.vec_path(), vectors_data)?;

        let meta = IndexMeta {
            dimension,
            metric: metric.to_string(),
            num_vectors,
            max_degree,
            ef_construction,
            adj_block_size: BLOCK_SIZE,
            entry_set: entry_set.iter().map(|&v| v).collect(),
            adj_layout_version: 3,
            pq: None,
            num_pages: Some(num_pages),
        };
        meta.write_to(&self.meta_path())?;

        Ok(())
    }

    pub fn adj_path(&self) -> PathBuf {
        self.dir.join("adjacency.dat")
    }

    pub fn vec_path(&self) -> PathBuf {
        self.dir.join("vectors.dat")
    }

    pub fn meta_path(&self) -> PathBuf {
        self.dir.join("meta.json")
    }

    pub fn codebook_path(&self) -> PathBuf {
        self.dir.join("pq_codebook.bin")
    }

    pub fn pages_path(&self) -> PathBuf {
        self.dir.join("adjacency_pages.dat")
    }

    pub fn adj_index_path(&self) -> PathBuf {
        self.dir.join("adj_index.dat")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_small_index() {
        let dir = tempfile::tempdir().unwrap();
        let writer = IndexWriter::new(dir.path());

        let num_vectors = 5u32;
        let dim = 4;
        let vectors: Vec<f32> = (0..num_vectors as usize * dim).map(|i| i as f32).collect();
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],
            vec![0, 2, 3],
            vec![0, 1],
            vec![1, 4],
            vec![3],
        ];
        let entry_set = vec![1u32, 0];

        writer
            .write(
                num_vectors,
                dim,
                "l2",
                32,
                200,
                &entry_set,
                &vectors,
                |vid| &adj[vid as usize],
            )
            .unwrap();

        // Verify files exist with correct sizes
        assert!(writer.adj_path().exists());
        assert!(writer.vec_path().exists());
        assert!(writer.meta_path().exists());

        let adj_size = std::fs::metadata(writer.adj_path()).unwrap().len();
        assert_eq!(adj_size, num_vectors as u64 * BLOCK_SIZE as u64);

        let vec_size = std::fs::metadata(writer.vec_path()).unwrap().len();
        assert_eq!(vec_size, (num_vectors as usize * dim * 4) as u64);

        // Verify meta roundtrips
        let meta = IndexMeta::load_from(&writer.meta_path()).unwrap();
        assert_eq!(meta.num_vectors, num_vectors);
        assert_eq!(meta.dimension, dim);
        assert_eq!(meta.entry_set, entry_set);
    }
}
