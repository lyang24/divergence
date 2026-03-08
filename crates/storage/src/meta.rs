use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use divergence_core::MetricType;

/// Index metadata, serialized as meta.json in the index directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub dimension: usize,
    pub metric: String,
    pub num_vectors: u32,
    pub max_degree: usize,
    pub ef_construction: usize,
    pub adj_block_size: usize,
    pub entry_set: Vec<u32>,
    /// Adjacency block layout version: 1 = IDs only, 2 = IDs + inline PQ codes.
    #[serde(default = "default_adj_layout_version")]
    pub adj_layout_version: u32,
    /// PQ configuration (present only when adj_layout_version >= 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pq: Option<PqMeta>,
    /// Number of pages in adjacency_pages.dat (v3 only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_pages: Option<u32>,
}

fn default_adj_layout_version() -> u32 {
    1
}

/// PQ codebook metadata stored in meta.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqMeta {
    /// Number of subquantizers (M). dim must be divisible by M.
    pub num_subquantizers: usize,
    /// Number of centroids per subspace (always 256 for PQ×8).
    pub num_centroids: usize,
    /// Dimension of each subspace (= dimension / num_subquantizers).
    pub subspace_dim: usize,
    /// Distance metric used for PQ training ("l2" or "ip").
    pub metric: String,
    /// Filename of the codebook file (relative to index directory).
    pub codebook_file: String,
}

impl IndexMeta {
    pub fn metric_type(&self) -> MetricType {
        match self.metric.as_str() {
            "l2" => MetricType::L2,
            "cosine" => MetricType::Cosine,
            "ip" | "inner_product" => MetricType::InnerProduct,
            _ => panic!("unknown metric: {}", self.metric),
        }
    }

    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    pub fn load_from(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}
