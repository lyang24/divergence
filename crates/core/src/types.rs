/// Unique identifier for a vector in the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct VectorId(pub u32);

/// Physical block identifier for on-disk adjacency blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// Distance metric used for vector comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    L2,
    Cosine,
    InnerProduct,
}
