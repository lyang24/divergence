pub mod adjacency;
pub mod meta;
pub mod pq_store;
pub mod vectors;
pub mod writer;

pub use adjacency::{
    decode_adj_block, decode_adj_block_view, encode_adj_block, encode_adj_block_v2,
    max_degree_v2, write_adjacency_file_v2, AdjBlockView, BLOCK_SIZE,
    // v3: page-packed adjacency
    AdjIndexEntry, bfs_reorder_graph, heavy_edge_reorder_graph, load_adj_index,
    neighbor_run_reorder_graph, packed_record_size, page_record_vid,
    twpp_reorder_graph, write_packed_adjacency,
};
pub use meta::IndexMeta;
pub use vectors::{f16_to_f32, load_vectors, write_vectors_file, write_vectors_fp16_file, write_vectors_int8_file};
pub use writer::IndexWriter;
