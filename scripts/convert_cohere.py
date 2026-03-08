#!/usr/bin/env python3
"""Convert Cohere parquet dataset to binary files for Rust benchmarks.

Usage:
    python3 scripts/convert_cohere.py [--n-vectors 100000] [--k 100]

Reads from: /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/
Writes to:  data/cohere_100k/
    vectors.bin   - f32 flat array, shape (n, 768)
    queries.bin   - f32 flat array, shape (nq, 768)
    gt.bin        - u32 flat array, shape (nq, k)
    meta.txt      - n, nq, dim, k on separate lines
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=100_000,
                        help="Number of train vectors to use (default 100000)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of ground-truth neighbors (default 100)")
    parser.add_argument("--src", type=str,
                        default="/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m",
                        help="Source parquet directory")
    parser.add_argument("--dst", type=str, default=None,
                        help="Output directory (default: data/cohere_{n}k/)")
    args = parser.parse_args()

    n = args.n_vectors
    k = args.k
    dst = args.dst or f"data/cohere_{n // 1000}k"
    os.makedirs(dst, exist_ok=True)

    # --- Load queries ---
    print(f"Loading queries from {args.src}/test.parquet ...")
    t0 = time.time()
    qt = pq.read_table(os.path.join(args.src, "test.parquet"))
    queries_list = qt.column("emb").to_pylist()
    nq = len(queries_list)
    dim = len(queries_list[0])
    queries = np.array(queries_list, dtype=np.float32)  # (nq, dim)
    print(f"  {nq} queries, dim={dim}, {time.time()-t0:.1f}s")

    # --- Load train vectors (subset) ---
    print(f"Loading {n} train vectors from {args.src}/shuffle_train.parquet ...")
    t0 = time.time()
    pf = pq.ParquetFile(os.path.join(args.src, "shuffle_train.parquet"))
    vectors_list = []
    for batch in pf.iter_batches(batch_size=10_000, columns=["emb"]):
        for emb in batch.column("emb").to_pylist():
            vectors_list.append(emb)
            if len(vectors_list) >= n:
                break
        if len(vectors_list) >= n:
            break
    vectors = np.array(vectors_list[:n], dtype=np.float32)  # (n, dim)
    print(f"  {vectors.shape[0]} vectors loaded, {time.time()-t0:.1f}s")

    # --- L2-normalize (cosine metric) ---
    print("L2-normalizing vectors and queries ...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    qnorms = np.linalg.norm(queries, axis=1, keepdims=True)
    qnorms[qnorms == 0] = 1.0
    queries /= qnorms

    # --- Compute brute-force ground truth (cosine = dot on normalized) ---
    print(f"Computing brute-force top-{k} (cosine) ...")
    t0 = time.time()
    gt = np.empty((nq, k), dtype=np.uint32)
    batch_size = 100  # queries per batch to avoid OOM
    for i in range(0, nq, batch_size):
        end = min(i + batch_size, nq)
        # (batch, dim) @ (dim, n) -> (batch, n)
        sims = queries[i:end] @ vectors.T
        # top-k indices (descending similarity = ascending -sim)
        topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        # sort the top-k by similarity (descending)
        for j in range(end - i):
            order = np.argsort(-sims[j, topk_idx[j]])
            gt[i + j] = topk_idx[j][order].astype(np.uint32)
        if (i + batch_size) % 500 == 0 or end == nq:
            print(f"  {end}/{nq} queries done")
    print(f"  Ground truth computed, {time.time()-t0:.1f}s")

    # --- Write binary files ---
    vec_path = os.path.join(dst, "vectors.bin")
    print(f"Writing {vec_path} ({vectors.nbytes / 1e6:.1f} MB) ...")
    vectors.tofile(vec_path)

    q_path = os.path.join(dst, "queries.bin")
    print(f"Writing {q_path} ({queries.nbytes / 1e6:.1f} MB) ...")
    queries.tofile(q_path)

    gt_path = os.path.join(dst, "gt.bin")
    print(f"Writing {gt_path} ({gt.nbytes / 1e6:.1f} MB) ...")
    gt.tofile(gt_path)

    meta_path = os.path.join(dst, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"{vectors.shape[0]}\n{nq}\n{dim}\n{k}\n")
    print(f"Written {meta_path}")

    # --- Sanity check ---
    print("\n--- Sanity Check ---")
    print(f"Vector norms (first 5): {np.linalg.norm(vectors[:5], axis=1)}")
    print(f"Query norms (first 5): {np.linalg.norm(queries[:5], axis=1)}")
    print(f"GT[0] first 5 neighbors: {gt[0, :5]}")
    sims0 = queries[0] @ vectors[gt[0, :5]].T
    print(f"GT[0] first 5 similarities: {sims0}")
    print(f"\nDone. Files in {dst}/")


if __name__ == "__main__":
    main()
