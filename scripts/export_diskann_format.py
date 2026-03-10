#!/usr/bin/env python3
"""Export Cohere dataset to DiskANN .bin format (8-byte header).

Reads from: data/cohere_100k/ (headerless .bin + meta.txt)
Writes to:  data/cohere_100k/diskann/
    base_f32.bin  - [u32 npoints][u32 ndims] + row-major f32
    query_f32.bin - [u32 nqueries][u32 ndims] + row-major f32
    gt.bin        - [u32 nqueries][u32 k] + per-query: [u32 k_i][u32 * k_i ids][f32 * k_i dists]
    checksums.sha256 - SHA256 of raw payloads (after header) for cross-format consistency

DiskANN ground truth format (read_groundtruth in DiskANN):
    Header: [nqueries: u32][k: u32]
    Per query: k neighbor IDs (u32) followed by k distances (f32)
    (No per-row k_i prefix — DiskANN's load_truthset reads flat rows of k ids then k dists)

Usage:
    python3 scripts/export_diskann_format.py [--src data/cohere_100k] [--normalize]
"""

import argparse
import hashlib
import os
import struct
import sys

import numpy as np


def write_bin_with_header(path: str, data: np.ndarray) -> str:
    """Write DiskANN .bin: [u32 nrows][u32 ncols] + flat row-major data.
    Returns SHA256 hex of the payload (excluding header)."""
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    nrows, ncols = data.shape
    payload = data.tobytes()
    sha = hashlib.sha256(payload).hexdigest()
    with open(path, "wb") as f:
        f.write(struct.pack("<II", nrows, ncols))
        f.write(payload)
    return sha


def check_norms(name: str, data: np.ndarray) -> tuple:
    """Check L2 norms. Returns (min, max, mean)."""
    norms = np.linalg.norm(data, axis=1)
    lo, hi, mu = float(norms.min()), float(norms.max()), float(norms.mean())
    print(f"  {name} norms: min={lo:.6f} max={hi:.6f} mean={mu:.6f}")
    return lo, hi, mu


def main():
    parser = argparse.ArgumentParser(description="Export to DiskANN .bin format")
    parser.add_argument("--src", default="data/cohere_100k",
                        help="Source directory with headerless .bin + meta.txt")
    parser.add_argument("--normalize", action="store_true",
                        help="Force L2-normalize base and query vectors")
    parser.add_argument("--verify", action="store_true",
                        help="Read back and verify round-trip correctness")
    parser.add_argument("--metric", default="l2",
                        choices=["l2", "cosine"],
                        help="Distance metric for GT distances (default: l2, "
                             "correct when vectors are unit-normalized)")
    args = parser.parse_args()

    src = args.src
    dst = os.path.join(src, "diskann")
    os.makedirs(dst, exist_ok=True)

    # --- Read meta ---
    meta_path = os.path.join(src, "meta.txt")
    with open(meta_path) as f:
        nums = [int(line.strip()) for line in f if line.strip()]
    n, nq, dim, k = nums[0], nums[1], nums[2], nums[3]
    print(f"Dataset: n={n}, nq={nq}, dim={dim}, k={k}")

    # --- Load base vectors ---
    print(f"Loading base vectors from {src}/vectors.bin ...")
    base = np.fromfile(os.path.join(src, "vectors.bin"), dtype=np.float32).reshape(n, dim)
    base_lo, base_hi, base_mu = check_norms("base", base)

    # --- Load query vectors ---
    print(f"Loading query vectors from {src}/queries.bin ...")
    queries = np.fromfile(os.path.join(src, "queries.bin"), dtype=np.float32).reshape(nq, dim)
    query_lo, query_hi, query_mu = check_norms("query", queries)

    # --- Normalization check ---
    norm_threshold = 0.01  # 1% deviation from unit norm
    base_needs_norm = abs(base_mu - 1.0) > norm_threshold or abs(base_hi - base_lo) > norm_threshold
    query_needs_norm = abs(query_mu - 1.0) > norm_threshold or abs(query_hi - query_lo) > norm_threshold

    if base_needs_norm or query_needs_norm:
        if args.normalize:
            print("  WARNING: Norms deviate from 1.0. Normalizing as requested.")
            if base_needs_norm:
                norms = np.linalg.norm(base, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                base /= norms
                check_norms("base (after normalize)", base)
            if query_needs_norm:
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                queries /= norms
                check_norms("query (after normalize)", queries)
        else:
            print("  WARNING: Norms deviate from 1.0!")
            print("  If using L2 metric as proxy for cosine, this WILL give wrong rankings.")
            print("  Either re-run with --normalize, or use --metric cosine in DiskANN.")
            if abs(base_mu - 1.0) > 0.05:
                print("  ERROR: Base norms are far from 1.0. Aborting.")
                print("  Re-run with --normalize to force L2-normalization.")
                sys.exit(1)
    else:
        print("  Norms OK: vectors are unit-normalized (L2 metric = cosine ranking)")

    # --- Load ground truth ---
    print(f"Loading ground truth from {src}/gt.bin ...")
    gt = np.fromfile(os.path.join(src, "gt.bin"), dtype=np.uint32).reshape(nq, k)

    # --- Compute GT distances ---
    print(f"Computing GT distances (metric={args.metric}) ...")
    gt_dists = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        neighbors = base[gt[i]]  # (k, dim)
        if args.metric == "l2":
            # L2 distance (on unit sphere, this ranks same as cosine)
            diff = queries[i] - neighbors
            gt_dists[i] = np.sum(diff * diff, axis=1)
        else:
            # cosine distance = 1 - dot(q, x)
            gt_dists[i] = 1.0 - np.sum(queries[i] * neighbors, axis=1)

    # --- Write DiskANN format files ---
    print(f"\nWriting DiskANN format files to {dst}/")

    base_path = os.path.join(dst, "base_f32.bin")
    base_sha = write_bin_with_header(base_path, base)
    print(f"  {base_path}: {os.path.getsize(base_path) / 1e6:.1f} MB")

    query_path = os.path.join(dst, "query_f32.bin")
    query_sha = write_bin_with_header(query_path, queries)
    print(f"  {query_path}: {os.path.getsize(query_path) / 1e6:.1f} MB")

    # DiskANN GT format (load_truthset):
    #   Header: [u32 nqueries][u32 k]
    #   Then ALL ids flat (nq * k u32), row-major
    #   Then ALL dists flat (nq * k f32), row-major
    #   NOT interleaved per-query — all IDs block first, then all dists block.
    gt_path = os.path.join(dst, "gt.bin")
    with open(gt_path, "wb") as f:
        f.write(struct.pack("<II", nq, k))
        f.write(gt.tobytes())           # all nq*k IDs (u32), row-major
        f.write(gt_dists.tobytes())     # all nq*k distances (f32), row-major
    gt_sha = hashlib.sha256(open(gt_path, "rb").read()[8:]).hexdigest()
    print(f"  {gt_path}: {os.path.getsize(gt_path) / 1e6:.1f} MB")

    # --- Also compute SHA256 of original headerless files for consistency ---
    orig_base_sha = hashlib.sha256(
        open(os.path.join(src, "vectors.bin"), "rb").read()
    ).hexdigest()
    orig_query_sha = hashlib.sha256(
        open(os.path.join(src, "queries.bin"), "rb").read()
    ).hexdigest()

    # If no normalization was applied, payloads must match
    if not args.normalize:
        assert base_sha == orig_base_sha, \
            f"Base payload mismatch! headered={base_sha} headerless={orig_base_sha}"
        assert query_sha == orig_query_sha, \
            f"Query payload mismatch! headered={query_sha} headerless={orig_query_sha}"
        print("\n  Payload consistency verified: headered == headerless (same bytes)")

    # --- Write checksums ---
    checksum_path = os.path.join(dst, "checksums.sha256")
    with open(checksum_path, "w") as f:
        f.write(f"# SHA256 of raw payloads (excluding 8-byte headers)\n")
        f.write(f"# Metric: {args.metric}\n")
        f.write(f"# Normalized: {args.normalize}\n")
        f.write(f"# n={n} nq={nq} dim={dim} k={k}\n")
        f.write(f"base_f32.bin  {base_sha}\n")
        f.write(f"query_f32.bin {query_sha}\n")
        f.write(f"gt.bin        {gt_sha}\n")
        if not args.normalize:
            f.write(f"# Original headerless files (should match):\n")
            f.write(f"vectors.bin   {orig_base_sha}\n")
            f.write(f"queries.bin   {orig_query_sha}\n")
    print(f"  {checksum_path}: written")

    # --- Verify round-trip ---
    if args.verify:
        print("\n--- Round-trip verification ---")
        # Read back headered base
        with open(base_path, "rb") as f:
            rn, rd = struct.unpack("<II", f.read(8))
            assert rn == n and rd == dim, f"Header mismatch: {rn}x{rd} vs {n}x{dim}"
            rbase = np.frombuffer(f.read(), dtype=np.float32).reshape(rn, rd)
        assert np.array_equal(base, rbase), "Base round-trip FAILED"
        print(f"  base: OK ({rn}x{rd})")

        # Read back headered query
        with open(query_path, "rb") as f:
            rn, rd = struct.unpack("<II", f.read(8))
            assert rn == nq and rd == dim, f"Header mismatch: {rn}x{rd} vs {nq}x{dim}"
            rquery = np.frombuffer(f.read(), dtype=np.float32).reshape(rn, rd)
        assert np.array_equal(queries, rquery), "Query round-trip FAILED"
        print(f"  query: OK ({rn}x{rd})")

        # Read back GT (DiskANN format: header + all IDs block + all dists block)
        with open(gt_path, "rb") as f:
            rnq, rk = struct.unpack("<II", f.read(8))
            assert rnq == nq and rk == k, f"GT header mismatch: {rnq}x{rk} vs {nq}x{k}"
            rids = np.frombuffer(f.read(nq * k * 4), dtype=np.uint32).reshape(nq, k)
            rdists = np.frombuffer(f.read(nq * k * 4), dtype=np.float32).reshape(nq, k)
            assert np.array_equal(gt, rids), "GT ids mismatch"
            assert np.allclose(gt_dists, rdists, atol=1e-6), "GT dists mismatch"
        print(f"  gt: OK ({nq}x{k})")
        print("  All round-trip checks PASSED")

    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"Files written to: {dst}/")
    print(f"  base_f32.bin:  {n} x {dim} f32  sha256={base_sha[:16]}...")
    print(f"  query_f32.bin: {nq} x {dim} f32  sha256={query_sha[:16]}...")
    print(f"  gt.bin:        {nq} x {k}  (ids:u32 + dists:f32)  sha256={gt_sha[:16]}...")
    print(f"Metric: {args.metric}")
    print(f"Normalized: {args.normalize}")
    if not args.normalize:
        print(f"Payload matches headerless originals: YES")


if __name__ == "__main__":
    main()
