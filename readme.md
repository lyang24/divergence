# Project Divergence

Project Divergence is a single-node, NVMe-native retrieval engine designed for high-throughput, low-latency vector search.

It focuses on one goal:

Fully utilize modern NVMe SSDs and CPU cores without relying on mmap or OS page cache.

The system is designed around explicit buffer management, asynchronous IO scheduling, and record-level tiering to achieve predictable tail latency under memory constraints.

---

## Motivation

Recent research shows:

- CPU is often the bottleneck in SSD-based vector search.
- NVMe bandwidth is severely underutilized in existing systems.
- Page-level caching mismatches vector access patterns.
- Record-level tiering significantly improves performance under skew.
- Two-stage search (compressed first, exact refine later) is consistently effective.

Most existing systems still rely on:

- mmap and implicit OS page cache
- page-granularity buffer management
- fixed index traversal strategies
- limited IO scheduling control

Project Divergence is built to address these issues directly.

---

## Design Goals

1. Maximize NVMe utilization without overwhelming CPU.
2. Provide predictable p99 latency under limited DRAM.
3. Support continuous performance-to-memory scaling.
4. Avoid global index rebuilds during layout evolution.
5. Maintain explicit control over IO and compute scheduling.

---

## Architecture Overview

Divergence is organized into two layers:

### Data Plane

The performance-critical execution engine.

Responsibilities:

- Thread-per-core execution model
- Coroutine-based async scheduling
- Batched NVMe IO using io_uring
- Explicit queue depth control
- Record-level buffer pool
- Two-stage scoring pipeline

### Control Plane

Responsible for:

- Heat tracking
- Object placement decisions
- Block reorganization
- Tier migration policies
- Background compaction

---

## Storage Model

The index is decomposed into independent object types:

- Routing metadata
- Candidate blocks (adjacency lists or posting lists)
- Compressed vector codes
- Exact vectors for refinement
- Versioned manifest metadata

Object storage is versioned locally.

Each object type can be independently cached, promoted, or evicted.

---

## Execution Pipeline

Each query follows a fixed pipeline:

1. Router  
   Identify candidate partitions or entry points.

2. Candidate Producer  
   Generate candidate blocks.

3. Scorer  
   Compute approximate distances using compressed codes.

4. Pruner  
   Eliminate low-potential candidates.

5. Refiner  
   Load exact vectors from NVMe and compute final distances.

Two-stage scoring minimizes NVMe reads while preserving recall.

---

## NVMe-Native IO Model

- io_uring with O_DIRECT
- Fixed buffer pools
- Controlled submission/completion queue depth
- Batched 4KB reads aligned to physical page size
- Overlapped compute and IO via coroutines

The goal is to approach device-level throughput while keeping CPU fully utilized.

---

## Buffer Management

Divergence uses record-level caching rather than page-level caching.

Advantages:

- Avoids lukewarm pages under skew
- Captures fine-grained access patterns
- Enables selective promotion of routing metadata
- Reduces memory waste

Eviction policy is heat-aware and object-type aware.

---

## Memory Scaling Model

Divergence is designed to degrade gracefully as memory decreases:

- With large DRAM: most routing metadata and compressed codes stay in memory.
- With limited DRAM: compressed codes remain hot; exact vectors fetched on demand.
- With minimal DRAM: search remains functional with increased NVMe reads.

Performance scales continuously rather than collapsing.

---

## Concurrency Model

- Thread-per-core
- No global locks
- Lock-sharded structures
- Optimistic read paths
- Backpressure via IO queue depth control

The system prioritizes stable latency under load.

---

## Roadmap

Phase 1:
- Single-node NVMe-native index
- Thread-per-core execution
- Record-level buffer pool
- Two-stage scoring

Phase 2:
- Heat-aware tier migration
- Block reorganization without rebuild
- Memory scaling benchmarks

Phase 3:
- Multi-tenant resource accounting
- Optional GPU acceleration
- Distributed resource graph

---

## Non-Goals (for now)

- Distributed execution
- Object storage as primary backend
- Embedding generation
- Learned routing models

These may be explored after the single-node core stabilizes.

---

## Status

Early development stage.

Focused on building a high-performance single-node NVMe-native retrieval core.
