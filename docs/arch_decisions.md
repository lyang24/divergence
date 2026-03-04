# Architecture Decisions

Decisions that cost us time to learn. Written down so nobody re-learns them.

---

## AD-1: FP32 autovectorization is unreliable baseline (2026-03-02)

**Observation**: Iterator-based `a.iter().zip(b).map(|(x,y)| (x-y)*(x-y)).sum()` is
6-9× slower than hand-written AVX2+FMA on dim=512-768 with `-C target-cpu=native`.

**Evidence** (kernel microbench, dim=768, L1 hot):

| Kernel | ns/call | vs hand SIMD |
|--------|---------|-------------|
| fp32 autovec | 588 | 0.11× |
| fp32 AVX2+FMA | 63 | 1.00× |

**Root cause**: LLVM's autovectorizer fails to fully vectorize the iterator chain in
release mode. It generates partial SIMD with scalar fallback.

**Decision**: `FP32SimdVectorBank` with hand-written AVX2+FMA is the default for all
metrics (L2, Cosine, InnerProduct). Never use `FP32VectorBank` (autovec) in benchmarks
or production. It exists only as a correctness reference.

**Regression gates** (`simd_regression_gate` test):

| Level | Metric | Threshold | Rationale |
|-------|--------|-----------|-----------|
| Kernel-only (L1 hot, dim=768) | L2 | SIMD ≥5× autovec | Observed 6-9×; 5× catches regressions without false positives |
| Kernel-only (L1 hot, dim=768) | Cosine | SIMD ≥2× autovec | Dot product autovectorizes better than L2; 2× is conservative floor |

E2E search gates are not in this test — IO/heap/visited dilute the kernel ratio.
Kernel-only gates are the correct level to catch SIMD codegen regressions.

---

## AD-2: FP16 storage does not provide stable speedup on x86 (2026-03-02)

**Observation**: Fused AVX2+f16c+FMA kernel (load f16 → cvtph2ps → FMA in one loop)
delivers inconsistent results vs FP32 hand SIMD.

**Evidence** (kernel microbench, dim=768):

| Dataset | FP16 vs FP32-SIMD |
|---------|------------------|
| L1 hot (100 vecs) | 0.64× (36% slower) |
| L3 cold (2000 vecs) | 0.98× (tied) |
| E2E search dim=512 | 0.96× (4% slower) |
| E2E search dim=768 | 1.07× (7% faster) |

**Root cause**: `vcvtph2ps` has 3-cycle latency on Haswell/Zen, eating the bandwidth
savings from 2B→4B reduction. Net effect is near-zero at dim 512-768.

**Decision**: FP16 is not worth the complexity (f16c detection, scratch fallback,
precision concerns). Code remains for reference but is not the default path.
Next cheap-distance candidate: int8 scalar quantization.

---

## AD-3: Cosine distance = pre-normalized dot product (2026-03-02)

**Decision**: For cosine metric, vectors are L2-normalized at index build time.
Query is L2-normalized before search. Cosine similarity = dot(q, v).

**Why**: Simplifies both FP32 and int8 kernels:
- FP32: dot kernel needs 2 accumulators instead of 6 (no norm tracking)
- Int8: same — just integer dot, no sqrt in hot loop
- Pre-computed norms stored in `FP32SimdVectorBank` become unnecessary for cosine

**Ranking optimization**: During graph traversal and candidate selection, use raw
`-dot(q, v)` as the distance (negate so that smaller = closer). The `1 - dot` form
is only needed for user-facing distance output. Skipping the constant subtraction
in the hot loop saves one op per distance call and avoids unnecessary floating-point work.

**Matches FAISS**: `METRIC_INNER_PRODUCT` + external normalization = standard approach.

---

## AD-4: Int8 quantization for pre-normalized vectors (2026-03-02)

**Prerequisite**: AD-3 guarantees all vectors are L2-normalized before quantization.
This means every component is in [-1, 1], so the quantization scheme can exploit this.

**Decision** (cosine path — the only production path):

Scale = 1. No training. No per-dataset parameter:
```
code[i] = clamp(round(x[i] * 127), -127, 127)    // signed i8, [-127, 127]
```
This is equivalent to `scale = 1` in a generic `round(x / scale * 127)` formula.
Correct because AD-3 guarantees L2-normalization → all components ∈ [-1, 1].

**Not** `scale = max(|x[i]|)` over all vectors/dims. That formula is **wrong** here:
even though maxabs ≈ 1 for normalized vectors, it introduces a dataset-dependent
parameter that (a) can be poisoned by outliers in the non-normalized case, and
(b) adds unnecessary state. Scale = 1 is unconditionally correct for unit vectors.

**For non-normalized paths** (future, if we add L2/IP without pre-normalization):
Use percentile-based scale (e.g., 99.9th percentile of |x[i]| across all vectors/dims),
NOT global max. Global max is vulnerable to outliers — a single extreme value locks the
scale and crushes the dynamic range for all other values, causing silent recall degradation.
```
scale = percentile(|x[i]|, 99.9)      // robust to outliers
code[i] = clamp(round(x[i] / scale * 127), -127, 127)
```

**Per-dimension scale is WRONG for cosine** (2026-03-02, validated on Cohere 100K):

Per-dim scale (`code[d] = x[d] / scale_d * 127`) distorts the dot product into a
re-weighted inner product `sum_d (q_d * v_d / scale_d^2)`. Dimensions with small
variance get artificially amplified. Result on Cohere 768d: recall dropped from
0.887 (uniform) to 0.377 (per-dim). Per-dim scale is only valid for L2/IP on
non-normalized vectors where you want to equalize dynamic range across dimensions.

**Upgrade path for cosine**: RaBitQ or product quantization — fundamentally different
quantization families that preserve the inner product structure. Uniform int8 + refine
R=2k already achieves 0.962 recall at 1.62x speedup, so this is not urgent.

**Why signed i8, not unsigned u8**:
- Pre-normalized vectors are centered around 0 → signed is natural
- Avoids offset correction in dot product kernel
- SimSIMD and most ANN systems use signed for normalized vectors

**Why avoid -128**: Value -128 has no positive counterpart in i8. Quantize to
[-127, 127] only. This avoids the known `vpmaddubsw` / `sign_epi8` bug where
negating i8 min wraps silently (`sign_epi8(-128) = -128`).

---

## AD-5: Int8 dot kernel uses widen-then-madd chain (2026-03-02)

**Decision**: Use `cvtepi8_epi16` + `madd_epi16` chain, NOT `maddubs_epi16`.
This kernel computes **dot product only** (not L2) — correct because AD-3 mandates
pre-normalization, so cosine distance = `-dot`.

**Instruction chain per 32 elements**:
```
lddqu_si256          load 32×i8
castsi256_si128      low 16 (free)
extracti128_si256    high 16
cvtepi8_epi16 ×2     sign-extend to i16
madd_epi16 ×2        multiply pairs + hadd → i32
add_epi32 ×2         accumulate
```

**Why not `maddubs_epi16`**:
1. Requires unsigned × signed — need abs/sign workaround
2. `sign_epi8(-128) = -128` → silent corruption
3. Uses saturating i16 addition → can clip large products
4. The widen approach is safer, equally fast on Haswell

**Performance constraints** (get these wrong → "correct but slow" kernel):

1. **Unroll ≥2×**: Process 64 elements (2×32) per loop iteration minimum. Without
   unrolling, the load→widen→madd→accumulate dependency chain serializes across
   iterations. 2× unroll lets the CPU overlap independent chains. 4× is better
   if register pressure allows (AVX2 has 16 ymm registers; 4× unroll uses ~12).

2. **Multiple i32 accumulators**: Use at least 2 independent `__m256i` accumulators
   (e.g., `acc0`, `acc1`), reduced to one after the loop. This breaks the
   loop-carried dependency on a single accumulator. Each `add_epi32` has 1-cycle
   latency — with 1 accumulator, the chain serializes; with 2+, the CPU can
   issue adds to different accumulators in parallel.

3. **Load alignment**: Use `lddqu_si256` (unaligned load). Quantized vectors are
   packed contiguously at 1 byte/element — alignment to 32B boundaries is not
   guaranteed. On Haswell+, unaligned loads from aligned addresses have no penalty.
   Explicitly aligning i8 storage to 32B is an option but adds padding waste (up to
   31 bytes per vector).
