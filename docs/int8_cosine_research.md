# Int8 Cosine Scalar Quantization — Research Distillation

Sources: SimSIMD (Ash Vardanian), FAISS (Meta), blog benchmarks.

---

## 1. The Instruction Chain (AVX2, No VNNI)

**Correct approach**: sign-extend i8→i16, then `_mm256_madd_epi16`.

```
Per iteration (32 int8 values):
  load 32×i8                    _mm256_lddqu_si256
  split low 16                  _mm256_castsi256_si128         (zero-cost)
  split high 16                 _mm256_extracti128_si256(v, 1)
  sign-extend low i8→i16        _mm256_cvtepi8_epi16           (16 → 16×i16)
  sign-extend high i8→i16       _mm256_cvtepi8_epi16           (16 → 16×i16)
  multiply+hadd pairs i16→i32   _mm256_madd_epi16              ×2 (low/high)
  accumulate i32                _mm256_add_epi32               ×2
```

For unsigned u8 (FAISS style): replace `cvtepi8_epi16` with `cvtepu8_epi16`.

**Horizontal reduction** (once, after loop):
```
  _mm256_castsi256_si128 + _mm256_extracti128_si256(v,1)   → 128-bit halves
  _mm_add_epi32                                             → 4×i32
  _mm_hadd_epi32 × 2                                       → scalar i32
  _mm_cvtsi128_si32
```

## 2. DO NOT Use `vpmaddubsw` / `_mm256_maddubs_epi16`

SimSIMD documents this explicitly (see `_mm256_maddubs_epi16` usage in dot.h):

- Requires one operand unsigned, one signed → need `abs_epi8` + `sign_epi8` workaround.
- **Fatal bug**: `sign_epi8(-128) = -128` (negating i8 min wraps). Silent wrong results.
- Also: `maddubs` uses **saturating** i16 addition. Large products clip silently.
- The widen approach (cvtepi8→madd_epi16) is both **safer and simpler**.

## 3. Cosine Distance: Two Strategies

### Strategy A: Full cosine with 6 accumulators (SimSIMD approach)

Track `dot(a,b)`, `||a||²`, `||b||²` in the kernel — 6 accumulators total.
Final: `1.0 - ab * rsqrt(a2) * rsqrt(b2)` with Newton-Raphson refinement.

**Pros**: No preprocessing. Works on raw quantized vectors.
**Cons**: 3× the FMA work per element. 6 register pressure.

### Strategy B: Pre-normalize → dot only (FAISS approach) ← RECOMMENDED

1. L2-normalize vectors **before** quantization.
2. At query time, L2-normalize query **before** quantization.
3. Cosine similarity = dot product (since ||a||=||b||=1).
4. Cosine distance = 1 - dot.
5. Kernel needs only **2 accumulators** (dot_lo, dot_hi).

**Pros**: 3× less work per element. Simpler kernel. Same accuracy.
**Cons**: Must normalize at index build time. Query normalization is O(dim), negligible.

This is what FAISS does: `METRIC_INNER_PRODUCT` + external normalization = cosine.

**We use Strategy B.**

## 4. FAISS Quantization Formula

### Per-dimension uniform quantizer (QT_8bit):

**Training** (offline, once per index):
```
For each dimension i:
  vmin[i] = min(all_vectors[:, i])
  vmax[i] = max(all_vectors[:, i])
  vdiff[i] = vmax[i] - vmin[i]    // stored as scale
```

**Encode** (float32 → uint8):
```
code[i] = clamp((x[i] - vmin[i]) / vdiff[i], 0, 1) * 255
```

**Decode** (uint8 → float32):
```
x[i] = vmin[i] + (code[i] / 255.0) * vdiff[i]
```

**Quantization error per component**: ≤ vdiff[i] / 510.

### For pre-normalized cosine vectors:

Components are in ~[-1/√dim, +1/√dim] for typical data.
dim=768 → range ~[-0.036, 0.036] per component.
Per-dimension min/max captures the actual range tightly.

### For pre-normalized vectors (our path — AD-3):

Since L2-normalization bounds all components to [-1, 1], no training step is needed:
```
code[i] = clamp(round(x[i] * 127), -127, 127)     // signed i8, [-127, 127]
```
No scale parameter. Direct mapping. This is the MVP quantizer.

### For non-normalized vectors (future path):

Use percentile-based scale to avoid outlier poisoning:
```
scale = percentile(|x[i]|, 99.9)      // NOT global max — outliers lock scale
code[i] = clamp(round(x[i] / scale * 127), -127, 127)
```

**Never use `max(|x[i]|)` over all vectors/dims** — a single extreme outlier
crushes dynamic range for all values, causing silent recall degradation.

Upgrade to per-dim (FAISS QT_8bit) if recall is insufficient.

## 5. Performance Expectations

### Kernel throughput (from blog benchmarks, 1536-d):

| Precision | AVX-512 (ns) | Relative |
|-----------|-------------|----------|
| f32       | 87.4        | 1.0×     |
| f16       | 51.8        | 1.7×     |
| i8 VNNI   | 25.9        | 3.4×     |

**On AVX2 (no VNNI)**: the widen approach adds overhead vs VNNI.
Expected: **int8 ≈ 2-2.5× faster than f32** on AVX2 Haswell.

### Why int8 wins where FP16 didn't:

- FP16: `vcvtph2ps` has 3-cycle latency → conversion eats the bandwidth savings.
- Int8: `cvtepi8_epi16` has 1-cycle latency → widening is essentially free.
- Int8: `madd_epi16` has 1-cycle throughput on Haswell → same as FP32 FMA.
- Int8: 4× less data to load (1 byte vs 4 bytes) → real bandwidth win.

### Bandwidth analysis (dim=768):

| Format | Bytes/vector | Cache lines (64B) |
|--------|-------------|-------------------|
| f32    | 3072        | 48                |
| f16    | 1536        | 24                |
| i8     | 768         | 12                |

At ~10ns/line from L3: f32=480ns load, i8=120ns load. Real win.

## 6. Overflow Safety

For dim=768, signed i8:
- Max single `madd_epi16` output: 2 × 127² = 32,258
- Over full 768 dims: 384 × 32,258 = 12.4M → fits i32 (max 2.1B)
- Safe up to ~66,000 dimensions.

## 7. Implementation Plan for Divergence

### Data flow:
```
Build time:
  vectors_f32 → L2-normalize → quantize(scale, offset) → vectors_i8 + metadata

Query time:
  query_f32 → L2-normalize → quantize(same scale) → query_i8

Search:
  Cheap stage:  int8_dot(query_i8, vector_i8)  → approximate cosine ranking
  Refine stage: fp32_cosine(query_f32, vector_f32) → exact top-k
```

### What to build (in order):

1. **`ScalarQuantizer`** struct: `scale: f32`, `offset: f32`, methods:
   - `train(vectors: &[f32], dim: usize)` — compute global scale
   - `encode(src: &[f32]) -> Vec<i8>` — quantize one vector
   - `encode_batch(src: &[f32], dim: usize) -> Vec<i8>` — quantize N vectors

2. **`Int8VectorBank`** implementing `VectorBank`:
   - Stores `&[i8]` (pre-quantized, pre-normalized vectors)
   - `distance()` calls the AVX2 int8 dot kernel
   - Query quantization happens once in `distance()` via cached quantized query

3. **AVX2 int8 dot kernel**: `dot_i8_avx(a: &[i8], b: &[i8]) -> i32`
   - 32 elements per iteration, 2-way accumulator
   - Horizontal reduction → scalar i32
   - Final: `1.0 - (dot as f32) * scale_correction`

4. **Wire into `disk_graph_search_refine`**:
   - `cheap_bank = Int8VectorBank`
   - `exact_bank = FP32SimdVectorBank(Cosine)`

### Key decisions locked:

| Decision | Choice | Why |
|----------|--------|-----|
| Metric | Cosine via pre-normalized dot | Simpler kernel (2 acc vs 6) |
| Quantization | Global symmetric i8 [-127,127] | Simple, upgrade to per-dim later |
| Instruction chain | cvtepi8_epi16 + madd_epi16 | Safe (-128 bug avoided) |
| Accumulator | i32 | Safe to 66K dims |
| Query handling | Quantize once, reuse | O(dim) per query, negligible |

## 8. SimSIMD References

- `simsimd/include/simsimd/dot.h` — int8 dot AVX2 kernel, `_mm256_maddubs_epi16` bug discussion
- `simsimd/include/simsimd/spatial.h` — int8 cosine AVX2, rsqrt + Newton-Raphson normalization
- `simsimd/include/simsimd/dot.h` — `reduce_i32x8_haswell` horizontal reduction

## 9. FAISS References

- `faiss/impl/ScalarQuantizer.cpp` — quantization formula (QT_8bit encode/decode)
- `faiss/impl/ScalarQuantizerCodec_avx.h` — AVX2 uint8 dot kernel
- `faiss/impl/ScalarQuantizer.h` — QT_8bit type definitions
