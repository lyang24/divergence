// SAQ proxy recall evaluation for Cohere 100K.
// Encodes all vectors with SAQ, computes accurate SAQ distances for all queries,
// measures proxy recall@k at various R thresholds.
//
// Build:
//   cd /mnt/nvme/SAQ && mkdir -p build && cd build
//   cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
//   cd /mnt/nvme/divergence/scripts
//   g++ -std=c++20 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl \
//       -I/mnt/nvme/SAQ -I/mnt/nvme/SAQ/saqlib \
//       -o saq_eval saq_eval.cpp \
//       -lfmt -lglog -lgflags -lpthread
//
// Run:
//   ./saq_eval /mnt/nvme/divergence/data/cohere_100k 100000 768 1000 100

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "utils/memory.hpp"

using namespace saqlib;

// Load raw f32 binary (no header, just n*dim floats)
std::vector<float> load_raw_f32(const char *path, size_t expected_floats) {
    std::vector<float> data(expected_floats);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    size_t read = fread(data.data(), sizeof(float), expected_floats, f);
    fclose(f);
    if (read != expected_floats) {
        fprintf(stderr, "Expected %zu floats, got %zu\n", expected_floats, read);
        exit(1);
    }
    return data;
}

// Load ground truth: binary file of u32, shape (nq, k) — our format is flat u32
std::vector<uint32_t> load_gt(const char *path, size_t nq, size_t k) {
    std::vector<uint32_t> gt(nq * k);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open GT file %s\n", path);
        exit(1);
    }
    size_t read = fread(gt.data(), sizeof(uint32_t), nq * k, f);
    fclose(f);
    if (read != nq * k) {
        fprintf(stderr, "GT: expected %zu u32s, got %zu\n", nq * k, read);
        exit(1);
    }
    return gt;
}

float compute_recall(const std::vector<uint32_t> &gt_ids, size_t gt_k,
                     const std::vector<std::pair<float, uint32_t>> &ranked, size_t R, size_t k) {
    // recall@k in top-R of ranked vs gt_ids[0..k]
    size_t hits = 0;
    size_t check_r = std::min(R, ranked.size());
    for (size_t i = 0; i < k && i < gt_k; ++i) {
        uint32_t target = gt_ids[i];
        for (size_t j = 0; j < check_r; ++j) {
            if (ranked[j].second == target) {
                ++hits;
                break;
            }
        }
    }
    return (float)hits / (float)k;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    if (argc < 6) {
        fprintf(stderr, "Usage: %s <data_dir> <n> <dim> <nq> <k> [avg_bits=4] [--norotate] [--eqseg=N]\n", argv[0]);
        return 1;
    }

    const char *data_dir = argv[1];
    size_t n = atoi(argv[2]);
    size_t dim = atoi(argv[3]);
    size_t nq = atoi(argv[4]);
    size_t k = atoi(argv[5]);
    float avg_bits = argc > 6 ? atof(argv[6]) : 4.0f;

    // Parse optional flags from remaining args
    bool use_rotation = true;
    int eqseg = 0;  // 0 = DP segmentation (default)
    for (int i = 7; i < argc; ++i) {
        if (strcmp(argv[i], "--norotate") == 0) {
            use_rotation = false;
        } else if (strncmp(argv[i], "--eqseg=", 8) == 0) {
            eqseg = atoi(argv[i] + 8);
        }
    }

    // Build suffix for export filenames (e.g., "_norot_eq16")
    std::string suffix;
    if (!use_rotation) suffix += "_norot";
    if (eqseg > 0) suffix += "_eq" + std::to_string(eqseg);

    char path[1024];

    // Load vectors
    snprintf(path, sizeof(path), "%s/vectors.bin", data_dir);
    fprintf(stderr, "Loading %zu vectors from %s...\n", n, path);
    auto vectors = load_raw_f32(path, n * dim);

    // Load queries
    snprintf(path, sizeof(path), "%s/queries.bin", data_dir);
    fprintf(stderr, "Loading %zu queries from %s...\n", nq, path);
    auto queries = load_raw_f32(path, nq * dim);

    // Load ground truth
    snprintf(path, sizeof(path), "%s/gt.bin", data_dir);
    fprintf(stderr, "Loading ground truth from %s...\n", path);
    auto gt = load_gt(path, nq, k);

    // Copy to Eigen matrices
    FloatRowMat data_mat(n, dim);
    for (size_t i = 0; i < n; ++i) {
        memcpy(data_mat.row(i).data(), &vectors[i * dim], dim * sizeof(float));
    }

    // --- SAQ encoding ---
    fprintf(stderr, "Configuring SAQ: avg_bits=%.1f, rotation=%s, eqseg=%d\n",
            avg_bits, use_rotation ? "random" : "none", eqseg);
    QuantizeConfig config;
    config.avg_bits = avg_bits;
    config.enable_segmentation = true;
    config.single.random_rotation = use_rotation;
    config.single.use_fastscan = false;  // use CaqSingleEstimator (scalar, no AVX-512 fastscan needed for correctness)
    config.single.caq_adj_rd_lmt = 6;
    if (eqseg > 0) {
        config.seg_eqseg = eqseg;
    }

    SaqDataMaker data_maker(config, dim);

    auto t0 = std::chrono::high_resolution_clock::now();
    data_maker.compute_variance(data_mat);
    auto saq_data = data_maker.return_data();
    auto t1 = std::chrono::high_resolution_clock::now();

    fprintf(stderr, "Variance + segmentation: %.1f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Print quant plan
    fprintf(stderr, "Quant plan (%zu segments):\n", saq_data->quant_plan.size());
    size_t total_bits = 0;
    for (auto &[dim_len, bits] : saq_data->quant_plan) {
        fprintf(stderr, "  segment: %zu dims, %zu bits\n", dim_len, bits);
        total_bits += dim_len * bits;
    }
    fprintf(stderr, "  total: %zu bits/vec = %.1f bytes/vec\n", total_bits, total_bits / 8.0);

    // Create quantizer
    SAQuantizerSingle quantizer(saq_data.get());

    // Encode all vectors
    fprintf(stderr, "Encoding %zu vectors...\n", n);
    SaqSingleDataWrapper wrapper(saq_data->quant_plan);
    size_t mem_size = SaqSingleDataWrapper::calculate_memory_size(saq_data->quant_plan);

    // Use raw pointers since UniqueArray has custom deleter and can't be default-constructed
    std::vector<uint8_t*> encoded(n, nullptr);

    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        encoded[i] = (uint8_t*)aligned_alloc(64, ((mem_size + 63) / 64) * 64);
        memset(encoded[i], 0, mem_size);
        wrapper.set_memory_base(encoded[i]);
        quantizer.quantize(data_mat.row(i), &wrapper);
    }
    t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Encoding: %.1f ms (%.1f us/vec)\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::micro>(t1 - t0).count() / n);

    // --- Evaluate proxy recall ---
    SearcherConfig searcher_cfg;
    searcher_cfg.dist_type = DistType::L2Sqr;  // L2 for now, check if cosine needed
    searcher_cfg.searcher_vars_bound_m = 4.0f;

    // Check if vectors are normalized (cosine)
    {
        float norm = 0;
        for (size_t d = 0; d < dim; ++d) norm += vectors[d] * vectors[d];
        fprintf(stderr, "First vector norm: %.4f (1.0 = normalized/cosine)\n", std::sqrt(norm));
    }

    size_t R_values[] = {100, 200, 500, 1000, 2000, 5000};
    int num_R = sizeof(R_values) / sizeof(R_values[0]);

    fprintf(stderr, "\n=== SAQ Proxy Recall (B=%.1f, %zu segments) ===\n", avg_bits, saq_data->quant_plan.size());
    printf("R\trecall@%zu\tavg_rela_err\tp50_us\tp99_us\n", k);

    // For each query: compute SAQ distance to all vectors, rank, measure recall
    std::vector<float> recalls_per_R(num_R, 0.0f);
    std::vector<double> all_rela_errors;
    std::vector<double> query_times_us;

    for (size_t qi = 0; qi < nq; ++qi) {
        FloatVec query_vec(dim);
        memcpy(query_vec.data(), &queries[qi * dim], dim * sizeof(float));

        auto tq0 = std::chrono::high_resolution_clock::now();

        // Create estimator for this query
        SaqSingleEstimator<DistType::L2Sqr> estimator(*saq_data, searcher_cfg, query_vec);

        // Compute SAQ distance to all vectors
        std::vector<std::pair<float, uint32_t>> dists(n);
        for (size_t vi = 0; vi < n; ++vi) {
            wrapper.set_memory_base(encoded[vi]);
            float d = estimator.compAccurateDist(wrapper);
            dists[vi] = {d, (uint32_t)vi};
        }

        auto tq1 = std::chrono::high_resolution_clock::now();
        query_times_us.push_back(std::chrono::duration<double, std::micro>(tq1 - tq0).count());

        // Sort by distance
        std::sort(dists.begin(), dists.end());

        // Compute recall at each R
        const uint32_t *this_gt = &gt[qi * k];
        for (int ri = 0; ri < num_R; ++ri) {
            recalls_per_R[ri] += compute_recall(
                std::vector<uint32_t>(this_gt, this_gt + k), k, dists, R_values[ri], k);
        }

        // Compute relative error for top-k
        float true_norm_q = query_vec.squaredNorm();
        for (size_t j = 0; j < std::min(k, n); ++j) {
            uint32_t vid = dists[j].second;
            FloatVec data_vec = data_mat.row(vid);
            float true_dist = (query_vec - data_vec).squaredNorm();
            float est_dist = dists[j].first;
            if (true_dist > 1e-10) {
                all_rela_errors.push_back(std::abs(est_dist - true_dist) / true_dist);
            }
        }

        if ((qi + 1) % 100 == 0) {
            fprintf(stderr, "  processed %zu/%zu queries\n", qi + 1, nq);
        }
    }

    // Sort times for percentiles
    std::sort(query_times_us.begin(), query_times_us.end());
    double p50 = query_times_us[nq / 2];
    double p99 = query_times_us[(size_t)(nq * 0.99)];

    // Average relative error
    double avg_rela_err = 0;
    if (!all_rela_errors.empty()) {
        avg_rela_err = std::accumulate(all_rela_errors.begin(), all_rela_errors.end(), 0.0) / all_rela_errors.size();
    }

    for (int ri = 0; ri < num_R; ++ri) {
        float avg_recall = recalls_per_R[ri] / nq;
        printf("%zu\t%.4f\t\t%.6f\t\t%.0f\t%.0f\n", R_values[ri], avg_recall, avg_rela_err, p50, p99);
    }

    // Export unpacked codes + factors for Rust consumption.
    // Format: header + quant_plan + rotation matrices + per-vector(unpacked_codes + factors)
    //
    // Per-vector data (per segment):
    //   codes: u8[dim_padded]   -- full b-bit code per dim, unpacked to 1 byte each
    //   fac_rescale: f32
    //   fac_error: f32
    //   o_l2norm: f32
    snprintf(path, sizeof(path), "%s/saq_unpacked%s.bin", data_dir, suffix.c_str());
    fprintf(stderr, "\nExporting unpacked SAQ data to %s...\n", path);
    {
        FILE *f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "Cannot open export file\n");
            return 1;
        }

        // Header
        uint32_t magic = 0x53415132; // "SAQ2" (unpacked format)
        uint32_t version = 1;
        uint32_t n32 = (uint32_t)n;
        uint32_t dim32 = (uint32_t)dim;
        uint32_t num_segments = (uint32_t)saq_data->quant_plan.size();
        fwrite(&magic, 4, 1, f);
        fwrite(&version, 4, 1, f);
        fwrite(&n32, 4, 1, f);
        fwrite(&dim32, 4, 1, f);
        fwrite(&num_segments, 4, 1, f);

        // Quant plan
        for (auto &[dim_len, bits] : saq_data->quant_plan) {
            uint32_t dl = (uint32_t)dim_len;
            uint32_t b = (uint32_t)bits;
            fwrite(&dl, 4, 1, f);
            fwrite(&b, 4, 1, f);
        }

        // Per-segment rotation matrices (Eigen row-major)
        for (size_t si = 0; si < saq_data->base_datas.size(); ++si) {
            auto &bd = saq_data->base_datas[si];
            uint32_t has_rotation = bd.rotator ? 1 : 0;
            fwrite(&has_rotation, 4, 1, f);
            if (bd.rotator) {
                auto &P = bd.rotator->get_P();
                uint32_t rows = (uint32_t)P.rows();
                uint32_t cols = (uint32_t)P.cols();
                fwrite(&rows, 4, 1, f);
                fwrite(&cols, 4, 1, f);
                // Eigen RowMajor — write row by row to ensure row-major layout
                for (int r = 0; r < P.rows(); ++r) {
                    for (int c = 0; c < P.cols(); ++c) {
                        float v = P(r, c);
                        fwrite(&v, sizeof(float), 1, f);
                    }
                }
            }
        }

        // Per-vector: re-encode each vector to get raw codes (bypasses packed format).
        // This avoids the complex CodeHelper packing/unpacking which has specialized
        // SIMD layouts that differ from the generic froce_compact format.
        size_t total_dim_padded = 0;
        for (auto &[dim_len, bits] : saq_data->quant_plan) {
            total_dim_padded += dim_len;
        }
        size_t per_vec_bytes = total_dim_padded + num_segments * 3 * sizeof(float);

        fprintf(stderr, "Per-vector export (re-encode): %zu code bytes + %u × 12 factor bytes = %zu bytes\n",
                total_dim_padded, num_segments, per_vec_bytes);

        for (size_t i = 0; i < n; ++i) {
            for (size_t si = 0; si < num_segments; ++si) {
                auto &bdata = saq_data->base_datas[si];
                size_t seg_dim = bdata.num_dim_pad;
                size_t seg_offset = 0;
                for (size_t s2 = 0; s2 < si; ++s2) seg_offset += saq_data->base_datas[s2].num_dim_pad;

                // Extract and optionally zero-pad the segment
                FloatVec seg_vec;
                size_t copy_len = std::min(seg_dim, dim - seg_offset);
                if (copy_len < seg_dim) {
                    seg_vec = FloatVec::Zero(seg_dim);
                    seg_vec.head(copy_len) = data_mat.row(i).segment(seg_offset, copy_len);
                } else {
                    seg_vec = data_mat.row(i).segment(seg_offset, seg_dim);
                }

                // Apply rotation (same as QuantizerSingle::quantize)
                FloatVec rotated = bdata.rotator ? seg_vec * bdata.rotator->get_P() : seg_vec;

                // Encode
                CAQEncoder encoder(seg_dim, bdata.num_bits, bdata.cfg);
                QuantBaseCode base_code;
                encoder.encode_and_fac(rotated, base_code, nullptr);

                // Export raw codes as u8
                std::vector<uint8_t> codes(seg_dim);
                for (size_t d = 0; d < seg_dim; ++d) {
                    codes[d] = (uint8_t)base_code.code[d];
                }
                fwrite(codes.data(), 1, seg_dim, f);

                // Export factors (fac_rescale already includes v_mx from rescale_vmx_to1)
                float fac_rescale = base_code.fac_rescale;
                float fac_error = base_code.fac_error;
                float o_l2norm = base_code.o_l2norm;
                fwrite(&fac_rescale, sizeof(float), 1, f);
                fwrite(&fac_error, sizeof(float), 1, f);
                fwrite(&o_l2norm, sizeof(float), 1, f);
            }

            if ((i + 1) % 10000 == 0) {
                fprintf(stderr, "  exported %zu/%zu vectors\n", i + 1, n);
            }
        }

        fclose(f);
        fprintf(stderr, "Export done: %zu vectors × %zu bytes = %.1f MB\n",
                n, per_vec_bytes, n * per_vec_bytes / 1e6);
    }

    // Also export reference distances for cross-validation (first 10 queries × all vectors)
    snprintf(path, sizeof(path), "%s/saq_ref_dists%s.bin", data_dir, suffix.c_str());
    fprintf(stderr, "Exporting reference distances (10 queries × %zu vectors)...\n", n);
    {
        FILE *f = fopen(path, "wb");
        uint32_t nq_ref = 10;
        fwrite(&nq_ref, 4, 1, f);
        uint32_t n32 = (uint32_t)n;
        fwrite(&n32, 4, 1, f);

        for (uint32_t qi = 0; qi < nq_ref; ++qi) {
            FloatVec query_vec(dim);
            memcpy(query_vec.data(), &queries[qi * dim], dim * sizeof(float));
            SaqSingleEstimator<DistType::L2Sqr> estimator(*saq_data, searcher_cfg, query_vec);

            for (size_t vi = 0; vi < n; ++vi) {
                wrapper.set_memory_base(encoded[vi]);
                float d = estimator.compAccurateDist(wrapper);
                fwrite(&d, sizeof(float), 1, f);
            }
        }
        fclose(f);
        fprintf(stderr, "Reference distances exported.\n");
    }

    // Cleanup
    for (size_t i = 0; i < n; ++i) {
        free(encoded[i]);
    }

    return 0;
}
