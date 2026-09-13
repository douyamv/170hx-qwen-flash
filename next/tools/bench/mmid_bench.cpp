// MUL_MAT_ID micro benchmark: shapes of Qwen3.8-Flash-Next routed experts.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_quant(ggml_backend_t, ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t total = ggml_nbytes(t);
    const size_t bs = ggml_type_size(t->type);
    const size_t chunk_blocks = (64u << 20) / bs;
    std::vector<uint8_t> buf(chunk_blocks * bs);
    for (size_t off = 0; off < total; off += buf.size()) {
        size_t n = std::min(buf.size(), total - off);
        for (auto & b : buf) b = (uint8_t) rng();
        // first two fields of Q4_K / Q5_1 blocks are fp16 scales: keep them small and finite
        for (size_t i = 0; i + 4 <= n; i += bs) {
            uint16_t h = 0x2400 | (rng() & 0xff); // ~2e-3 .. fp16
            memcpy(buf.data() + i, &h, 2);
            memcpy(buf.data() + i + 2, &h, 2);
        }
        ggml_backend_tensor_set(t, buf.data(), off, n);
    }
}

int main(int argc, char ** argv) {
    const int n_expert = 512, n_used = 10;
    const bool skew = argc > 1 && atoi(argv[1]) == 1;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no cuda\n"); return 1; }

    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * up   = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, 2560, 640, n_expert);
    ggml_tensor * down = ggml_new_tensor_3d(wctx, GGML_TYPE_Q5_1, 640, 2560, n_expert);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    fill_quant(be, up, 1); fill_quant(be, down, 2);
    fprintf(stderr, "weights: up %.0f MiB, down %.0f MiB, skew=%d\n", ggml_nbytes(up)/1048576.0, ggml_nbytes(down)/1048576.0, skew);

    std::mt19937 rng(42);
    std::vector<double> zipf(n_expert);
    double zs = 0; for (int e = 0; e < n_expert; ++e) { zipf[e] = 1.0 / std::pow(e + 1, 0.6); zs += zipf[e]; }
    std::discrete_distribution<int> pick(zipf.begin(), zipf.end());

    for (int n_tok : {256, 512, 1024, 2048}) {
        ggml_init_params p = { ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, n_tok);
        ggml_tensor * xd  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 640, n_used, n_tok);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        ggml_tensor * o1 = ggml_mul_mat_id(ctx, up, x, ids);
        ggml_tensor * o2 = ggml_mul_mat_id(ctx, down, xd, ids);
        ggml_cgraph * g1 = ggml_new_graph(ctx); ggml_build_forward_expand(g1, o1);
        ggml_cgraph * g2 = ggml_new_graph(ctx); ggml_build_forward_expand(g2, o2);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        std::vector<float> xv(2560 * (size_t) n_tok), xdv(640 * (size_t) n_used * n_tok);
        std::normal_distribution<float> nd(0, 1);
        for (auto & v : xv) v = nd(rng);
        for (auto & v : xdv) v = nd(rng);
        std::vector<int32_t> iv(n_used * (size_t) n_tok);
        std::vector<int> load(n_expert, 0);
        for (int t = 0; t < n_tok; ++t) {
            std::vector<char> used(n_expert, 0);
            for (int k = 0; k < n_used; ++k) {
                int e; do { e = skew ? pick(rng) : (int)(rng() % n_expert); } while (used[e]);
                used[e] = 1; iv[t * n_used + k] = e; load[e]++;
            }
        }
        int mx = 0; for (int l : load) mx = std::max(mx, l);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(xd, xdv.data(), 0, ggml_nbytes(xd));
        ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));

        auto bench = [&](ggml_cgraph * g, ggml_tensor * out, const char * name) {
            for (int i = 0; i < 3; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
            const int reps = 10;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::vector<float> ov(ggml_nelements(out));
            ggml_backend_tensor_get(out, ov.data(), 0, ggml_nbytes(out));
            double sum = 0; for (float v : ov) sum += std::fabs((double) v);
            printf("%-5s n_tok=%5d max_load=%3d avg_load=%5.1f  %8.3f ms/call  %7.1f us/token  checksum=%.9e\n",
                   name, n_tok, mx, n_tok * (double) n_used / n_expert, ms, ms * 1000.0 / n_tok, sum);
        };
        bench(g1, o1, "gate");
        bench(g2, o2, "down");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
