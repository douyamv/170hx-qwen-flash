// mul_mat_id test: Q4_K experts (with and without fused swiglu gate) and Q5_1 experts; NEXT_MOEA on vs off; CPU reference
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    struct C { int K, N, E, U, T; ggml_type t; bool gate; };
    std::vector<C> cases = {{2560, 640, 64, 10, 5, GGML_TYPE_Q4_K, false}, {2560, 640, 64, 10, 5, GGML_TYPE_Q4_K, true}, {640, 2560, 64, 10, 5, GGML_TYPE_Q5_1, false},
                            {2560, 640, 64, 10, 1, GGML_TYPE_Q4_K, true}, {640, 2560, 64, 10, 1, GGML_TYPE_Q5_1, false}, {2560, 640, 128, 10, 8, GGML_TYPE_Q4_K, true}, {640, 2560, 128, 10, 8, GGML_TYPE_Q5_1, false}};
    for (const C & c : cases) {
        std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
        const size_t nw = (size_t) c.K * c.N * c.E;
        std::vector<float> w(nw), wg(c.gate ? nw : 0), x((size_t) c.K * c.T);
        for (auto & v : w) v = nd(rng); for (auto & v : wg) v = nd(rng); for (auto & v : x) v = nd(rng);
        std::vector<int32_t> ids((size_t) c.U * c.T);
        for (int t = 0; t < c.T; ++t) { std::vector<int> perm(c.E); for (int i = 0; i < c.E; ++i) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng); for (int u = 0; u < c.U; ++u) ids[(size_t) t * c.U + u] = perm[u]; }
        if (c.T >= 2) ids[c.U] = ids[0]; // share one expert between token 0 and 1
        ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * W = ggml_new_tensor_3d(ctx, c.t, c.K, c.N, c.E);
        ggml_tensor * WG = c.gate ? ggml_new_tensor_3d(ctx, c.t, c.K, c.N, c.E) : nullptr;
        ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, 1, c.T);
        ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.U, c.T);
        ggml_tensor * Y = ggml_mul_mat_id(ctx, W, X, I);
        if (c.gate) { ggml_tensor * G = ggml_mul_mat_id(ctx, WG, X, I); Y = ggml_swiglu_split(ctx, G, Y); }
        ggml_cgraph * g = ggml_new_graph(ctx); ggml_build_forward_expand(g, Y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> wq(ggml_nbytes(W)); ggml_quantize_chunk(c.t, w.data(), wq.data(), 0, (int64_t) c.N * c.E, c.K, nullptr); ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
        std::vector<uint8_t> wgq; if (c.gate) { wgq.resize(ggml_nbytes(WG)); ggml_quantize_chunk(c.t, wg.data(), wgq.data(), 0, (int64_t) c.N * c.E, c.K, nullptr); ggml_backend_tensor_set(WG, wgq.data(), 0, wgq.size()); }
        ggml_backend_tensor_set(X, x.data(), 0, x.size() * 4); ggml_backend_tensor_set(I, ids.data(), 0, ids.size() * 4);
        ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be);
        std::vector<float> y((size_t) c.N * c.U * c.T); ggml_backend_tensor_get(Y, y.data(), 0, y.size() * 4);
        // CPU reference from dequantized experts
        std::vector<float> wd(nw), wgd(c.gate ? nw : 0);
        ggml_get_type_traits(c.t)->to_float(wq.data(), wd.data(), nw); if (c.gate) ggml_get_type_traits(c.t)->to_float(wgq.data(), wgd.data(), nw);
        double maxabs = 0, maxref = 0;
        for (int t = 0; t < c.T; ++t) for (int u = 0; u < c.U; ++u) {
            const int e = ids[(size_t) t * c.U + u];
            for (int n = 0; n < c.N; n += 7) {
                double r = 0, rg = 0;
                for (int k = 0; k < c.K; ++k) { const double xv = x[(size_t) t * c.K + k]; r += (double) wd[((size_t) e * c.N + n) * c.K + k] * xv; if (c.gate) rg += (double) wgd[((size_t) e * c.N + n) * c.K + k] * xv; }
                if (c.gate) r = r * (rg / (1.0 + exp(-rg)));
                const double got = y[((size_t) t * c.U + u) * c.N + n];
                maxabs = std::max(maxabs, fabs(r - got)); maxref = std::max(maxref, fabs(r));
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 30; ++i) ggml_backend_graph_compute(be, g);
        ggml_backend_synchronize(be);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 30;
        // distinct experts touched
        std::vector<int> seen(c.E, 0); int distinct = 0; for (int v : ids) if (!seen[v]++) distinct++;
        const double bytes = (double) distinct * ggml_row_size(c.t, c.K) * c.N * (c.gate ? 2 : 1);
        printf("%s K=%d N=%d E=%d U=%d T=%d gate=%d  %8.1f us  %6.0f GB/s (distinct experts %d)  maxabs %.3e (ref max %.1f)\n", ggml_type_name(c.t), c.K, c.N, c.E, c.U, c.T, (int) c.gate, us, bytes / us / 1e3, distinct, maxabs, maxref);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    return 0;
}
