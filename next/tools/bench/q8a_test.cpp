// ggml-level test: Q8_0 [K,N] x F32 [K,B] on CUDA, NEXT_Q8A on (default) vs off (env), plus timing
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
int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    struct C { int K, N, B; ggml_type t = GGML_TYPE_Q8_0; };
    std::vector<C> cases = {{2560, 6144, 1}, {2560, 6144, 5}, {2560, 10240, 5}, {2560, 12288, 4}, {6144, 2560, 5}, {2560, 640, 5}, {10240, 320, 5}, {320, 10240, 5}, {2560, 2560, 8}, {2560, 248320, 5}, {2560, 6144, 1, GGML_TYPE_Q4_0}, {2560, 248320, 1, GGML_TYPE_Q4_0}, {2560, 248320, 4, GGML_TYPE_Q4_0}};
    for (const C & c : cases) {
        std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> w((size_t) c.K * c.N), x((size_t) c.K * c.B);
        for (auto & v : w) v = nd(rng); for (auto & v : x) v = nd(rng);
        ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, c.t, c.K, c.N);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.B);
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        ggml_cgraph * g = ggml_new_graph(ctx); ggml_build_forward_expand(g, Y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> wq(ggml_nbytes(W)); ggml_quantize_chunk(c.t, w.data(), wq.data(), 0, c.N, c.K, nullptr);
        ggml_backend_tensor_set(W, wq.data(), 0, wq.size()); ggml_backend_tensor_set(X, x.data(), 0, x.size() * 4);
        ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be);
        std::vector<float> y((size_t) c.N * c.B); ggml_backend_tensor_get(Y, y.data(), 0, y.size() * 4);
        // reference on CPU from the dequantized weights
        std::vector<float> wd((size_t) c.K * c.N); ggml_get_type_traits(c.t)->to_float(wq.data(), wd.data(), wd.size());
        double maxabs = 0, maxref = 0;
        for (int b = 0; b < c.B; ++b) for (int n = 0; n < c.N; n += (c.N > 4096 ? 97 : 1)) {
            double r = 0; for (int k = 0; k < c.K; ++k) r += (double) wd[(size_t) n * c.K + k] * x[(size_t) b * c.K + k];
            maxabs = std::max(maxabs, fabs(r - y[(size_t) b * c.N + n])); maxref = std::max(maxref, fabs(r));
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) ggml_backend_graph_compute(be, g);
        ggml_backend_synchronize(be);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
        const double bytes = (double) c.K * c.N * (c.t == GGML_TYPE_Q4_0 ? 18.0 / 32 : 34.0 / 32);
        printf("%s K=%5d N=%6d B=%d  ", ggml_type_name(c.t)); printf("%8.1f us  %6.0f GB/s  maxabs %.3e (ref max %.1f)\n", c.K, c.N, c.B, us, bytes / us / 1e3, maxabs, maxref);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    return 0;
}
