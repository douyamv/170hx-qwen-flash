// q8a plan sweep helper: q8a_test2 K N B [type=q8_0|q4_0]  -> prints "K N B <min us over 5x100 graph evals>"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: q8a_test2 K N B [q8_0|q4_0]\n"); return 1; }
    const int K = atoi(argv[1]), N = atoi(argv[2]), B = atoi(argv[3]);
    const ggml_type t = (argc > 4 && strcmp(argv[4], "q4_0") == 0) ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) K * N), x((size_t) K * B);
    for (auto & v : w) v = nd(rng); for (auto & v : x) v = nd(rng);
    ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, t, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, B);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * g = ggml_new_graph(ctx); ggml_build_forward_expand(g, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> wq(ggml_nbytes(W)); ggml_quantize_chunk(t, w.data(), wq.data(), 0, N, K, nullptr);
    ggml_backend_tensor_set(W, wq.data(), 0, wq.size()); ggml_backend_tensor_set(X, x.data(), 0, x.size() * 4);
    ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be);
    double us = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) ggml_backend_graph_compute(be, g);
        ggml_backend_synchronize(be);
        us = std::min(us, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 100);
    }
    printf("%d %d %d %.2f\n", K, N, B, us);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return 0;
}
