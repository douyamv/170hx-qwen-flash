// exact-equality test of ggml_top_k on CUDA (radix path) vs the argsort fallback (env NEXT_TOPK_SORT=1), incl. ties
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
static std::vector<int> run(ggml_backend_t be, const std::vector<float> & data, int ncols, int nrows, int k, double * ms) {
    ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * y = ggml_top_k(ctx, x, k);
    ggml_cgraph * g = ggml_new_graph(ctx); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(x, data.data(), 0, data.size() * 4);
    ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) ggml_backend_graph_compute(be, g);
    ggml_backend_synchronize(be);
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
    std::vector<int> out((size_t) k * nrows);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return out;
}
int main(int argc, char ** argv) {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    const bool sort_path = getenv("NEXT_TOPK_SORT") != nullptr;
    struct C { int ncols, nrows, k; int tie_bits; };
    std::vector<C> cases = {{70000, 5, 2051, 0}, {70000, 5, 2051, 12}, {262144, 5, 2051, 10}, {262144, 1, 2051, 0}, {16384, 5, 2051, 8}, {8192, 5, 2051, 0}, {70000, 5, 100, 6}, {70000, 5, 4096, 8}};
    for (const C & c : cases) {
        std::mt19937 rng(123); std::normal_distribution<float> nd(0.f, 3.f);
        std::vector<float> d((size_t) c.ncols * c.nrows);
        for (auto & v : d) { v = nd(rng); if (c.tie_bits) { v = ldexpf(roundf(ldexpf(v, c.tie_bits - 8)), 8 - c.tie_bits); } }
        // sprinkle -inf like a mask and some exact duplicates
        for (size_t i = 0; i < d.size(); i += 97) d[i] = -INFINITY;
        for (size_t i = 5; i + 1 < d.size(); i += 101) d[i + 1] = d[i];
        double ms; std::vector<int> out = run(be, d, c.ncols, c.nrows, c.k, &ms);
        unsigned long long h = 1469598103934665603ULL; for (int v : out) { h ^= (unsigned) v; h *= 1099511628211ULL; }
        // sanity: sorted desc by value, ties by index asc
        bool ok = true; double sum = 0;
        for (int r = 0; r < c.nrows && ok; ++r) for (int i = 1; i < c.k; ++i) {
            float a = d[(size_t) r * c.ncols + out[(size_t) r * c.k + i - 1]], b = d[(size_t) r * c.ncols + out[(size_t) r * c.k + i]];
            if (a < b || (a == b && out[(size_t) r * c.k + i - 1] > out[(size_t) r * c.k + i])) { ok = false; break; }
        }
        for (int r = 0; r < c.nrows; ++r) for (int i = 0; i < c.k; ++i) sum += d[(size_t) r * c.ncols + out[(size_t) r * c.k + i]];
        printf("%s ncols=%6d nrows=%d k=%4d ties=%2d  %8.3f ms/op  hash %016llx  sorted_ok=%d  sum=%.3f\n", sort_path ? "argsort" : "radix  ", c.ncols, c.nrows, c.k, c.tie_bits, ms, h, (int) ok, sum);
    }
    return 0;
}
