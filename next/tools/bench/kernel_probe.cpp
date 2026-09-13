// Per-kernel cost probe: (1) tiny elementwise kernels, (2) Q8_0 matmul per shape, (3) matmul of 1 token vs weight rows.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

static double time_graph(ggml_backend_t be, ggml_cgraph * g, int reps = 300) {
    for (int i = 0; i < 5; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}
static void fill_any(ggml_tensor * t, std::mt19937 & rng) {
    std::vector<uint8_t> buf(ggml_nbytes(t));
    for (auto & b : buf) b = (uint8_t) rng();
    if (t->type == GGML_TYPE_Q8_0) for (size_t i = 0; i < buf.size(); i += 34) { uint16_t h = 0x2400 | (rng() & 0xff); memcpy(&buf[i], &h, 2); }
    if (t->type == GGML_TYPE_F32) for (size_t i = 0; i + 3 < buf.size(); i += 4) { float f = (float)((int)(rng() % 2001) - 1000) / 1000.0f; memcpy(&buf[i], &f, 4); }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}
static void run(ggml_backend_t be, const char * label, int n_ops, std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)> build) {
    ggml_init_params p = { ggml_tensor_overhead() * (n_ops * 4 + 16) + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);
    std::vector<ggml_tensor *> leaves;
    for (int i = 0; i < n_ops; ++i) ggml_build_forward_expand(g, build(ctx, leaves));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(11);
    for (auto * t : leaves) fill_any(t, rng);
    double ms = time_graph(be, g);
    printf("%-44s %4d ops  %8.3f ms/graph  %7.2f us/op\n", label, n_ops, ms, ms * 1000 / n_ops);
    fflush(stdout);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}
int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    for (int n : {4, 10240, 262144}) {
        char label[64]; snprintf(label, sizeof label, "add F32 (%d elems)", n);
        run(be, label, 200, [n](ggml_context * c, std::vector<ggml_tensor *> & L) {
            auto * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, n); auto * b = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
            L.push_back(a); L.push_back(b); return ggml_add(c, a, b); });
    }
    for (int n : {10240}) {
        run(be, "scale F32 (10240 elems)", 200, [n](ggml_context * c, std::vector<ggml_tensor *> & L) {
            auto * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, n); L.push_back(a); return ggml_scale(c, a, 0.5f); });
        run(be, "rms_norm F32 (10240 elems)", 200, [n](ggml_context * c, std::vector<ggml_tensor *> & L) {
            auto * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, n); L.push_back(a); return ggml_rms_norm(c, a, 1e-6f); });
    }
    struct S { int in, out; };
    for (S s : std::vector<S>{{2560,640},{640,2560},{2560,6144},{6144,2560},{2560,10240},{10240,320},{320,10240},{2560,12288}}) {
        for (int B : {1, 4}) {
            char label[64]; snprintf(label, sizeof label, "mul_mat Q8_0 %5d->%-5d B=%d (%4.1f MiB)", s.in, s.out, B, s.in * s.out * 34.0 / 32 / 1048576);
            run(be, label, 100, [s, B](ggml_context * c, std::vector<ggml_tensor *> & L) {
                auto * w = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, s.in, s.out); auto * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.in, B);
                L.push_back(w); L.push_back(x); return ggml_mul_mat(c, w, x); });
        }
    }
    ggml_backend_free(be);
    return 0;
}
