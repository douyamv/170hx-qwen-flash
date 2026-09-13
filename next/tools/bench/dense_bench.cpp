// Dense projection micro benchmark: 16 layers x 10 Qwen3.8-Flash-Next dense matmuls, Q8_0 vs F16 weights.
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
struct shape { int in, out; };
static const std::vector<shape> SHAPES = {{2560,6144},{2560,10240},{6144,2560},{2560,640},{2560,640},{640,2560},{10240,320},{320,10240},{10240,320},{320,10240}};

static void fill(ggml_tensor * t, std::mt19937 & rng) {
    std::vector<uint8_t> buf(ggml_nbytes(t));
    if (t->type == GGML_TYPE_Q8_0) {
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t) rng();
        for (size_t i = 0; i < buf.size(); i += 34) { uint16_t h = 0x2400 | (rng() & 0xff); memcpy(&buf[i], &h, 2); }
    } else if (t->type == GGML_TYPE_F16) {
        std::normal_distribution<float> nd(0, 0.02f);
        for (size_t i = 0; i + 1 < buf.size(); i += 2) { uint16_t h = ggml_fp32_to_fp16(nd(rng)); memcpy(&buf[i], &h, 2); }
    } else {
        std::normal_distribution<float> nd(0, 1);
        for (size_t i = 0; i + 3 < buf.size(); i += 4) { float f = nd(rng); memcpy(&buf[i], &f, 4); }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

static void bench(ggml_backend_t be, ggml_type wtype, int B, int layers) {
    const size_t n_nodes = SHAPES.size() * layers;
    ggml_init_params p = { ggml_tensor_overhead() * (n_nodes * 4 + 64) + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    std::vector<ggml_tensor *> xs, ws;
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 8192, false);
    for (int l = 0; l < layers; ++l) {
        for (auto s : SHAPES) {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, s.in, s.out);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.in, B);
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_build_forward_expand(g, y);
            ws.push_back(w); xs.push_back(x);
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(3);
    for (auto * w : ws) fill(w, rng);
    for (auto * x : xs) fill(x, rng);
    for (int i = 0; i < 5; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
    const int reps = 200;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) { ggml_backend_graph_compute(be, g); ggml_backend_synchronize(be); }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    size_t wbytes = 0; for (auto * w : ws) wbytes += ggml_nbytes(w);
    printf("weights=%-5s B=%d  %3zu matmuls  %7.3f ms/graph  %6.1f us/matmul  (weights %.0f MiB)\n",
           ggml_type_name(wtype), B, n_nodes, ms, ms * 1000.0 / n_nodes, wbytes / 1048576.0);
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    for (int B : {1, 4}) {
        for (ggml_type t : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
            bench(be, t, B, 16);
        }
    }
    ggml_backend_free(be);
    return 0;
}
