// mul_mat_id on REAL expert weights from the mini GGUF (Q4_K up/gate fused with swiglu, Q5_1 down with per-slot inputs)
// vs a CPU reference; run twice (NEXT_MOEA=1 / 0) and compare the printed errors.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
static std::vector<uint8_t> read_tensor(const char * path, gguf_context * g, ggml_context * meta, const char * name, ggml_tensor ** t_out) {
    const int64_t id = gguf_find_tensor(g, name);
    if (id < 0) { fprintf(stderr, "tensor %s not found\n", name); exit(1); }
    ggml_tensor * t = ggml_get_tensor(meta, name);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    std::vector<uint8_t> buf(ggml_nbytes(t));
    FILE * f = fopen(path, "rb"); fseek(f, (long) off, SEEK_SET); size_t n = fread(buf.data(), 1, buf.size(), f); fclose(f);
    if (n != buf.size()) { fprintf(stderr, "short read %s\n", name); exit(1); }
    *t_out = t;
    return buf;
}
int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/mini/Qwen3.8-Flash-Next-mini4-noPLE.gguf";
    const int il = argc > 2 ? atoi(argv[2]) : 1;
    const int T = argc > 3 ? atoi(argv[3]) : 5;
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    char nu[64], ng[64], nd[64];
    snprintf(nu, 64, "blk.%d.ffn_up_exps.weight", il); snprintf(ng, 64, "blk.%d.ffn_gate_exps.weight", il); snprintf(nd, 64, "blk.%d.ffn_down_exps.weight", il);
    ggml_tensor *tu, *tg, *td;
    auto bu = read_tensor(path, g, meta, nu, &tu); auto bg = read_tensor(path, g, meta, ng, &tg); auto bd = read_tensor(path, g, meta, nd, &td);
    printf("up %s [%lld,%lld,%lld] gate %s down %s [%lld,%lld,%lld]\n", ggml_type_name(tu->type), (long long) tu->ne[0], (long long) tu->ne[1], (long long) tu->ne[2],
        ggml_type_name(tg->type), ggml_type_name(td->type), (long long) td->ne[0], (long long) td->ne[1], (long long) td->ne[2]);
    const int K = (int) tu->ne[0], N = (int) tu->ne[1], E = (int) tu->ne[2], U = 10;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    std::mt19937 rng(7); std::normal_distribution<float> nd_(0.f, 1.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    // activations with outliers (real hidden states have a few large channels)
    std::vector<float> x((size_t) K * T); for (auto & v : x) { v = nd_(rng); if (ud(rng) < 0.01f) v *= 25.f; }
    std::vector<int32_t> ids((size_t) U * T);
    for (int t = 0; t < T; ++t) { std::vector<int> perm(E); for (int i = 0; i < E; ++i) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng); for (int u = 0; u < U; ++u) ids[(size_t) t * U + u] = perm[u]; }
    if (T >= 2) ids[U] = ids[0];
    // --- graph 1: up/gate fused ---
    ggml_init_params p = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * W  = ggml_new_tensor_3d(ctx, tu->type, K, N, E);
    ggml_tensor * WG = ggml_new_tensor_3d(ctx, tg->type, K, N, E);
    ggml_tensor * X  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * I  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, T);
    ggml_tensor * Y  = ggml_mul_mat_id(ctx, W, X, I);
    ggml_tensor * G  = ggml_mul_mat_id(ctx, WG, X, I);
    ggml_tensor * H  = ggml_swiglu_split(ctx, G, Y);
    // --- down with a random per-slot input of the same shape as H ---
    ggml_tensor * WD = ggml_new_tensor_3d(ctx, td->type, (int) td->ne[0], (int) td->ne[1], E);
    ggml_tensor * XD = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int) td->ne[0], U, T);
    ggml_tensor * D  = ggml_mul_mat_id(ctx, WD, XD, I);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, H); ggml_build_forward_expand(gf, D);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(W, bu.data(), 0, bu.size()); ggml_backend_tensor_set(WG, bg.data(), 0, bg.size()); ggml_backend_tensor_set(WD, bd.data(), 0, bd.size());
    ggml_backend_tensor_set(X, x.data(), 0, x.size() * 4); ggml_backend_tensor_set(I, ids.data(), 0, ids.size() * 4);
    const int KD = (int) td->ne[0], ND = (int) td->ne[1];
    std::vector<float> xd((size_t) KD * U * T); for (auto & v : xd) { v = nd_(rng); if (ud(rng) < 0.01f) v *= 25.f; }
    ggml_backend_tensor_set(XD, xd.data(), 0, xd.size() * 4);
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    std::vector<float> h((size_t) N * U * T), d((size_t) ND * U * T);
    ggml_backend_tensor_get(H, h.data(), 0, h.size() * 4); ggml_backend_tensor_get(D, d.data(), 0, d.size() * 4);
    // CPU reference on dequantized rows (every 3rd output row)
    const auto * tr_u = ggml_get_type_traits(tu->type); const auto * tr_d = ggml_get_type_traits(td->type);
    std::vector<float> rowu(K), rowg(K), rowd(KD);
    double maxabs_h = 0, maxref_h = 0, sumabs_h = 0, sumref_h = 0, maxabs_d = 0, maxref_d = 0, sumabs_d = 0, sumref_d = 0; long cnt_h = 0, cnt_d = 0;
    const size_t rs_u = ggml_row_size(tu->type, K), rs_d = ggml_row_size(td->type, KD);
    for (int t = 0; t < T; ++t) for (int u = 0; u < U; ++u) {
        const int e = ids[(size_t) t * U + u];
        for (int n = 0; n < N; n += 3) {
            tr_u->to_float(bu.data() + ((size_t) e * N + n) * rs_u, rowu.data(), K); tr_u->to_float(bg.data() + ((size_t) e * N + n) * rs_u, rowg.data(), K);
            double su = 0, sg = 0; for (int k = 0; k < K; ++k) { su += (double) rowu[k] * x[(size_t) t * K + k]; sg += (double) rowg[k] * x[(size_t) t * K + k]; }
            const double r = su * (sg / (1.0 + exp(-sg))); const double got = h[((size_t) t * U + u) * N + n];
            maxabs_h = std::max(maxabs_h, fabs(r - got)); maxref_h = std::max(maxref_h, fabs(r)); sumabs_h += fabs(r - got); sumref_h += fabs(r); cnt_h++;
        }
        for (int n = 0; n < ND; n += 7) {
            tr_d->to_float(bd.data() + ((size_t) e * ND + n) * rs_d, rowd.data(), KD);
            double sd = 0; for (int k = 0; k < KD; ++k) sd += (double) rowd[k] * xd[((size_t) t * U + u) * KD + k];
            const double got = d[((size_t) t * U + u) * ND + n];
            maxabs_d = std::max(maxabs_d, fabs(sd - got)); maxref_d = std::max(maxref_d, fabs(sd)); sumabs_d += fabs(sd - got); sumref_d += fabs(sd); cnt_d++;
        }
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 30; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 30;
    printf("layer %d T=%d  up/gate(swiglu): maxabs %.4g (ref max %.4g) mean rel %.3e | down: maxabs %.4g (ref max %.4g) mean rel %.3e | graph %.1f us\n",
        il, T, maxabs_h, maxref_h, sumabs_h / sumref_h, maxabs_d, maxref_d, sumabs_d / sumref_d, us);
    // dump outputs for a bitwise/rel comparison between runs
    const char * dump = getenv("MOEA_DUMP");
    if (dump) { FILE * f = fopen(dump, "wb"); fwrite(h.data(), 4, h.size(), f); fwrite(d.data(), 4, d.size(), f); fclose(f); }
    ggml_backend_buffer_free(buf); ggml_free(ctx); gguf_free(g); ggml_free(meta);
    return 0;
}
