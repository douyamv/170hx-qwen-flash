// Standalone micro-benchmark: bandwidth-oriented Q8_0 GEMV for decode (B = 1..8 columns) with a 16-byte-aligned
// split layout (scales fp16 contiguous, quants int8 contiguous) vs the same math on the standard Q8_0 layout.
// Weight matrix: N rows x K cols. Activations: B columns of K floats, quantized to int8 blocks of 32 (Q8_1 style: fp32 scale).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// ---- reference (matches ggml's Q8_0 x Q8_1 dot: per-block int32 dot * d_w * d_a, fp32 accumulate over blocks)
static void ref_gemv(const std::vector<int8_t> & wq, const std::vector<float> & wd, const std::vector<int8_t> & aq, const std::vector<float> & ad, int N, int K, int B, std::vector<float> & out) {
    const int nb = K / 32;
    for (int b = 0; b < B; ++b) for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int i = 0; i < nb; ++i) {
            int s = 0; for (int j = 0; j < 32; ++j) s += (int) wq[(size_t) n * K + i * 32 + j] * (int) aq[(size_t) b * K + i * 32 + j];
            acc += (float) s * wd[(size_t) n * nb + i] * ad[(size_t) b * nb + i];
        }
        out[(size_t) b * N + n] = acc;
    }
}

// ---- kernel: split layout. wq: [N][K] int8 (16B aligned rows since K % 32 == 0), wd: [N][K/32] half. aq: [B][K] int8, ad: [B][K/32] float.
// one warp per (row-group of RPW rows); lanes stride over 16-byte chunks of the row; SPLITK partial sums over K-ranges.
template <int B, int RPW>
__global__ void __launch_bounds__(256) q8_gemv_split(const int8_t * __restrict__ wq, const half * __restrict__ wd,
        const int8_t * __restrict__ aq, const float * __restrict__ ad, float * __restrict__ out, int N, int K, int splitk) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n0 = (warp / splitk) * RPW;          // first row of this warp
    const int ks = warp % splitk;                  // K split index
    if (n0 >= N) return;
    const int nb = K >> 5;                          // blocks of 32
    const int nchunk = K >> 4;                      // 16-byte chunks per row
    const int cpk = nchunk / splitk;                // chunks per split
    const int c0 = ks * cpk;
    float acc[RPW][B];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    // each lane handles chunks c0+lane, c0+lane+32, ... ; two chunks per block of 32 -> block index = chunk>>1
    for (int c = c0 + lane; c < c0 + cpk; c += 32) {
        const int blk = c >> 1;
        int4 a[B];
#pragma unroll
        for (int b = 0; b < B; ++b) a[b] = *reinterpret_cast<const int4 *>(aq + (size_t) b * K + (size_t) c * 16);
        float dab[B];
#pragma unroll
        for (int b = 0; b < B; ++b) dab[b] = ad[(size_t) b * nb + blk];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int n = n0 + r;
            const int4 w = *reinterpret_cast<const int4 *>(wq + (size_t) n * K + (size_t) c * 16);
            const float d = __half2float(wd[(size_t) n * nb + blk]);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                int s = 0;
                s = __dp4a(w.x, a[b].x, s); s = __dp4a(w.y, a[b].y, s); s = __dp4a(w.z, a[b].z, s); s = __dp4a(w.w, a[b].w, s);
                acc[r][b] += (float) s * d * dab[b];
            }
        }
    }
    // warp reduce
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = acc[r][b];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[r][b] = v;
        }
    if (lane < RPW * B) {
        const int r = lane / B, b = lane % B;
        const float v = acc[r][b];   // lane holds all (warp-uniform after xor reduce)
        // NB: after xor-reduce every lane holds the full sum, so pick by lane index
        float vv = 0.f;
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for (int bb = 0; bb < B; ++bb) if (rr == r && bb == b) vv = acc[rr][bb];
        (void) v;
        if (splitk == 1) out[(size_t) b * N + n0 + r] = vv;
        else atomicAdd(out + (size_t) b * N + n0 + r, vv);
    }
}

// quantize activations: B x K floats -> int8 + per-32 scale (Q8_1-like, symmetric absmax)
__global__ void quant_act(const float * __restrict__ x, int8_t * __restrict__ q, float * __restrict__ d, int K) {
    const int b = blockIdx.y; const int blk = blockIdx.x * blockDim.x + threadIdx.x; // one thread per block of 32
    if (blk * 32 >= K) return;
    const float * xr = x + (size_t) b * K + blk * 32;
    float amax = 0.f; for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xr[j]));
    const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
    d[(size_t) b * (K / 32) + blk] = dd;
    int8_t * qr = q + (size_t) b * K + blk * 32;
    for (int j = 0; j < 32; ++j) qr[j] = (int8_t) lrintf(xr[j] * id);
}

struct Shape { int K, N; const char * name; };
int main(int argc, char ** argv) {
    const int B_list[] = {1, 4, 5, 8};
    Shape shapes[] = {{2560, 6144, "attn_gate 2560->6144"}, {2560, 10240, "qkv 2560->10240"}, {2560, 12288, "attn_q 2560->12288"}, {6144, 2560, "ssm_out 6144->2560"}, {2560, 640, "router/shexp 2560->640"}, {10240, 320, "hc 10240->320"}, {320, 10240, "hc 320->10240"}, {2560, 2560, "2560->2560"}};
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (const Shape & sh : shapes) {
        const int K = sh.K, N = sh.N, nb = K / 32;
        std::vector<int8_t> wq((size_t) N * K); std::vector<float> wd((size_t) N * nb); std::vector<half> wdh((size_t) N * nb);
        srand(1); for (auto & v : wq) v = (int8_t) (rand() % 255 - 127); for (size_t i = 0; i < wd.size(); ++i) { wdh[i] = __float2half(0.001f * (1 + rand() % 100)); wd[i] = __half2float(wdh[i]); }
        int8_t * d_wq; half * d_wd; CK(cudaMalloc(&d_wq, wq.size())); CK(cudaMalloc(&d_wd, wdh.size() * 2));
        CK(cudaMemcpy(d_wq, wq.data(), wq.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_wd, wdh.data(), wdh.size() * 2, cudaMemcpyHostToDevice));
        const double bytes = (double) N * K + (double) N * nb * 2;
        for (int B : B_list) {
            std::vector<float> x((size_t) B * K); for (auto & v : x) v = (rand() % 2001 - 1000) / 250.f;
            float * d_x; int8_t * d_aq; float * d_ad; float * d_out;
            CK(cudaMalloc(&d_x, x.size() * 4)); CK(cudaMalloc(&d_aq, (size_t) B * K)); CK(cudaMalloc(&d_ad, (size_t) B * nb * 4)); CK(cudaMalloc(&d_out, (size_t) B * N * 4));
            CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            quant_act<<<dim3((nb + 127) / 128, B), 128>>>(d_x, d_aq, d_ad, K); CK(cudaDeviceSynchronize());
            std::vector<int8_t> aq((size_t) B * K); std::vector<float> ad((size_t) B * nb);
            CK(cudaMemcpy(aq.data(), d_aq, aq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ad.data(), d_ad, ad.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<float> ref((size_t) B * N); ref_gemv(wq, wd, aq, ad, N, K, B, ref);
            // choose splitk so that warps >= ~2048: rows per warp RPW=2
            const int RPW = 2; const int warps_needed = 2048; int splitk = 1; while ((N / RPW) * splitk < warps_needed && splitk < 16 && ((K / 16) / (splitk * 2)) >= 32) splitk *= 2;
            const int nwarps = (N / RPW) * splitk; const int threads = 256; const int blocks = (nwarps * 32 + threads - 1) / threads;
            auto launch = [&]() {
                if (splitk > 1) CK(cudaMemsetAsync(d_out, 0, (size_t) B * N * 4));
                switch (B) { case 1: q8_gemv_split<1, 2><<<blocks, threads>>>(d_wq, d_wd, d_aq, d_ad, d_out, N, K, splitk); break;
                             case 4: q8_gemv_split<4, 2><<<blocks, threads>>>(d_wq, d_wd, d_aq, d_ad, d_out, N, K, splitk); break;
                             case 5: q8_gemv_split<5, 2><<<blocks, threads>>>(d_wq, d_wd, d_aq, d_ad, d_out, N, K, splitk); break;
                             case 8: q8_gemv_split<8, 2><<<blocks, threads>>>(d_wq, d_wd, d_aq, d_ad, d_out, N, K, splitk); break; }
            };
            launch(); CK(cudaDeviceSynchronize());
            std::vector<float> out((size_t) B * N); CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
            double maxrel = 0; for (size_t i = 0; i < out.size(); ++i) { double r = fabs(out[i] - ref[i]) / (fabs(ref[i]) + 1e-3); maxrel = std::max(maxrel, r); }
            const int iters = 200; CK(cudaEventRecord(e0)); for (int i = 0; i < iters; ++i) launch(); CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); const double us = ms * 1000.0 / iters;
            printf("%-26s B=%d splitk=%2d warps=%5d  %7.1f us  %7.0f GB/s  maxrel %.2e\n", sh.name, B, splitk, nwarps, us, bytes / us / 1e3, maxrel);
            CK(cudaFree(d_x)); CK(cudaFree(d_aq)); CK(cudaFree(d_ad)); CK(cudaFree(d_out));
        }
        CK(cudaFree(d_wq)); CK(cudaFree(d_wd));
    }
    return 0;
}
