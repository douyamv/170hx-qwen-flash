#include "q8a.cuh"
#include <mutex>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// ---------------------------------------------------------------------------------------------------------------------
// repacked weight cache: attached to the weight tensor through tensor->extra (a fresh tensor struct always starts with
// extra == nullptr, so a reused device address cannot hand out a stale layout); freed with the process.
struct q8a_weight {
    uint32_t magic = 0x51384131u; // 'Q8A1'
    int64_t  N = 0, K = 0;
    ggml_type type = GGML_TYPE_Q8_0;
    int8_t * qs = nullptr;   // Q8_0: [N][K] int8;  Q4_0: [N][K/2] packed nibbles (ggml order: low = j, high = j+16)
    half   * d  = nullptr;   // [N][K/32]
    bool     failed = false; // allocation failed once: keep using mmvq for this tensor
};
static std::mutex q8a_mutex;

static bool q8a_enabled() {
    static const bool enabled = [] { const char * e = getenv("NEXT_Q8A"); return e == nullptr || atoi(e) != 0; }();
    return enabled;
}

// block_q8_0 {half d; int8 qs[32]} (34 bytes) -> split layout. one thread per block.
static __global__ void q8a_repack(const uint8_t * __restrict__ src, int8_t * __restrict__ qs, half * __restrict__ d, const int64_t nblocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t * b = src + i * 34;
    half dh; memcpy(&dh, b, 2);
    d[i] = dh;
    // quants are 2-byte aligned in the source: copy as 16 ushorts
    const uint16_t * q16 = (const uint16_t *) (b + 2);
    uint16_t * o16 = (uint16_t *) (qs + i * 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) o16[j] = q16[j];
}

// block_q4_0 {half d; uint8 qs[16]} (18 bytes) -> split layout
static __global__ void q4a_repack(const uint8_t * __restrict__ src, int8_t * __restrict__ qs, half * __restrict__ d, const int64_t nblocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t * b = src + i * 18;
    half dh; memcpy(&dh, b, 2);
    d[i] = dh;
    const uint16_t * q16 = (const uint16_t *) (b + 2);
    uint16_t * o16 = (uint16_t *) (qs + i * 16);
#pragma unroll
    for (int j = 0; j < 8; ++j) o16[j] = q16[j];
}

// activations F32 [K, B] (row stride s11 elements) -> int8 [B][K] + fp32 absmax/127 scales [B][K/32]. one warp per block of 32.
static __global__ void q8a_quantize(const float * __restrict__ x, const int64_t s11, int8_t * __restrict__ aq, float * __restrict__ ad, float * __restrict__ asum, const int K) {
    const int lane = threadIdx.x & 31;
    const int blk  = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int nb   = K >> 5;
    if (blk >= nb) return;
    const int b = blockIdx.y;
    const float v = x[(int64_t) b * s11 + blk * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
    const int q = (int) lrintf(v * id);
    aq[(int64_t) b * K + blk * 32 + lane] = (int8_t) q;
    int qs = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qs += __shfl_xor_sync(0xffffffff, qs, o);
    if (lane == 0) {
        ad[(int64_t) b * nb + blk] = dd;
        if (asum) asum[(int64_t) b * nb + blk] = (float) qs;   // sum of the int8 values, for the Q4_0 -8 offset
    }
}

// Q4_0 GEMV: one lane per 32-weight block (16 bytes of nibbles), two rows per warp
template <int B>
__global__ void __launch_bounds__(256) q4a_gemv(const int8_t * __restrict__ wq, const half * __restrict__ wd,
        const int8_t * __restrict__ aq, const float * __restrict__ ad, const float * __restrict__ asum,
        float * __restrict__ out, const int N, const int K) {
    constexpr int RPW = 2;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n0   = warp * RPW;
    if (n0 >= N) return;
    const int nb = K >> 5;
    const bool has_r1 = n0 + 1 < N;
    float acc[RPW][B];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    for (int blk = lane; blk < nb; blk += 32) {
        int4 alo[B], ahi[B]; float dab[B], sab[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            alo[b] = *reinterpret_cast<const int4 *>(aq + (int64_t) b * K + (int64_t) blk * 32);
            ahi[b] = *reinterpret_cast<const int4 *>(aq + (int64_t) b * K + (int64_t) blk * 32 + 16);
            dab[b] = ad[(int64_t) b * nb + blk];
            sab[b] = asum[(int64_t) b * nb + blk];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r == 1 && !has_r1) break;
            const int n = n0 + r;
            const int4 q = *reinterpret_cast<const int4 *>(wq + (int64_t) n * (K / 2) + (int64_t) blk * 16);
            const int lx = q.x & 0x0F0F0F0F, ly = q.y & 0x0F0F0F0F, lz = q.z & 0x0F0F0F0F, lw = q.w & 0x0F0F0F0F;
            const int hx = (q.x >> 4) & 0x0F0F0F0F, hy = (q.y >> 4) & 0x0F0F0F0F, hz = (q.z >> 4) & 0x0F0F0F0F, hw = (q.w >> 4) & 0x0F0F0F0F;
            const float d = __half2float(wd[(int64_t) n * nb + blk]);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                int s = 0;
                s = __dp4a(lx, alo[b].x, s); s = __dp4a(ly, alo[b].y, s); s = __dp4a(lz, alo[b].z, s); s = __dp4a(lw, alo[b].w, s);
                s = __dp4a(hx, ahi[b].x, s); s = __dp4a(hy, ahi[b].y, s); s = __dp4a(hz, ahi[b].z, s); s = __dp4a(hw, ahi[b].w, s);
                acc[r][b] += ((float) s - 8.0f * sab[b]) * d * dab[b];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = acc[r][b];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[r][b] = v;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r == 1 && !has_r1) break;
#pragma unroll
            for (int b = 0; b < B; ++b) out[(int64_t) b * N + n0 + r] = acc[r][b];
        }
    }
}

// one warp per R rows x one K-split; lanes stride over 16-byte chunks (2 chunks per 32-quant block). the R weight loads
// of an iteration are issued together so a warp keeps R*512 bytes in flight.
// splitk == 1: write dst (+bias). splitk > 1: write partials [splitk][B][N], reduced deterministically afterwards.
template <int B, int R>
__global__ void __launch_bounds__(256) q8a_gemv(const int8_t * __restrict__ wq, const half * __restrict__ wd,
        const int8_t * __restrict__ aq, const float * __restrict__ ad, const float * __restrict__ bias,
        float * __restrict__ out, const int N, const int K, const int splitk) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n0   = (warp / splitk) * R;
    const int ks   = warp % splitk;
    if (n0 >= N) return;
    const int nb     = K >> 5;
    const int nchunk = K >> 4;
    const int cpk    = nchunk / splitk;
    const int c0     = ks * cpk;
    float acc[R][B];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    for (int c = c0 + lane; c < c0 + cpk; c += 32) {
        const int blk = c >> 1;
        int4 w[R]; float d[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (n0 + r < N) {
                w[r] = *reinterpret_cast<const int4 *>(wq + (int64_t) (n0 + r) * K + (int64_t) c * 16);
                d[r] = __half2float(wd[(int64_t) (n0 + r) * nb + blk]);
            } else {
                w[r] = make_int4(0, 0, 0, 0); d[r] = 0.f;
            }
        }
        int4 a[B]; float dab[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            a[b]   = *reinterpret_cast<const int4 *>(aq + (int64_t) b * K + (int64_t) c * 16);
            dab[b] = ad[(int64_t) b * nb + blk];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int b = 0; b < B; ++b) {
                int s = 0;
                s = __dp4a(w[r].x, a[b].x, s); s = __dp4a(w[r].y, a[b].y, s); s = __dp4a(w[r].z, a[b].z, s); s = __dp4a(w[r].w, a[b].w, s);
                acc[r][b] += (float) s * d[r] * dab[b];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = acc[r][b];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[r][b] = v;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (n0 + r >= N) break;
#pragma unroll
            for (int b = 0; b < B; ++b) {
                if (splitk == 1) {
                    out[(int64_t) b * N + n0 + r] = acc[r][b] + (bias ? bias[n0 + r] : 0.f);
                } else {
                    out[((int64_t) ks * B + b) * N + n0 + r] = acc[r][b];
                }
            }
        }
    }
}

typedef void (*q8a_kernel_t)(const int8_t *, const half *, const int8_t *, const float *, const float *, float *, int, int, int);

template <int B> static q8a_kernel_t q8a_kernel_r(int R) {
    switch (R) {
        case 1: return q8a_gemv<B, 1>;
        case 2: return q8a_gemv<B, 2>;
        case 3: return q8a_gemv<B, 3>;
        default: return q8a_gemv<B, 4>;
    }
}

static q8a_kernel_t q8a_kernel(int B, int R) {
    switch (B) {
        case 1: return q8a_kernel_r<1>(R);
        case 2: return q8a_kernel_r<2>(R);
        case 3: return q8a_kernel_r<3>(R);
        case 4: return q8a_kernel_r<4>(R);
        case 5: return q8a_kernel_r<5>(R);
        case 6: return q8a_kernel_r<6>(R);
        case 7: return q8a_kernel_r<7>(R);
        default: return q8a_kernel_r<8>(R);
    }
}

struct q8a_plan { int R; int splitk; };
struct q8a_plan_entry { int N, K, B, R, splitk; };

// measured on CMP 170HX (70 SMs, 1.39 TB/s) with next/tools/bench/q8a_test2 + sweep2.sh: only shapes where the plan
// beats the rule below by more than 3%. {N, K, B, rows per warp, K-splits}
static const q8a_plan_entry q8a_measured[] = {
//Q8A_TABLE_BEGIN
    {10240, 320, 3, 3, 2},
    {10240, 320, 4, 1, 4},
    {320, 2560, 1, 1, 1},
    {320, 2560, 3, 1, 1},
    {320, 2560, 4, 1, 1},
    {320, 2560, 5, 1, 1},
    {640, 2560, 1, 4, 4},
    {640, 2560, 3, 1, 1},
    {640, 2560, 4, 1, 1},
    {640, 2560, 5, 1, 1},
    {2560, 2560, 3, 3, 1},
    {2560, 2560, 4, 3, 4},
    {2560, 2560, 5, 4, 4},
    {6144, 2560, 4, 3, 4},
    {6144, 2560, 5, 3, 1},
    {10240, 2560, 1, 1, 4},
    {10240, 2560, 4, 2, 4},
    {10240, 2560, 5, 4, 1},
    {12288, 2560, 3, 3, 1},
    {12288, 2560, 4, 3, 1},
    {12288, 2560, 5, 3, 1},
    {2560, 6144, 1, 1, 1},
//Q8A_TABLE_END
};

// the opt-v5 rule: 2 rows per warp; for K >= 4096 split K (whole 32-quant blocks) until ~2048 warps are in flight
static q8a_plan q8a_plan_rule(int N, int K) {
    const int nchunk = K / 16, warps_n = (N + 1) / 2;
    int s = 1;
    if (K >= 4096) {
        while (warps_n * s < 2048 && s < 8 && nchunk % (s * 2) == 0 && nchunk / (s * 2) >= 64) s *= 2;
    }
    return {2, s};
}

static bool q8a_plan_valid(int K, int R, int s) {
    const int nchunk = K / 16;
    return R >= 1 && R <= 4 && s >= 1 && s <= 8 && nchunk % (2 * s) == 0 && nchunk / s >= 64;
}

// plan lookup order: NEXT_Q8A_RPW / NEXT_Q8A_SPLITK (experiments) > $NEXT_OPT_DIR/q8a_plans ("N K B R splitk" per line,
// read once) > the measured table > the rule
static q8a_plan q8a_choose(int N, int K, int B) {
    static const int force_r = [] { const char * e = getenv("NEXT_Q8A_RPW");    return e ? atoi(e) : 0; }();
    static const int force_s = [] { const char * e = getenv("NEXT_Q8A_SPLITK"); return e ? atoi(e) : 0; }();
    if (force_r || force_s) {
        q8a_plan p = q8a_plan_rule(N, K);
        if (force_r) p.R = force_r;
        if (force_s) p.splitk = force_s;
        if (q8a_plan_valid(K, p.R, p.splitk)) return p;
    }
    static const std::vector<q8a_plan_entry> file_plans = [] {
        std::vector<q8a_plan_entry> v;
        const char * dir = getenv("NEXT_OPT_DIR");
        if (dir == nullptr) return v;
        FILE * f = fopen((std::string(dir) + "/q8a_plans").c_str(), "r");
        if (f == nullptr) return v;
        q8a_plan_entry e;
        while (fscanf(f, "%d %d %d %d %d", &e.N, &e.K, &e.B, &e.R, &e.splitk) == 5) v.push_back(e);
        fclose(f);
        GGML_LOG_INFO("q8a: %zu plans read from %s/q8a_plans\n", v.size(), dir);
        return v;
    }();
    for (const auto & e : file_plans) {
        if (e.N == N && e.K == K && e.B == B && q8a_plan_valid(K, e.R, e.splitk)) return {e.R, e.splitk};
    }
    for (const auto & e : q8a_measured) {
        if (e.N == N && e.K == K && e.B == B && q8a_plan_valid(K, e.R, e.splitk)) return {e.R, e.splitk};
    }
    return q8a_plan_rule(N, K);
}

static __global__ void q8a_reduce(const float * __restrict__ part, const float * __restrict__ bias, float * __restrict__ out, const int N, const int B, const int splitk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // over B*N
    if (i >= B * N) return;
    float s = 0.f;
    for (int k = 0; k < splitk; ++k) s += part[(int64_t) k * B * N + i];
    out[i] = s + (bias ? bias[i % N] : 0.f);
}

// ---------------------------------------------------------------------------------------------------------------------
bool ggml_cuda_q8a_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                             const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!q8a_enabled() || ids != nullptr) return false;
    if ((src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->type == GGML_TYPE_Q4_0 && fusion && fusion->x_bias) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->ne[1] < 1 || src1->ne[1] > 8) return false;
    if (src0->ne[0] % 64 != 0 || src0->ne[0] < 256) return false;               // K: whole 16-byte chunk pairs per split
    if (!ggml_is_contiguous(src0) || src0->buffer == nullptr) return false;
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
    if (src0->extra != nullptr) {
        const q8a_weight * w = (const q8a_weight *) src0->extra;
        if (w->magic != 0x51384131u || w->failed || w->N != src0->ne[1] || w->K != src0->ne[0]) return false;
    } else {
        // first use: the repacked copy costs as much VRAM as the tensor itself, so very large tensors (the Q8_0
        // output head) stay on mmvq unless NEXT_Q8A_MAX_MB raises the cap; and never repack inside a graph capture
        static const int64_t max_mb = [] { const char * e = getenv("NEXT_Q8A_MAX_MB"); return e ? atoll(e) : 400; }();
        if ((int64_t) ggml_nbytes(src0) > max_mb * 1024 * 1024) return false;
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ctx.stream(), &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) return false;
        (void) cudaGetLastError();
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->nb[1] != (size_t) dst->ne[0] * sizeof(float)) return false;
    if (fusion) {
        // only a fused bias is supported; gates / scales stay on the mmvq path
        if (fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale) return false;
        if (fusion->x_bias && (fusion->x_bias->type != GGML_TYPE_F32 || fusion->x_bias->ne[0] != dst->ne[0] || !ggml_is_contiguous(fusion->x_bias))) return false;
    }
    return true;
}

static q8a_weight * q8a_get_weight(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(q8a_mutex);
    ggml_tensor * t = const_cast<ggml_tensor *>(src0);
    if (t->extra != nullptr) {
        q8a_weight * w = (q8a_weight *) t->extra;
        if (w->magic != 0x51384131u) return nullptr;              // extra belongs to someone else (e.g. split buffers)
        if (w->failed || w->N != src0->ne[1] || w->K != src0->ne[0]) return nullptr;
        return w;
    }
    const int64_t N = src0->ne[1], K = src0->ne[0], nblocks = N * (K / 32);
    q8a_weight * w = new q8a_weight();
    w->N = N; w->K = K; w->type = src0->type;
    const size_t qs_bytes = (size_t) nblocks * (src0->type == GGML_TYPE_Q4_0 ? 16 : 32);
    ggml_cuda_set_device(ctx.device);
    if (cudaMalloc(&w->qs, qs_bytes) != cudaSuccess || cudaMalloc(&w->d, (size_t) nblocks * sizeof(half)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (w->qs) { cudaFree(w->qs); w->qs = nullptr; }
        w->failed = true;
        GGML_LOG_WARN("%s: could not allocate the aligned layout for %s (%lld MiB), keeping mmvq\n", __func__, src0->name, (long long) (nblocks * 34 >> 20));
        t->extra = w;
        return nullptr;
    }
    if (src0->type == GGML_TYPE_Q4_0) {
        q4a_repack<<<(unsigned) ((nblocks + 255) / 256), 256, 0, stream>>>((const uint8_t *) src0->data, w->qs, w->d, nblocks);
    } else {
        q8a_repack<<<(unsigned) ((nblocks + 255) / 256), 256, 0, stream>>>((const uint8_t *) src0->data, w->qs, w->d, nblocks);
    }
    CUDA_CHECK(cudaGetLastError());
    // the repack must complete before any later stream (or a captured graph) reads the layout
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t->extra = w;
    return w;
}

bool ggml_cuda_mul_mat_q8a(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                           const ggml_cuda_mm_fusion_args_host * fusion) {
    const int N = (int) src0->ne[1], K = (int) src0->ne[0], B = (int) src1->ne[1];
    cudaStream_t stream = ctx.stream();
    const q8a_weight * wp = q8a_get_weight(ctx, src0, stream);
    if (wp == nullptr) {
        return false; // allocation failed or the tensor's extra is foreign: the caller falls back to mmvq
    }
    const q8a_weight & w = *wp;

    const int nb = K / 32;
    const bool is_q4 = w.type == GGML_TYPE_Q4_0;
    ggml_cuda_pool_alloc<int8_t> aq(ctx.pool(), (size_t) B * K);
    ggml_cuda_pool_alloc<float>  ad(ctx.pool(), (size_t) B * nb);
    ggml_cuda_pool_alloc<float>  as(ctx.pool());
    if (is_q4) as.alloc((size_t) B * nb);
    {
        const dim3 grid((nb + 7) / 8, B);
        q8a_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, src1->nb[1] / sizeof(float), aq.get(), ad.get(), is_q4 ? as.get() : nullptr, K);
    }
    if (is_q4) {
        const int blocks = ((N + 1) / 2 + 7) / 8;
        float * out = (float *) dst->data;
        switch (B) {
            case 1: q4a_gemv<1><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 2: q4a_gemv<2><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 3: q4a_gemv<3><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 4: q4a_gemv<4><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 5: q4a_gemv<5><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 6: q4a_gemv<6><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 7: q4a_gemv<7><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            case 8: q4a_gemv<8><<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), as.get(), out, N, K); break;
            default: GGML_ABORT("q4a: unsupported batch");
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const int nchunk = K / 16;
    const q8a_plan plan = q8a_choose(N, K, B);
    const int splitk = plan.splitk;
    static const bool verbose = getenv("NEXT_Q8A_VERBOSE") != nullptr;
    if (verbose) {
        static std::mutex m; static std::unordered_map<int64_t, int> seen;
        std::lock_guard<std::mutex> lock(m);
        const int64_t key = ((int64_t) N << 40) | ((int64_t) K << 8) | B;
        if (!seen[key]++) GGML_LOG_INFO("q8a: N=%d K=%d B=%d -> R=%d splitk=%d (%lld warps)\n", N, K, B, plan.R, splitk,
            (long long) (((N + plan.R - 1) / plan.R) * splitk));
    }
    GGML_ASSERT(nchunk % splitk == 0);
    const float * bias = (fusion && fusion->x_bias) ? (const float *) fusion->x_bias->data : nullptr;
    float * out = (float *) dst->data;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (splitk > 1) {
        part.alloc((size_t) splitk * B * N);
        out = part.get();
    }
    const int64_t nwarps = ((N + plan.R - 1) / plan.R) * (int64_t) splitk;
    const int blocks = (int) ((nwarps + 7) / 8);
    q8a_kernel(B, plan.R)<<<blocks, 256, 0, stream>>>(w.qs, w.d, aq.get(), ad.get(), bias, out, N, K, splitk);
    if (splitk > 1) {
        q8a_reduce<<<(B * N + 255) / 256, 256, 0, stream>>>(part.get(), bias, (float *) dst->data, N, B, splitk);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
