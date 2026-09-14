#include "moea.cuh"
#include "mmid.cuh"
#include <string>
#include <sys/stat.h>

// NEXT: expert-grouped decode GEMV for mul_mat_id (v3).
//  * activations are quantized once per token to int8 with a per-32 (scale, sum) pair;
//  * only experts that received tokens get thread blocks (compact expert list);
//  * a warp owns 4 (Q4_K) / 8 (Q5_1) consecutive rows of one expert: each 8-lane (4-lane) group streams its row with
//    16-byte (8-byte) loads, so the whole warp keeps 32 independent loads in flight per iteration;
//  * tokens of an expert are processed 2 per pass so the register footprint stays small (<= 85 regs, 3 blocks/SM).
#define MOEA_MAX_TOK 8

// NEXT_MOEA=0 disables the path at startup; $NEXT_OPT_DIR/moea_off (checked about once per second) disables it at
// runtime for A/B tests without a reload (the CUDA graph is re-captured when the kernel sequence changes)
static bool moea_enabled() {
    static const bool enabled = [] { const char * e = getenv("NEXT_MOEA"); return e == nullptr || atoi(e) != 0; }();
    if (!enabled) return false;
    static const char * dir = getenv("NEXT_OPT_DIR");
    if (dir == nullptr) return true;
    static int64_t last_check = 0;
    static bool off = false;
    const int64_t now = ggml_time_ms();
    if (now - last_check > 1000) {
        last_check = now;
        const std::string path = std::string(dir) + "/moea_off";
        struct stat st;
        off = stat(path.c_str(), &st) == 0;
    }
    return !off;
}

// activations F32 [K, ne11, ntok] -> int8 [ncol][K] + float2 {scale, sum} per 32-block [ncol][K/32], one column per
// (token, expert slot); ne11 == 1 (up/gate: all experts of a token share the input) gives one column per token
static __global__ void moea_quantize(const float * __restrict__ x, const int64_t s_slot, const int64_t s_tok, const int ne11,
        int8_t * __restrict__ aq, float2 * __restrict__ ads, const int K) {
    const int lane = threadIdx.x & 31;
    const int blk  = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int nb   = K >> 5;
    if (blk >= nb) return;
    const int t = blockIdx.y;   // column
    const float v = x[(int64_t) (t / ne11) * s_tok + (int64_t) (t % ne11) * s_slot + blk * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
    const int q = (int) lrintf(v * id);
    aq[(int64_t) t * K + blk * 32 + lane] = (int8_t) q;
    int qs = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qs += __shfl_xor_sync(0xffffffff, qs, o);
    if (lane == 0) ads[(int64_t) t * nb + blk] = make_float2(dd, (float) qs);
}

// list of experts with at least one token (order irrelevant: every block only writes the rows of its own expert)
static __global__ void moea_active_experts(const int32_t * __restrict__ bounds, const int n_expert, int32_t * __restrict__ list, int32_t * __restrict__ count) {
    __shared__ int s_cnt;
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        if (bounds[e + 1] > bounds[e]) { list[atomicAdd(&s_cnt, 1)] = e; }
    }
    __syncthreads();
    if (threadIdx.x == 0) *count = s_cnt;
}

__device__ __forceinline__ int moea_dot16(const int4 & w, const int4 & a) {
    int s = __dp4a(w.x, a.x, 0); s = __dp4a(w.y, a.y, s); s = __dp4a(w.z, a.z, s); return __dp4a(w.w, a.w, s);
}

// ---------------------------------------------------------------------------------------------------------------------
// Q4_K: block = 256 weights = 144 bytes {d, dmin, scales[12], qs[128]}. 8 lanes per row; lane owns 16 bytes of qs =
// 16 weights of sub-block 2g (low nibbles) + 16 weights of sub-block 2g+1 (high nibbles), g = (lane>>1)&3.
template <int TG, bool GATE>
__device__ __forceinline__ void moea_q4k_rows(const char * __restrict__ xr, const char * __restrict__ gr,
        const int8_t * __restrict__ aq, const float2 * __restrict__ ads, const int * __restrict__ tok,
        const int nb, const int nbk, const int K, const int lane, float * __restrict__ acc, float * __restrict__ accg) {
    const int g    = (lane >> 1) & 3;
    const int hlf  = lane & 1;
    const int j0   = 2 * g;
    const bool hi  = j0 >= 4;
    const int sh   = 8 * (j0 & 3);
    const float msel = hlf ? 0.f : 1.f;       // the per-sub-block min term is applied once (both lanes share the sum)
#pragma unroll
    for (int t = 0; t < TG; ++t) { acc[t] = 0.f; accg[t] = 0.f; }
#pragma unroll 2
    for (int kb = 0; kb < nbk; ++kb) {
        const int ablk = kb * 8 + j0;
        int4 alo[TG], ahi[TG]; float4 sc[TG];   // sc = {d0, sum0, d1, sum1}
#pragma unroll
        for (int t = 0; t < TG; ++t) {
            const int8_t * a = aq + (size_t) tok[t] * K + (size_t) ablk * 32 + hlf * 16;
            alo[t] = *reinterpret_cast<const int4 *>(a);
            ahi[t] = *reinterpret_cast<const int4 *>(a + 32);
            sc[t]  = *reinterpret_cast<const float4 *>(ads + (size_t) tok[t] * nb + ablk);
        }
#pragma unroll
        for (int which = 0; which < (GATE ? 2 : 1); ++which) {
            const char * bp = (which == 0 ? xr : gr) + (size_t) kb * 144;
            const int4 hdr = *reinterpret_cast<const int4 *>(bp);
            const int4 qv  = *reinterpret_cast<const int4 *>(bp + 16 + g * 32 + hlf * 16);
            const half2 dm = *reinterpret_cast<const half2 *>(&hdr.x);
            const float d = __low2float(dm), dmin = __high2float(dm);
            const uint32_t A = ((uint32_t) hdr.y) >> sh, B = ((uint32_t) hdr.z) >> sh, C = ((uint32_t) hdr.w) >> sh;
            const int s0 = hi ? (int) ((C & 0xF) | ((A >> 2) & 0x30))          : (int) (A & 63);
            const int m0 = hi ? (int) (((C >> 4) & 0xF) | ((B >> 2) & 0x30))   : (int) (B & 63);
            const int s1 = hi ? (int) (((C >> 8) & 0xF) | ((A >> 10) & 0x30))  : (int) ((A >> 8) & 63);
            const int m1 = hi ? (int) (((C >> 12) & 0xF) | ((B >> 10) & 0x30)) : (int) ((B >> 8) & 63);
            const int4 lo = make_int4(qv.x & 0x0F0F0F0F, qv.y & 0x0F0F0F0F, qv.z & 0x0F0F0F0F, qv.w & 0x0F0F0F0F);
            const int4 hv = make_int4((qv.x >> 4) & 0x0F0F0F0F, (qv.y >> 4) & 0x0F0F0F0F, (qv.z >> 4) & 0x0F0F0F0F, (qv.w >> 4) & 0x0F0F0F0F);
            const float ds0 = d * (float) s0, ds1 = d * (float) s1, dm0 = dmin * (float) m0 * msel, dm1 = dmin * (float) m1 * msel;
#pragma unroll
            for (int t = 0; t < TG; ++t) {
                const float v = sc[t].x * (ds0 * (float) moea_dot16(lo, alo[t]) - dm0 * sc[t].y)
                              + sc[t].z * (ds1 * (float) moea_dot16(hv, ahi[t]) - dm1 * sc[t].w);
                if (which == 0) acc[t] += v; else accg[t] += v;
            }
        }
    }
}

template <bool GATE>
__global__ void __launch_bounds__(256, 3) moea_q4k(const char * __restrict__ wx, const char * __restrict__ wg,
        const int8_t * __restrict__ aq, const float2 * __restrict__ ads,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const int32_t * __restrict__ elist, const int32_t * __restrict__ ecount,
        float * __restrict__ dst, const int N, const int K, const int col_div, const size_t expert_stride, const size_t row_stride) {
    if ((int) blockIdx.y >= *ecount) return;
    const int e  = elist[blockIdx.y];
    const int c0 = bounds[e], c1 = bounds[e + 1];
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row  = warp * 4 + (lane >> 3);
    if (row >= N) return;                        // N % 4 == 0: warp-uniform
    const int nb = K >> 5, nbk = K >> 8;
    const char * xr = wx + (size_t) e * expert_stride + (size_t) row * row_stride;
    const char * gr = GATE ? wg + (size_t) e * expert_stride + (size_t) row * row_stride : nullptr;
    for (int t0 = c0; t0 < c1; t0 += 2) {
        const int ntok = min(2, c1 - t0);
        int tok[2];
        tok[0] = ids_dst[t0] / col_div;
        tok[1] = ntok > 1 ? ids_dst[t0 + 1] / col_div : tok[0];
        float acc[2] = {0.f, 0.f}, accg[2] = {0.f, 0.f};
        if (ntok == 2) moea_q4k_rows<2, GATE>(xr, gr, aq, ads, tok, nb, nbk, K, lane, acc, accg);
        else           moea_q4k_rows<1, GATE>(xr, gr, aq, ads, tok, nb, nbk, K, lane, acc, accg);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            float v = acc[t], w = accg[t];
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) { v += __shfl_xor_sync(0xffffffff, v, o); w += __shfl_xor_sync(0xffffffff, w, o); }
            if ((lane & 7) == 0 && t < ntok) {
                if (GATE) v = v * (w / (1.0f + expf(-w)));   // x * silu(gate)
                dst[(size_t) ids_dst[t0 + t] * N + row] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Q5_1: block = 32 weights = 24 bytes {d, m, qh, qs[16]} (8-byte aligned). 4 lanes per row (8 rows per warp), one block
// per lane per iteration.
__device__ __forceinline__ uint32_t moea_q5_spread(const uint32_t nib) {   // bits 0..3 -> bit 4 of bytes 0..3
    return ((nib * 0x00204081u) & 0x01010101u) << 4;
}

template <int TG>
__device__ __forceinline__ void moea_q51_rows(const char * __restrict__ xr, const int8_t * __restrict__ aq, const float2 * __restrict__ ads,
        const int * __restrict__ tok, const int nb, const int K, const int l4, float * __restrict__ acc) {
#pragma unroll
    for (int t = 0; t < TG; ++t) acc[t] = 0.f;
#pragma unroll 2
    for (int blk = l4; blk < nb; blk += 4) {
        int4 alo[TG], ahi[TG]; float2 sc[TG];
#pragma unroll
        for (int t = 0; t < TG; ++t) {
            const int8_t * a = aq + (size_t) tok[t] * K + (size_t) blk * 32;
            alo[t] = *reinterpret_cast<const int4 *>(a);
            ahi[t] = *reinterpret_cast<const int4 *>(a + 16);
            sc[t]  = ads[(size_t) tok[t] * nb + blk];
        }
        const char * bp = xr + (size_t) blk * 24;
        const uint2 h0 = *reinterpret_cast<const uint2 *>(bp);
        const uint2 q0 = *reinterpret_cast<const uint2 *>(bp + 8);
        const uint2 q1 = *reinterpret_cast<const uint2 *>(bp + 16);
        const half2 dm = *reinterpret_cast<const half2 *>(&h0.x);
        const float d = __low2float(dm), m = __high2float(dm);
        const uint32_t qh = h0.y;
        const uint32_t qs[4] = { q0.x, q0.y, q1.x, q1.y };
        int lo[4], hv[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t qhw = qh >> (4 * w);
            lo[w] = (int) ((qs[w] & 0x0F0F0F0Fu) | moea_q5_spread(qhw & 0xF));
            hv[w] = (int) (((qs[w] >> 4) & 0x0F0F0F0Fu) | moea_q5_spread((qhw >> 16) & 0xF));
        }
        const int4 lo4 = make_int4(lo[0], lo[1], lo[2], lo[3]), hv4 = make_int4(hv[0], hv[1], hv[2], hv[3]);
#pragma unroll
        for (int t = 0; t < TG; ++t) {
            const int s = moea_dot16(lo4, alo[t]) + moea_dot16(hv4, ahi[t]);
            acc[t] += sc[t].x * (d * (float) s + m * sc[t].y);
        }
    }
}

__global__ void __launch_bounds__(256, 3) moea_q51(const char * __restrict__ wx,
        const int8_t * __restrict__ aq, const float2 * __restrict__ ads,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const int32_t * __restrict__ elist, const int32_t * __restrict__ ecount,
        float * __restrict__ dst, const int N, const int K, const int col_div, const size_t expert_stride, const size_t row_stride) {
    if ((int) blockIdx.y >= *ecount) return;
    const int e  = elist[blockIdx.y];
    const int c0 = bounds[e], c1 = bounds[e + 1];
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row  = warp * 8 + (lane >> 2);
    if (row >= N) return;                        // N % 8 == 0: warp-uniform
    const int nb = K >> 5;
    const int l4 = lane & 3;
    const char * xr = wx + (size_t) e * expert_stride + (size_t) row * row_stride;
    for (int t0 = c0; t0 < c1; t0 += 2) {
        const int ntok = min(2, c1 - t0);
        int tok[2];
        tok[0] = ids_dst[t0] / col_div;
        tok[1] = ntok > 1 ? ids_dst[t0 + 1] / col_div : tok[0];
        float acc[2] = {0.f, 0.f};
        if (ntok == 2) moea_q51_rows<2>(xr, aq, ads, tok, nb, K, l4, acc);
        else           moea_q51_rows<1>(xr, aq, ads, tok, nb, K, l4, acc);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            float v = acc[t];
            v += __shfl_xor_sync(0xffffffff, v, 1);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            if (l4 == 0 && t < ntok) dst[(size_t) ids_dst[t0 + t] * N + row] = v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
bool ggml_cuda_moea_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                              const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_UNUSED(ctx);
    if (!moea_enabled() || ids == nullptr) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_1) return false;
    const int64_t K = src0->ne[0], N = src0->ne[1];
    if (src0->ne[3] != 1 || K < 256) return false;
    if (src0->type == GGML_TYPE_Q4_K && (K % 256 != 0 || N % 4 != 0)) return false;
    // unfused Q4_K is no faster than mmvq, but it is kept on this path so the numerics do not depend on whether the
    // scheduler happened to fuse up/gate/swiglu (fusion depends on node order, which differs between device splits)
    if (src0->type == GGML_TYPE_Q5_1 && (K % 32 != 0 || N % 8 != 0)) return false;
    if ((src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]) || src1->ne[3] != 1 || src1->ne[2] > MOEA_MAX_TOK) return false;   // shared or per-slot activations
    if (src1->nb[0] != sizeof(float)) return false;
    if (ids->ne[1] != src1->ne[2] || ids->nb[0] != sizeof(int32_t)) return false;
    if (!ggml_is_contiguous(dst) || dst->ne[0] != N || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) return false;
    if (src0->nb[0] != ggml_type_size(src0->type)) return false;
    if (fusion) {
        if (fusion->x_bias || fusion->gate_bias || fusion->x_scale || fusion->gate_scale) return false;
        if (fusion->gate) {
            if (src0->type != GGML_TYPE_Q4_K) return false;   // fused gate only for the Q4_K up/gate pair
            if (fusion->gate->type != src0->type || !ggml_are_same_stride(fusion->gate, src0)) return false;
            if (fusion->glu_op != GGML_GLU_OP_SWIGLU) return false;
        }
    }
    return true;
}

bool ggml_cuda_mul_mat_moea(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                            ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], N = (int) src0->ne[1], n_expert = (int) src0->ne[2];
    const int n_tokens = (int) src1->ne[2], n_expert_used = (int) ids->ne[0];
    const int ne11 = (int) src1->ne[1];                 // 1: shared per token; n_expert_used: one column per (token, slot)
    const int n_cols = n_tokens * ne11;
    const int col_div = ne11 == 1 ? n_expert_used : 1;   // ids_dst = token*n_expert_used + slot -> activation column
    const int nb = K / 32;
    ggml_cuda_pool_alloc<int8_t> aq(ctx.pool(), (size_t) n_cols * K);
    ggml_cuda_pool_alloc<float2> ads(ctx.pool(), (size_t) n_cols * nb);
    moea_quantize<<<dim3((nb + 7) / 8, n_cols), 256, 0, stream>>>((const float *) src1->data, src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), ne11,
        aq.get(), ads.get(), K);
    // expert grouping: compact (token, slot) list sorted by expert + bounds + list of active experts
    const int ne_rows = n_tokens * n_expert_used;
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);
    ggml_cuda_pool_alloc<int32_t> elist(ctx.pool(), n_expert + 1);
    const int si1  = (int) (ids->nb[1] / ids->nb[0]);
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        n_expert, n_tokens, n_expert_used, (int) src1->ne[1], si1, sis1, false, stream);
    moea_active_experts<<<1, 1024, 0, stream>>>(bounds.get(), n_expert, elist.get(), elist.get() + n_expert);
    const int n_active_max = std::min(n_expert, ne_rows);
    const size_t expert_stride = src0->nb[2], row_stride = src0->nb[1];
    if (src0->type == GGML_TYPE_Q4_K) {
        const dim3 grid((N + 31) / 32, n_active_max);
        if (fusion && fusion->gate) {
            moea_q4k<true><<<grid, 256, 0, stream>>>((const char *) src0->data, (const char *) fusion->gate->data, aq.get(), ads.get(),
                ids_dst.get(), bounds.get(), elist.get(), elist.get() + n_expert, (float *) dst->data, N, K, col_div, expert_stride, row_stride);
        } else {
            moea_q4k<false><<<grid, 256, 0, stream>>>((const char *) src0->data, nullptr, aq.get(), ads.get(),
                ids_dst.get(), bounds.get(), elist.get(), elist.get() + n_expert, (float *) dst->data, N, K, col_div, expert_stride, row_stride);
        }
    } else {
        const dim3 grid((N + 63) / 64, n_active_max);
        moea_q51<<<grid, 256, 0, stream>>>((const char *) src0->data, aq.get(), ads.get(),
            ids_dst.get(), bounds.get(), elist.get(), elist.get() + n_expert, (float *) dst->data, N, K, col_div, expert_stride, row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
