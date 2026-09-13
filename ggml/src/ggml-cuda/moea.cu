#include "moea.cuh"
#include "mmid.cuh"

#define MOEA_MAX_TOK 8

// opt-in (NEXT_MOEA=1): correct, but not yet faster than mmvq on the CMP 170HX (register pressure in the Q4_K path)
static bool moea_enabled() {
    static const bool enabled = [] { const char * e = getenv("NEXT_MOEA"); return e != nullptr && atoi(e) != 0; }();
    return enabled;
}

// activations F32 [K, ntok] (column stride s_col elements) -> int8 [ntok][K], scale [ntok][K/32], int block sums [ntok][K/32]
static __global__ void moea_quantize(const float * __restrict__ x, const int64_t s_col, int8_t * __restrict__ aq, float * __restrict__ ad,
        float * __restrict__ asum, const int K) {
    const int lane = threadIdx.x & 31;
    const int blk  = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int nb   = K >> 5;
    if (blk >= nb) return;
    const int t = blockIdx.y;
    const float v = x[(int64_t) t * s_col + blk * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
    const int q = (int) lrintf(v * id);
    aq[(int64_t) t * K + blk * 32 + lane] = (int8_t) q;
    int qs = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qs += __shfl_xor_sync(0xffffffff, qs, o);
    if (lane == 0) { ad[(int64_t) t * nb + blk] = dd; asum[(int64_t) t * nb + blk] = (float) qs; }
}

__device__ __forceinline__ int dp4a_sum_bytes(int v) { return __dp4a(v, 0x01010101, 0); }

// ---------------------------------------------------------------------------------------------------------------------
// Q4_K experts: block = 256 weights = 144 bytes {d, dmin, scales[12], qs[128]}. 8 lanes per block, each lane owns
// 16 bytes of qs = 16 weights of sub-block 2g (low nibbles) + 16 weights of sub-block 2g+1 (high nibbles), g = group 0..3.
// A warp covers 4 blocks (1024 weights) per iteration; two rows per warp; up to MOEA_MAX_TOK tokens of the same expert.
template <bool GATE>
__global__ void __launch_bounds__(256) moea_q4k(const char * __restrict__ wx, const char * __restrict__ wg,
        const int8_t * __restrict__ aq, const float * __restrict__ ad, const float * __restrict__ asum,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        float * __restrict__ dst, const int N, const int K, const int n_expert_used, const size_t expert_stride, const size_t row_stride,
        const int glu_op) {
    const int e = blockIdx.y;
    const int c0 = bounds[e], c1 = bounds[e + 1];
    const int ntok = c1 - c0;
    if (ntok <= 0) return;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n0 = warp * 2;
    if (n0 >= N) return;
    const bool has_r1 = n0 + 1 < N;
    const int nb   = K >> 5;        // 32-wide activation blocks
    const int nbk  = K >> 8;        // Q4_K blocks per row
    const int g    = (lane >> 1) & 3;   // 64-weight group within the block
    const int hlf  = lane & 1;          // which 16-byte half of the group's 32 bytes
    const int bofs = lane >> 3;         // block offset within the warp's 4-block window
    int tok[MOEA_MAX_TOK];
#pragma unroll
    for (int t = 0; t < MOEA_MAX_TOK; ++t) tok[t] = t < ntok ? ids_dst[c0 + t] / n_expert_used : 0;
    float acc[2][MOEA_MAX_TOK], accg[2][MOEA_MAX_TOK];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) { acc[r][t] = 0.f; accg[r][t] = 0.f; }
    const char * xe = wx + (size_t) e * expert_stride;
    const char * ge = GATE ? wg + (size_t) e * expert_stride : nullptr;
    for (int kb = bofs; kb < nbk; kb += 4) {
        const int j0 = 2 * g;                       // sub-block index of the low nibbles
        const int ablk0 = kb * 8 + j0;              // activation block for sub-block j0 (32-wide)
        const int ablk1 = ablk0 + 1;
        // activations for this lane's 16+16 weights, per token
        int4 alo[MOEA_MAX_TOK], ahi[MOEA_MAX_TOK];
        float da0[MOEA_MAX_TOK], da1[MOEA_MAX_TOK];
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) {
            if (t < ntok) {
                const int8_t * a = aq + (size_t) tok[t] * K;
                alo[t] = *reinterpret_cast<const int4 *>(a + ablk0 * 32 + hlf * 16);
                ahi[t] = *reinterpret_cast<const int4 *>(a + ablk1 * 32 + hlf * 16);
                da0[t] = ad[(size_t) tok[t] * nb + ablk0];
                da1[t] = ad[(size_t) tok[t] * nb + ablk1];
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 1 && !has_r1) break;
            const int n = n0 + r;
#pragma unroll
            for (int which = 0; which < (GATE ? 2 : 1); ++which) {
                const char * blockp = (which == 0 ? xe : ge) + (size_t) n * row_stride + (size_t) kb * 144;
                const int4 hdr = *reinterpret_cast<const int4 *>(blockp);            // d, dmin, scales[12]
                const int4 qv  = *reinterpret_cast<const int4 *>(blockp + 16 + g * 32 + hlf * 16);
                const half2 dm = *reinterpret_cast<const half2 *>(&hdr.x);
                const float d = __low2float(dm), dmin = __high2float(dm);
                const uint8_t * sc = reinterpret_cast<const uint8_t *>(&hdr.y);   // scales[0..11] = bytes 4..15 of hdr
                int s0, m0, s1, m1;
                {   // get_scale_min_k4 for j0 (< 4) and j0+1 (< 4): q[j]&63, q[j+4]&63
                    s0 = sc[j0] & 63; m0 = sc[j0 + 4] & 63;
                    s1 = sc[j0 + 1] & 63; m1 = sc[j0 + 5] & 63;
                }
                // wait: for the second half of sub-blocks (j >= 4) the 6-bit values are packed differently
                if (j0 >= 4) {
                    s0 = (sc[j0 + 4] & 0xF) | ((sc[j0 - 4] >> 6) << 4);  m0 = (sc[j0 + 4] >> 4) | ((sc[j0] >> 6) << 4);
                    s1 = (sc[j0 + 5] & 0xF) | ((sc[j0 - 3] >> 6) << 4);  m1 = (sc[j0 + 5] >> 4) | ((sc[j0 + 1] >> 6) << 4);
                }
                const int lx = qv.x & 0x0F0F0F0F, ly = qv.y & 0x0F0F0F0F, lz = qv.z & 0x0F0F0F0F, lw = qv.w & 0x0F0F0F0F;
                const int hx = (qv.x >> 4) & 0x0F0F0F0F, hy = (qv.y >> 4) & 0x0F0F0F0F, hz = (qv.z >> 4) & 0x0F0F0F0F, hw = (qv.w >> 4) & 0x0F0F0F0F;
#pragma unroll
                for (int t = 0; t < MOEA_MAX_TOK; ++t) {
                    if (t >= ntok) break;
                    int dlo = 0, dhi = 0;
                    dlo = __dp4a(lx, alo[t].x, dlo); dlo = __dp4a(ly, alo[t].y, dlo); dlo = __dp4a(lz, alo[t].z, dlo); dlo = __dp4a(lw, alo[t].w, dlo);
                    dhi = __dp4a(hx, ahi[t].x, dhi); dhi = __dp4a(hy, ahi[t].y, dhi); dhi = __dp4a(hz, ahi[t].z, dhi); dhi = __dp4a(hw, ahi[t].w, dhi);
                    const int slo = dp4a_sum_bytes(alo[t].x) + dp4a_sum_bytes(alo[t].y) + dp4a_sum_bytes(alo[t].z) + dp4a_sum_bytes(alo[t].w);
                    const int shi = dp4a_sum_bytes(ahi[t].x) + dp4a_sum_bytes(ahi[t].y) + dp4a_sum_bytes(ahi[t].z) + dp4a_sum_bytes(ahi[t].w);
                    const float v = d * (da0[t] * (float) (s0 * dlo) + da1[t] * (float) (s1 * dhi))
                                  - dmin * (da0[t] * (float) (m0 * slo) + da1[t] * (float) (m1 * shi));
                    if (which == 0) acc[r][t] += v; else accg[r][t] += v;
                }
            }
        }
    }
    // reduce across the warp
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) {
            float v = acc[r][t], w = accg[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffff, v, o); w += __shfl_xor_sync(0xffffffff, w, o); }
            acc[r][t] = v; accg[r][t] = w;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 1 && !has_r1) break;
            for (int t = 0; t < ntok; ++t) {
                float v = acc[r][t];
                if (GATE) {
                    const float gv = accg[r][t];
                    v = glu_op == 0 ? v * (gv / (1.0f + expf(-gv))) : v * gv;   // 0: swiglu, else plain product
                }
                dst[(size_t) ids_dst[c0 + t] * N + n0 + r] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Q5_1 experts: block = 32 weights = 24 bytes {d, m, qh, qs[16]} (8-byte aligned). One lane per block, two rows per warp.
__global__ void __launch_bounds__(256) moea_q51(const char * __restrict__ wx,
        const int8_t * __restrict__ aq, const float * __restrict__ ad, const float * __restrict__ asum,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        float * __restrict__ dst, const int N, const int K, const int n_expert_used, const size_t expert_stride, const size_t row_stride) {
    const int e = blockIdx.y;
    const int c0 = bounds[e], c1 = bounds[e + 1];
    const int ntok = c1 - c0;
    if (ntok <= 0) return;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n0 = warp * 2;
    if (n0 >= N) return;
    const bool has_r1 = n0 + 1 < N;
    const int nb = K >> 5;
    int tok[MOEA_MAX_TOK];
#pragma unroll
    for (int t = 0; t < MOEA_MAX_TOK; ++t) tok[t] = t < ntok ? ids_dst[c0 + t] / n_expert_used : 0;
    float acc[2][MOEA_MAX_TOK];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) acc[r][t] = 0.f;
    const char * xe = wx + (size_t) e * expert_stride;
    for (int blk = lane; blk < nb; blk += 32) {
        int4 alo[MOEA_MAX_TOK], ahi[MOEA_MAX_TOK]; float da[MOEA_MAX_TOK], sa[MOEA_MAX_TOK];
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) {
            if (t < ntok) {
                const int8_t * a = aq + (size_t) tok[t] * K + blk * 32;
                alo[t] = *reinterpret_cast<const int4 *>(a); ahi[t] = *reinterpret_cast<const int4 *>(a + 16);
                da[t] = ad[(size_t) tok[t] * nb + blk]; sa[t] = asum[(size_t) tok[t] * nb + blk];
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 1 && !has_r1) break;
            const char * bp = xe + (size_t) (n0 + r) * row_stride + (size_t) blk * 24;
            const uint2 h0 = *reinterpret_cast<const uint2 *>(bp);          // d,m | qh
            const uint2 q0 = *reinterpret_cast<const uint2 *>(bp + 8);      // qs[0..7]
            const uint2 q1 = *reinterpret_cast<const uint2 *>(bp + 16);     // qs[8..15]
            const half2 dm = *reinterpret_cast<const half2 *>(&h0.x);
            const float d = __low2float(dm), m = __high2float(dm);
            const uint32_t qh = h0.y;
            const int qs[4] = { (int) q0.x, (int) q0.y, (int) q1.x, (int) q1.y };
            int lo[4], hi[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                // low nibbles: weights 4w..4w+3 (bit 4 from qh bits 4w..4w+3); high nibbles: weights 16+4w.. (qh bits 16+4w..)
                int l = qs[w] & 0x0F0F0F0F, hgh = (qs[w] >> 4) & 0x0F0F0F0F;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    l   |= ((qh >> (4 * w + b)) & 1) << (8 * b + 4);
                    hgh |= ((qh >> (16 + 4 * w + b)) & 1) << (8 * b + 4);
                }
                lo[w] = l; hi[w] = hgh;
            }
#pragma unroll
            for (int t = 0; t < MOEA_MAX_TOK; ++t) {
                if (t >= ntok) break;
                int s = 0;
                s = __dp4a(lo[0], alo[t].x, s); s = __dp4a(lo[1], alo[t].y, s); s = __dp4a(lo[2], alo[t].z, s); s = __dp4a(lo[3], alo[t].w, s);
                s = __dp4a(hi[0], ahi[t].x, s); s = __dp4a(hi[1], ahi[t].y, s); s = __dp4a(hi[2], ahi[t].z, s); s = __dp4a(hi[3], ahi[t].w, s);
                acc[r][t] += da[t] * (d * (float) s + m * sa[t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < MOEA_MAX_TOK; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[r][t] = v;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 1 && !has_r1) break;
            for (int t = 0; t < ntok; ++t) dst[(size_t) ids_dst[c0 + t] * N + n0 + r] = acc[r][t];
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
    if (src0->ne[3] != 1 || K % 256 != 0 || K < 256 || N % 2 != 0) return false;   // Q4_K needs whole 256-blocks; Q5_1 fine
    if (src1->ne[1] != 1 || src1->ne[3] != 1 || src1->ne[2] > MOEA_MAX_TOK) return false;  // one activation column per token
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
    const int nb = K / 32;
    // per-token activation quantization (all experts of a token share the same input column)
    ggml_cuda_pool_alloc<int8_t> aq(ctx.pool(), (size_t) n_tokens * K);
    ggml_cuda_pool_alloc<float>  ad(ctx.pool(), (size_t) n_tokens * nb);
    ggml_cuda_pool_alloc<float>  as(ctx.pool(), (size_t) n_tokens * nb);
    moea_quantize<<<dim3((nb + 7) / 8, n_tokens), 256, 0, stream>>>((const float *) src1->data, src1->nb[2] / sizeof(float), aq.get(), ad.get(), as.get(), K);
    // expert grouping: compact (token, slot) list sorted by expert + bounds
    const int ne_rows = n_tokens * n_expert_used;
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);
    const int si1  = (int) (ids->nb[1] / ids->nb[0]);
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        n_expert, n_tokens, n_expert_used, (int) src1->ne[1], si1, sis1, false, stream);
    const int nwarps = (N + 1) / 2;
    const dim3 grid((nwarps + 7) / 8, n_expert);
    const size_t expert_stride = src0->nb[2], row_stride = src0->nb[1];
    if (src0->type == GGML_TYPE_Q4_K) {
        const bool gate = fusion && fusion->gate;
        if (gate) {
            moea_q4k<true><<<grid, 256, 0, stream>>>((const char *) src0->data, (const char *) fusion->gate->data, aq.get(), ad.get(), as.get(),
                ids_dst.get(), bounds.get(), (float *) dst->data, N, K, n_expert_used, expert_stride, row_stride, 0);
        } else {
            moea_q4k<false><<<grid, 256, 0, stream>>>((const char *) src0->data, nullptr, aq.get(), ad.get(), as.get(),
                ids_dst.get(), bounds.get(), (float *) dst->data, N, K, n_expert_used, expert_stride, row_stride, 0);
        }
    } else {
        moea_q51<<<grid, 256, 0, stream>>>((const char *) src0->data, aq.get(), ad.get(), as.get(),
            ids_dst.get(), bounds.get(), (float *) dst->data, N, K, n_expert_used, expert_stride, row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
