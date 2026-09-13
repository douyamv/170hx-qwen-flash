#include "qsa.cuh"
#include "../ggml-qsa.h"

static __global__ void qsa_pool_norm(const char * raw, const int32_t * ids, const float * gamma,
        float * out, size_t row_stride, float eps) {
    const int64_t b = blockIdx.x;
    const int c = threadIdx.x;
    float x = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const char * block = raw + size_t(ids[b*4+j])*row_stride + (c/32)*34;
        x += __half2float(*reinterpret_cast<const half *>(block)) * float(reinterpret_cast<const int8_t *>(block+2)[c%32]);
    }
    x *= .25f;
    float ss = x*x;
#pragma unroll
    for (int offset = 16; offset; offset /= 2) ss += __shfl_down_sync(0xffffffff,ss,offset);
    __shared__ float sums[4];
    __shared__ float inv;
    if ((c & 31) == 0) sums[c/32] = ss;
    __syncthreads();
    if (c == 0) inv = rsqrtf((sums[0]+sums[1]+sums[2]+sums[3])/128 + eps);
    __syncthreads();
    out[b*128+c] = (x*inv)*gamma[c];
}
static __global__ void qsa_expand(const char * score, const int32_t * ids, const char * mask,
        float * out, int64_t n, size_t score_stride, size_t mask_stride) {
    const int64_t j = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q = blockIdx.y;
    if (j < n) {
        const float * s = reinterpret_cast<const float *>(score + q*score_stride);
        const half * m = reinterpret_cast<const half *>(mask + q*mask_stride);
        out[q*n+j] = s[ids[j]] + __half2float(m[j]);
    }
}
static __global__ void qsa_mask_select(const char * mask, const char * ids, half * out,
        int64_t ns, int64_t np, size_t mask_stride, size_t ids_stride) {
    const int64_t j = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q = blockIdx.y;
    if (j < np) {
        const half * m = reinterpret_cast<const half *>(mask + q*mask_stride);
        const int32_t * row = reinterpret_cast<const int32_t *>(ids + q*ids_stride);
        out[q*np+j] = j < ns ? m[row[j]] : __float2half(-INFINITY);
    }
}
// NEXT: fused get_rows(Q8_0) + pad + F32->F16 cast for the QSA compact path. One block per (selected row, query);
// each thread dequantizes 4 consecutive elements (2-byte aligned quant loads: Q8_0 blocks are 34 bytes) and stores
// 8 bytes. Padding rows (s >= ns) are zero-filled. Replaces three kernels (~175 us) per K/V per layer with one (~8 us).
static __global__ void qsa_gather_f16(const char * __restrict__ cache, const char * __restrict__ ids, half * __restrict__ out,
        int64_t d, int64_t nh, int64_t ns, int64_t np, size_t cell_stride, size_t ids_stride) {
    const int64_t s = blockIdx.x;
    const int64_t q = blockIdx.y;
    const int64_t n = d*nh;
    if (s >= ns) {
        for (int64_t g = threadIdx.x*4; g < n; g += blockDim.x*4) {
            const int64_t h = g / d, i = g - h*d;
            *reinterpret_cast<uint2 *>(out + d*(s + np*(h + nh*q)) + i) = make_uint2(0u, 0u);
        }
        return;
    }
    const int32_t cell = reinterpret_cast<const int32_t *>(ids + q*ids_stride)[s];
    const char * row = cache + size_t(cell)*cell_stride;
    for (int64_t g = threadIdx.x*4; g < n; g += blockDim.x*4) {
        const char * block = row + (g >> 5)*34;
        const float dsc = __half2float(*reinterpret_cast<const half *>(block));
        const uint16_t * qp = reinterpret_cast<const uint16_t *>(block + 2 + (g & 31));
        const uint16_t q01 = qp[0], q23 = qp[1];
        const float v0 = dsc*float(int8_t(q01 & 0xff)), v1 = dsc*float(int8_t(q01 >> 8));
        const float v2 = dsc*float(int8_t(q23 & 0xff)), v3 = dsc*float(int8_t(q23 >> 8));
        const int64_t h = g / d, i = g - h*d;
        half2 * o = reinterpret_cast<half2 *>(out + d*(s + np*(h + nh*q)) + i);
        o[0] = __floats2half2_rn(v0, v1);
        o[1] = __floats2half2_rn(v2, v3);
    }
}
static __global__ void hc_mix_tail(const char * xn, const char * gl, float * out, int64_t n_embd, int64_t hc, size_t xn_stride, size_t gl_stride) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (i >= n_embd) return;
    const float * x = reinterpret_cast<const float *>(xn + t*xn_stride);
    const float * g = reinterpret_cast<const float *>(gl + t*gl_stride);
    float s = 0.0f;
    for (int64_t h = 0; h < hc; ++h) {
        const int64_t k = h*n_embd + i;
        s += x[k] * (1.0f/(1.0f + expf(-g[k])));
    }
    out[t*n_embd + i] = s*(1.0f/(float) hc);
}
static __global__ void hc_combine(const char * res, const char * bo, const char * inj, float * out, int64_t n_embd, int64_t hc,
        size_t res_nb1, size_t res_nb2, size_t bo_stride, size_t inj_stride) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t h = blockIdx.y;
    const int64_t t = blockIdx.z;
    if (i >= n_embd) return;
    const float lg = reinterpret_cast<const float *>(inj + t*inj_stride)[h]*(1.0f/(float) hc);
    const float w  = 2.0f*(1.0f/(1.0f + expf(-lg)));
    const float r  = reinterpret_cast<const float *>(res + t*res_nb2 + h*res_nb1)[i];
    const float b  = reinterpret_cast<const float *>(bo + t*bo_stride)[i];
    out[(t*hc + h)*n_embd + i] = r + b*w;
}
void ggml_cuda_op_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_qsa_kind(dst) == 5) {
        const auto * xn = dst->src[0]; const auto * gl = dst->src[1];
        const int64_t n_embd = dst->ne[0], nt = dst->ne[1], hc = xn->ne[0]/n_embd;
        hc_mix_tail<<<dim3((n_embd+255)/256, nt), 256, 0, ctx.stream()>>>((const char *)xn->data, (const char *)gl->data, (float *)dst->data, n_embd, hc, xn->nb[1], gl->nb[1]);
        return;
    }
    if (ggml_qsa_kind(dst) == 6) {
        const auto * res = dst->src[0]; const auto * bo = dst->src[1]; const auto * inj = dst->src[2];
        const int64_t n_embd = dst->ne[0], hc = dst->ne[1], nt = dst->ne[2];
        hc_combine<<<dim3((n_embd+255)/256, hc, nt), 256, 0, ctx.stream()>>>((const char *)res->data, (const char *)bo->data, (const char *)inj->data, (float *)dst->data,
            n_embd, hc, res->nb[1], res->nb[2], bo->nb[1], inj->nb[1]);
        return;
    }
    const auto * a = dst->src[0];
    const auto * ids = dst->src[1];
    const auto * c = dst->src[2];
    if (ggml_qsa_kind(dst) == 1) {
        qsa_pool_norm<<<dst->ne[1],128,0,ctx.stream()>>>((const char *)a->data,(const int32_t *)ids->data,
            (const float *)c->data,(float *)dst->data,a->nb[1],ggml_qsa_epsilon(dst));
    } else if (ggml_qsa_kind(dst) == 4) {
        qsa_gather_f16<<<dim3(dst->ne[1],dst->ne[3]),128,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,dst->ne[0],dst->ne[2],ids->ne[0],dst->ne[1],a->nb[2],ids->nb[1]);
    } else if (ggml_qsa_kind(dst) == 3) {
        qsa_mask_select<<<dim3((dst->ne[0]+255)/256,dst->ne[3]),256,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,ids->ne[0],dst->ne[0],a->nb[1],ids->nb[1]);
    } else {
        qsa_expand<<<dim3((dst->ne[0]+255)/256,dst->ne[1]),256,0,ctx.stream()>>>((const char *)a->data,
            (const int32_t *)ids->data,(const char *)c->data,(float *)dst->data,dst->ne[0],a->nb[1],c->nb[1]);
    }
}
