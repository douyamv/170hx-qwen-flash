#include "ggml-qsa.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static void qsa_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const int kind = (int)(intptr_t) userdata;
    const struct ggml_tensor * a = dst->src[0];
    const int32_t * ids = dst->src[1]->data;
    float * out = dst->data;
    if (kind == 1) {
        const float * gamma = dst->src[2]->data;
        for (int64_t b = ith; b < dst->ne[1]; b += nth) {
            float x[128] = {0}, row[128];
            for (int j = 0; j < 4; ++j) {
                dequantize_row_q8_0((const block_q8_0 *)((const char *)a->data + (size_t)ids[b*4+j]*a->nb[1]), row, 128);
                for (int d = 0; d < 128; ++d) x[d] += row[d];
            }
            float ss = 0;
            for (int d = 0; d < 128; ++d) { x[d] *= .25f; ss += x[d]*x[d]; }
            const float inv = 1.0f/sqrtf(ss/128 + ggml_qsa_epsilon(dst));
            for (int d = 0; d < 128; ++d) out[b*128+d] = (x[d]*inv)*gamma[d];
        }
    } else if (kind == 4) {
        // dst [d, np, nh, nq] F16; src[0] cache [d, nh, nk] Q8_0 rows; src[1] ids [ns, nq]
        const struct ggml_tensor * indices = dst->src[1];
        const int64_t d = dst->ne[0], np = dst->ne[1], nh = dst->ne[2], ns = indices->ne[0];
        ggml_fp16_t * result = dst->data;
        float row[4096];
        GGML_ASSERT(d*nh <= 4096);
        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const int32_t * sel = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);
            for (int64_t s = 0; s < np; ++s) {
                if (s < ns) {
                    dequantize_row_q8_0((const block_q8_0 *)((const char *)a->data + (size_t)sel[s]*a->nb[2]), row, d*nh);
                }
                for (int64_t h = 0; h < nh; ++h) {
                    ggml_fp16_t * o = result + d*(s + np*(h + nh*q));
                    for (int64_t i = 0; i < d; ++i) {
                        o[i] = s < ns ? ggml_fp32_to_fp16(row[h*d + i]) : 0;
                    }
                }
            }
        }
    } else if (kind == 3) {
        ggml_fp16_t * result = dst->data;
        const struct ggml_tensor * indices = dst->src[1];
        const ggml_fp16_t padding = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)a->data + q*a->nb[1]);
            const int32_t * row = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);
            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                result[q*dst->ne[0]+j] = j < indices->ne[0] ? m[row[j]] : padding;
            }
        }
    } else if (kind == 5) {
        // hyper-connection mix tail: gate, per-element multiply, mean over the hc streams
        const struct ggml_tensor * g = dst->src[1];
        const int64_t n_embd = dst->ne[0], nt = dst->ne[1], hc = a->ne[0]/n_embd;
        const float inv_hc = 1.0f/(float) hc;
        for (int64_t t = ith; t < nt; t += nth) {
            const float * x  = (const float *)((const char *)a->data + t*a->nb[1]);
            const float * gl = (const float *)((const char *)g->data + t*g->nb[1]);
            float * o = out + t*n_embd;
            for (int64_t i = 0; i < n_embd; ++i) {
                float s = 0.0f;
                for (int64_t h = 0; h < hc; ++h) {
                    const int64_t k = h*n_embd + i;
                    s += x[k] * (1.0f/(1.0f + expf(-gl[k])));
                }
                o[i] = s*inv_hc;
            }
        }
    } else if (kind == 6) {
        // hyper-connection combine: residual + block_out * 2*sigmoid(inject/hc)
        const struct ggml_tensor * bo = dst->src[1];
        const struct ggml_tensor * inj = dst->src[2];
        const int64_t n_embd = dst->ne[0], hc = dst->ne[1], nt = dst->ne[2];
        const float inv_hc = 1.0f/(float) hc;
        for (int64_t t = ith; t < nt; t += nth) {
            for (int64_t h = 0; h < hc; ++h) {
                const float lg = ((const float *)((const char *)inj->data + t*inj->nb[1]))[h]*inv_hc;
                const float w  = 2.0f*(1.0f/(1.0f + expf(-lg)));
                const float * r = (const float *)((const char *)a->data + t*a->nb[2] + h*a->nb[1]);
                const float * b = (const float *)((const char *)bo->data + t*bo->nb[1]);
                float * o = out + (t*hc + h)*n_embd;
                for (int64_t i = 0; i < n_embd; ++i) o[i] = r[i] + b[i]*w;
            }
        }
    } else {
        const struct ggml_tensor * mask = dst->src[2];
        for (int64_t q = ith; q < dst->ne[1]; q += nth) {
            const float * scores = (const float *)((const char *)a->data + q*a->nb[1]);
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)mask->data + q*mask->nb[1]);
            for (int64_t j = 0; j < dst->ne[0]; ++j) out[q*dst->ne[0]+j] = scores[ids[j]] + ggml_fp16_to_fp32(m[j]);
        }
    }
}

int ggml_qsa_kind(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) return 0;
    struct ggml_custom_op_params p; memcpy(&p,t->op_params,sizeof(p));
    return p.fun == qsa_cpu ? (int)(intptr_t)p.userdata : 0;
}
float ggml_qsa_epsilon(const struct ggml_tensor * t) {
    float eps; memcpy(&eps,(const char *)t->op_params+sizeof(struct ggml_custom_op_params),sizeof(eps));return eps;
}
struct ggml_tensor * ggml_qsa_pool_norm(struct ggml_context * ctx, struct ggml_tensor * raw, struct ggml_tensor * ids, struct ggml_tensor * gamma, float eps) {
    GGML_ASSERT(raw->type == GGML_TYPE_Q8_0 && raw->ne[0] == 128 && raw->ne[2] == 1 && raw->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[0]%4 == 0 && ids->ne[1] == 1);
    GGML_ASSERT(gamma->type == GGML_TYPE_F32 && ggml_nelements(gamma) == 128);
    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(gamma));
    struct ggml_tensor * args[] = {raw,ids,gamma};
    struct ggml_tensor * t = ggml_custom_4d(ctx,GGML_TYPE_F32,128,ids->ne[0]/4,1,1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)1);
    memcpy((char *)t->op_params+sizeof(struct ggml_custom_op_params),&eps,sizeof(eps));return t;
}
struct ggml_tensor * ggml_qsa_expand(struct ggml_context * ctx, struct ggml_tensor * score, struct ggml_tensor * ids, struct ggml_tensor * mask) {
    GGML_ASSERT(score->type == GGML_TYPE_F32 && score->ne[2] == 1 && score->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ggml_is_contiguous(ids));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->ne[0] == ids->ne[0] && mask->ne[1] >= score->ne[1]);
    struct ggml_tensor * args[] = {score,ids,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,ids->ne[0],score->ne[1],1,1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)2);
}

struct ggml_tensor * ggml_qsa_mask_select(struct ggml_context * ctx, struct ggml_tensor * mask, struct ggml_tensor * ids, int64_t padded) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(ggml_fp16_t));
    GGML_ASSERT(mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1 && mask->ne[1] >= ids->ne[1]);
    GGML_ASSERT(padded >= ids->ne[0] && padded%256 == 0);
    struct ggml_tensor * args[] = {mask,ids};
    return ggml_custom_4d(ctx,GGML_TYPE_F16,padded,1,1,ids->ne[1],args,2,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)3);
}

// Fused get_rows + pad + cast for the QSA compact path: one pass instead of three.
GGML_API struct ggml_tensor * ggml_qsa_gather_f16(struct ggml_context * ctx, struct ggml_tensor * cache, struct ggml_tensor * ids, int64_t padded) {
    GGML_ASSERT(cache->type == GGML_TYPE_Q8_0 && cache->ne[0] % 32 == 0 && cache->ne[3] == 1);
    GGML_ASSERT(cache->nb[1] == ggml_row_size(cache->type, cache->ne[0]) && cache->nb[2] == cache->ne[1]*cache->nb[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(padded >= ids->ne[0] && padded % 256 == 0);
    struct ggml_tensor * args[] = {cache, ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F16, cache->ne[0], padded, cache->ne[1], ids->ne[1], args, 2, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)4);
}

struct ggml_tensor * ggml_hc_mix_tail(struct ggml_context * ctx, struct ggml_tensor * xn, struct ggml_tensor * glogits, int hc) {
    GGML_ASSERT(xn->type == GGML_TYPE_F32 && glogits->type == GGML_TYPE_F32);
    GGML_ASSERT(xn->ne[0] == glogits->ne[0] && xn->ne[1] == glogits->ne[1] && xn->ne[2] == 1 && xn->ne[3] == 1 && glogits->ne[2] == 1);
    GGML_ASSERT(hc > 0 && xn->ne[0] % hc == 0 && xn->nb[0] == sizeof(float) && glogits->nb[0] == sizeof(float));
    struct ggml_tensor * args[] = {xn, glogits};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xn->ne[0]/hc, xn->ne[1], 1, 1, args, 2, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)5);
}
struct ggml_tensor * ggml_hc_combine(struct ggml_context * ctx, struct ggml_tensor * residual, struct ggml_tensor * block_out, struct ggml_tensor * inject, int hc) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32 && block_out->type == GGML_TYPE_F32 && inject->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->ne[1] == hc && residual->ne[3] == 1 && residual->ne[0] == block_out->ne[0] && residual->ne[2] == block_out->ne[1]);
    GGML_ASSERT(inject->ne[0] == hc && inject->ne[1] == residual->ne[2] && inject->ne[2] == 1);
    GGML_ASSERT(residual->nb[0] == sizeof(float) && block_out->nb[0] == sizeof(float) && inject->nb[0] == sizeof(float));
    struct ggml_tensor * args[] = {residual, block_out, inject};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, residual->ne[0], hc, residual->ne[2], 1, args, 3, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)6);
}
