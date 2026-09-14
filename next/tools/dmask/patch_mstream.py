# Multi-stream (one KV stream per slot) support for the fork's decode fast paths:
#   qsa_pool_norm / qsa_expand / qsa_mask_select / qsa_gather_f16 get a stream dimension, the device-side KQ mask gets a
#   per-mask-stream -> cache-stream map, and the qwen4exp gates drop their n_stream == 1 requirement.
import re, sys
S = '/home/douya/src/llama.cpp-flashnext-20260913/'
def rd(p): return open(S + p).read()
def wr(p, s): open(S + p, 'w').write(s)
def rep(s, old, new, path):
    assert s.count(old) == 1, (path, old[:80]); return s.replace(old, new, 1)
def replace_fn(s, start_marker, new_text, path):
    i = s.index(start_marker); j = s.index('\nstatic __global__', i + 1) if '\nstatic __global__' in s[i + 1:] else s.index('\nvoid ggml_cuda_op_qsa', i)
    return s[:i] + new_text.rstrip('\n') + '\n' + s[j:]

# ---------------------------------------------------------------- ggml-qsa.h
h = rd('ggml/src/ggml-qsa.h')
h = rep(h, 'GGML_API struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, int64_t n_kv, int64_t n_rows, enum ggml_type type);',
        'GGML_API struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, struct ggml_tensor * kvs, int64_t n_kv, int64_t n_tps, int64_t n_ns, enum ggml_type type);', 'ggml-qsa.h')
wr('ggml/src/ggml-qsa.h', h)

# ---------------------------------------------------------------- ggml-qsa.c
c = rd('ggml/src/ggml-qsa.c')
# CPU kind 1 (pool_norm) per stream
c = rep(c, '''        const float * gamma = dst->src[2]->data;
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
        }''', '''        const float * gamma = dst->src[2]->data;
        const struct ggml_tensor * idt = dst->src[1];
        const int64_t n_blocks = dst->ne[1];
        for (int64_t bs = ith; bs < n_blocks*dst->ne[2]; bs += nth) {
            const int64_t s = bs / n_blocks, b = bs % n_blocks;
            const int32_t * ids_s = (const int32_t *)((const char *)idt->data + s*idt->nb[1]);
            const char * raw_s = (const char *)a->data + s*a->nb[2];
            float x[128] = {0}, row[128];
            for (int j = 0; j < 4; ++j) {
                dequantize_row_q8_0((const block_q8_0 *)(raw_s + (size_t)ids_s[b*4+j]*a->nb[1]), row, 128);
                for (int d = 0; d < 128; ++d) x[d] += row[d];
            }
            float ss = 0;
            for (int d = 0; d < 128; ++d) { x[d] *= .25f; ss += x[d]*x[d]; }
            const float inv = 1.0f/sqrtf(ss/128 + ggml_qsa_epsilon(dst));
            for (int d = 0; d < 128; ++d) out[bs*128+d] = (x[d]*inv)*gamma[d];
        }''', 'qsa.c kind1')
# CPU kind 4 (gather) per stream
c = rep(c, '''        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const int32_t * sel = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);
            for (int64_t s = 0; s < np; ++s) {
                if (s < ns) {
                    dequantize_row_q8_0((const block_q8_0 *)((const char *)a->data + (size_t)sel[s]*a->nb[2]), row, d*nh);
                }''', '''        const int64_t n_str = a->ne[3], n_tps = dst->ne[3] / n_str;
        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const int32_t * sel = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);
            const char * base = (const char *)a->data + (q / n_tps)*a->nb[3];
            for (int64_t s = 0; s < np; ++s) {
                if (s < ns) {
                    dequantize_row_q8_0((const block_q8_0 *)(base + (size_t)sel[s]*a->nb[2]), row, d*nh);
                }''', 'qsa.c kind4')
# CPU kind 3 (mask_select) per stream
c = rep(c, '''        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)a->data + q*a->nb[1]);
            const int32_t * row = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);''', '''        const int64_t n_tps = a->ne[1];
        for (int64_t q = ith; q < dst->ne[3]; q += nth) {
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)a->data + (q / n_tps)*a->nb[3] + (q % n_tps)*a->nb[1]);
            const int32_t * row = (const int32_t *)((const char *)indices->data + q*indices->nb[1]);''', 'qsa.c kind3')
# CPU kind 9 (device mask) per stream
old9 = c[c.index('    } else if (kind == 9) {'):c.index('    } else if (kind == 5) {')]
new9 = '''    } else if (kind == 9) {
        const struct ggml_tensor * cpt = dst->src[0];
        const int32_t * cp = cpt->data; const int32_t * ps = dst->src[1]->data; const int32_t * kvs = dst->src[2]->data;
        const int64_t n_kv = dst->ne[0], n_tps = dst->ne[1], n_ns = dst->ne[3], cp_stride = cpt->nb[1]/sizeof(int32_t);
        for (int64_t st = ith; st < n_tps*n_ns; st += nth) {
            const int64_t s = st / n_tps, t = st % n_tps;
            const int32_t * cps = cp + (int64_t) kvs[s]*cp_stride;
            const int32_t p1 = ps[s*n_tps + t];
            for (int64_t c2 = 0; c2 < n_kv; ++c2) {
                const bool keep = cps[c2] >= 0 && cps[c2] <= p1;
                if (dst->type == GGML_TYPE_F16) ((ggml_fp16_t *) dst->data)[st*n_kv + c2] = ggml_fp32_to_fp16(keep ? 0.0f : -INFINITY);
                else ((float *) dst->data)[st*n_kv + c2] = keep ? 0.0f : -INFINITY;
            }
        }
'''
c = c.replace(old9, new9, 1)
# CPU kind 2 (expand, the else branch) per stream
c = rep(c, '''        const struct ggml_tensor * mask = dst->src[2];
        for (int64_t q = ith; q < dst->ne[1]; q += nth) {
            const float * scores = (const float *)((const char *)a->data + q*a->nb[1]);
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)mask->data + q*mask->nb[1]);
            for (int64_t j = 0; j < dst->ne[0]; ++j) out[q*dst->ne[0]+j] = scores[ids[j]] + ggml_fp16_to_fp32(m[j]);
        }''', '''        const struct ggml_tensor * mask = dst->src[2];
        const struct ggml_tensor * idt = dst->src[1];
        const int64_t n_tps = dst->ne[1];
        for (int64_t qs = ith; qs < n_tps*dst->ne[2]; qs += nth) {
            const int64_t s = qs / n_tps, q = qs % n_tps;
            const float * scores = (const float *)((const char *)a->data + s*a->nb[2] + q*a->nb[1]);
            const int32_t * ids_s = (const int32_t *)((const char *)idt->data + s*idt->nb[1]);
            const ggml_fp16_t * m = (const ggml_fp16_t *)((const char *)mask->data + s*mask->nb[3] + q*mask->nb[1]);
            for (int64_t j = 0; j < dst->ne[0]; ++j) out[qs*dst->ne[0]+j] = scores[ids_s[j]] + ggml_fp16_to_fp32(m[j]);
        }''', 'qsa.c kind2')
# constructors
c = rep(c, '''    GGML_ASSERT(raw->type == GGML_TYPE_Q8_0 && raw->ne[0] == 128 && raw->ne[2] == 1 && raw->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[0]%4 == 0 && ids->ne[1] == 1);''', '''    GGML_ASSERT(raw->type == GGML_TYPE_Q8_0 && raw->ne[0] == 128 && raw->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[0]%4 == 0 && ids->ne[1] == raw->ne[2]); // one ids row per stream''', 'ctor pool')
c = rep(c, '    struct ggml_tensor * t = ggml_custom_4d(ctx,GGML_TYPE_F32,128,ids->ne[0]/4,1,1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)1);',
        '    struct ggml_tensor * t = ggml_custom_4d(ctx,GGML_TYPE_F32,128,ids->ne[0]/4,ids->ne[1],1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)1);', 'ctor pool2')
c = rep(c, '''    GGML_ASSERT(score->type == GGML_TYPE_F32 && score->ne[2] == 1 && score->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ggml_is_contiguous(ids));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->ne[0] == ids->ne[0] && mask->ne[1] >= score->ne[1]);
    struct ggml_tensor * args[] = {score,ids,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,ids->ne[0],score->ne[1],1,1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)2);''', '''    GGML_ASSERT(score->type == GGML_TYPE_F32 && score->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[1] == score->ne[2] && ggml_is_contiguous(ids)); // [n_kv, n_stream]
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->ne[0] == ids->ne[0] && mask->ne[1] >= score->ne[1] && mask->ne[3] == score->ne[2]);
    struct ggml_tensor * args[] = {score,ids,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,ids->ne[0],score->ne[1],score->ne[2],1,args,3,qsa_cpu,GGML_N_TASKS_MAX,(void *)(intptr_t)2);''', 'ctor expand')
c = rep(c, '''    GGML_ASSERT(mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1 && mask->ne[1] >= ids->ne[1]);''', '''    GGML_ASSERT(mask->ne[2] == 1); // [n_kv, n_tps, 1, n_stream]
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1 && mask->ne[1]*mask->ne[3] >= ids->ne[1]); // ids rows = n_tps*n_stream''', 'ctor select')
c = rep(c, '    GGML_ASSERT(cache->type == GGML_TYPE_Q8_0 && cache->ne[0] % 32 == 0 && cache->ne[3] == 1);',
        '    GGML_ASSERT(cache->type == GGML_TYPE_Q8_0 && cache->ne[0] % 32 == 0); // [d, h, n_kv, n_stream]', 'ctor gather')
c = rep(c, '''struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, int64_t n_kv, int64_t n_rows, enum ggml_type type) {
    GGML_ASSERT(cell_pos->type == GGML_TYPE_I32 && pos->type == GGML_TYPE_I32 && cell_pos->ne[0] >= n_kv && n_rows >= pos->ne[0]);
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_F32);
    struct ggml_tensor * args[] = {cell_pos, pos};
    return ggml_custom_4d(ctx, type, n_kv, n_rows, 1, 1, args, 2, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)9);''', '''struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, struct ggml_tensor * kvs, int64_t n_kv, int64_t n_tps, int64_t n_ns, enum ggml_type type) {
    GGML_ASSERT(cell_pos->type == GGML_TYPE_I32 && pos->type == GGML_TYPE_I32 && kvs->type == GGML_TYPE_I32);
    GGML_ASSERT(cell_pos->ne[0] >= n_kv && pos->ne[0] == n_tps*n_ns && kvs->ne[0] == n_ns && cell_pos->nb[0] == sizeof(int32_t));
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_F32);
    struct ggml_tensor * args[] = {cell_pos, pos, kvs};
    return ggml_custom_4d(ctx, type, n_kv, n_tps, 1, n_ns, args, 3, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)9);''', 'ctor mask')
wr('ggml/src/ggml-qsa.c', c)

# ---------------------------------------------------------------- qsa.cu
u = rd('ggml/src/ggml-cuda/qsa.cu')
u = replace_fn(u, 'static __global__ void qsa_pool_norm(', '''static __global__ void qsa_pool_norm(const char * raw, const int32_t * ids, const float * gamma,
        float * out, size_t row_stride, size_t raw_stream_stride, size_t ids_stream_stride, int64_t n_blocks, float eps) {
    const int64_t b = blockIdx.x;
    const int64_t s = blockIdx.y;
    const int c = threadIdx.x;
    const int32_t * ids_s = reinterpret_cast<const int32_t *>(reinterpret_cast<const char *>(ids) + s*ids_stream_stride);
    const char * raw_s = raw + s*raw_stream_stride;
    float x = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const char * block = raw_s + size_t(ids_s[b*4+j])*row_stride + (c/32)*34;
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
    out[(s*n_blocks + b)*128+c] = (x*inv)*gamma[c];
}
''', 'pool_norm')
u = replace_fn(u, 'static __global__ void qsa_expand(', '''static __global__ void qsa_expand(const char * score, const int32_t * ids, const char * mask,
        float * out, int64_t n, size_t score_stride, size_t mask_stride, size_t score_stream_stride, size_t ids_stream_stride, size_t mask_stream_stride, int64_t n_tps) {
    const int64_t j = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (j < n) {
        const float * sc = reinterpret_cast<const float *>(score + s*score_stream_stride + q*score_stride);
        const int32_t * ids_s = reinterpret_cast<const int32_t *>(reinterpret_cast<const char *>(ids) + s*ids_stream_stride);
        const half * m = reinterpret_cast<const half *>(mask + s*mask_stream_stride + q*mask_stride);
        out[(s*n_tps + q)*n+j] = sc[ids_s[j]] + __half2float(m[j]);
    }
}
''', 'expand')
u = replace_fn(u, 'static __global__ void qsa_mask_select(', '''static __global__ void qsa_mask_select(const char * mask, const char * ids, half * out,
        int64_t ns, int64_t np, size_t mask_stride, size_t ids_stride, size_t mask_stream_stride, int64_t n_tps) {
    const int64_t j = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q = blockIdx.y;
    if (j < np) {
        const half * m = reinterpret_cast<const half *>(mask + (q / n_tps)*mask_stream_stride + (q % n_tps)*mask_stride);
        const int32_t * row = reinterpret_cast<const int32_t *>(ids + q*ids_stride);
        out[q*np+j] = j < ns ? m[row[j]] : __float2half(-INFINITY);
    }
}
''', 'mask_select')
u = replace_fn(u, 'static __global__ void qsa_gather_f16(', '''static __global__ void qsa_gather_f16(const char * __restrict__ cache, const char * __restrict__ ids, half * __restrict__ out,
        int64_t d, int64_t nh, int64_t ns, int64_t np, size_t cell_stride, size_t ids_stride, size_t cache_stream_stride, int64_t n_tps) {
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
    const char * row = cache + (q / n_tps)*cache_stream_stride + size_t(cell)*cell_stride;
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
''', 'gather')
u = replace_fn(u, 'static __global__ void kq_mask_dev(', '''static __global__ void kq_mask_dev(const int32_t * __restrict__ cell_pos, const int32_t * __restrict__ pos, const int32_t * __restrict__ kvs, T * __restrict__ out, int64_t n_kv, int64_t n_tps, int64_t cp_stride) {
    const int64_t c = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (c >= n_kv) return;
    const int cp = cell_pos[int64_t(kvs[s])*cp_stride + c];
    const bool keep = cp >= 0 && cp <= pos[s*n_tps + t];
    out[(s*n_tps + t)*n_kv + c] = keep ? T(0.0f) : T(-INFINITY);
}
''', 'kq_mask_dev')
# the template line before kq_mask_dev must stay: replace_fn kept "template <typename T>" above? check
assert 'template <typename T>\nstatic __global__ void kq_mask_dev(' in u, 'template header lost'
u = rep(u, '''        const auto * cp = dst->src[0]; const auto * ps = dst->src[1];
        const int64_t n_kv = dst->ne[0], n_rows = dst->ne[1], n_tok = ps->ne[0];
        const dim3 grid((unsigned)((n_kv + 255)/256), (unsigned) n_rows);
        if (dst->type == GGML_TYPE_F16) {
            kq_mask_dev<half><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (half *) dst->data, n_kv, n_tok);
        } else {
            kq_mask_dev<float><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (float *) dst->data, n_kv, n_tok);
        }''', '''        const auto * cp = dst->src[0]; const auto * ps = dst->src[1]; const auto * kvs = dst->src[2];
        const int64_t n_kv = dst->ne[0], n_tps = dst->ne[1], n_ns = dst->ne[3], cp_stride = cp->nb[1]/sizeof(int32_t);
        const dim3 grid((unsigned)((n_kv + 255)/256), (unsigned) n_tps, (unsigned) n_ns);
        if (dst->type == GGML_TYPE_F16) {
            kq_mask_dev<half><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (const int32_t *) kvs->data, (half *) dst->data, n_kv, n_tps, cp_stride);
        } else {
            kq_mask_dev<float><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (const int32_t *) kvs->data, (float *) dst->data, n_kv, n_tps, cp_stride);
        }''', 'dispatch 9')
u = rep(u, '''        qsa_pool_norm<<<dst->ne[1],128,0,ctx.stream()>>>((const char *)a->data,(const int32_t *)ids->data,
            (const float *)c->data,(float *)dst->data,a->nb[1],ggml_qsa_epsilon(dst));''', '''        qsa_pool_norm<<<dim3(dst->ne[1],dst->ne[2]),128,0,ctx.stream()>>>((const char *)a->data,(const int32_t *)ids->data,
            (const float *)c->data,(float *)dst->data,a->nb[1],a->nb[2],ids->nb[1],dst->ne[1],ggml_qsa_epsilon(dst));''', 'dispatch 1')
u = rep(u, '''        qsa_gather_f16<<<dim3(dst->ne[1],dst->ne[3]),128,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,dst->ne[0],dst->ne[2],ids->ne[0],dst->ne[1],a->nb[2],ids->nb[1]);''', '''        qsa_gather_f16<<<dim3(dst->ne[1],dst->ne[3]),128,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,dst->ne[0],dst->ne[2],ids->ne[0],dst->ne[1],a->nb[2],ids->nb[1],a->nb[3],dst->ne[3]/a->ne[3]);''', 'dispatch 4')
u = rep(u, '''        qsa_mask_select<<<dim3((dst->ne[0]+255)/256,dst->ne[3]),256,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,ids->ne[0],dst->ne[0],a->nb[1],ids->nb[1]);''', '''        qsa_mask_select<<<dim3((dst->ne[0]+255)/256,dst->ne[3]),256,0,ctx.stream()>>>((const char *)a->data,
            (const char *)ids->data,(half *)dst->data,ids->ne[0],dst->ne[0],a->nb[1],ids->nb[1],a->nb[3],a->ne[1]);''', 'dispatch 3')
u = rep(u, '''        qsa_expand<<<dim3((dst->ne[0]+255)/256,dst->ne[1]),256,0,ctx.stream()>>>((const char *)a->data,
            (const int32_t *)ids->data,(const char *)c->data,(float *)dst->data,dst->ne[0],a->nb[1],c->nb[1]);''', '''        qsa_expand<<<dim3((dst->ne[0]+255)/256,dst->ne[1],dst->ne[2]),256,0,ctx.stream()>>>((const char *)a->data,
            (const int32_t *)ids->data,(const char *)c->data,(float *)dst->data,dst->ne[0],a->nb[1],c->nb[1],a->nb[2],ids->nb[1],c->nb[3],dst->ne[1]);''', 'dispatch 2')
wr('ggml/src/ggml-cuda/qsa.cu', u)

# ---------------------------------------------------------------- qwen4exp.cpp gates + compact
q = rd('src/models/qwen4exp.cpp')
q = rep(q, '    if (getenv("NEXT_QSA_OPT") && n_stream == 1 && n_tps <= 8 && r == 4 && idx_dim == 128 &&', '    if (getenv("NEXT_QSA_OPT") && n_tps <= 8 && r == 4 && idx_dim == 128 && // NEXT: stream-aware kernels', 'gate pool')
q = rep(q, '    if (getenv("NEXT_QSA_OPT") && n_stream == 1 && n_tps <= 8 && blk_bias &&', '    if (getenv("NEXT_QSA_OPT") && n_tps <= 8 && blk_bias && // NEXT: stream-aware kernel', 'gate expand')
q = rep(q, '    if (getenv("NEXT_QSA_OPT") && n_tokens <= 8 && kq_mask->ne[3] == 1 &&', '    if (getenv("NEXT_QSA_OPT") && n_tokens/kq_mask->ne[3] <= 8 && // NEXT: <= 8 tokens per stream', 'gate compact')
q = rep(q, '        if (k->ne[0] == 256 && v->ne[0] == 256 && k->ne[1] == 2 && v->ne[1] == 2 && k->ne[3] == 1 && v->ne[3] == 1 &&', '        if (k->ne[0] == 256 && v->ne[0] == 256 && k->ne[1] == 2 && v->ne[1] == 2 && k->ne[3] == kq_mask->ne[3] && v->ne[3] == kq_mask->ne[3] &&', 'gate compact kv')
q = rep(q, '''    ggml_tensor * kc = nullptr;
    ggml_tensor * vc = nullptr;
    if (!next_opt_flag("qsa_gather_unfused")) {  // NEXT: fused gather+dequant+cast (see qsa.cu) is the default now
        auto * ids2 = ggml_reshape_2d(ctx, ggml_cont(ctx, indices), ns, nq);
        kc = ggml_qsa_gather_f16(ctx, k, ids2, np);
        vc = ggml_qsa_gather_f16(ctx, v, ids2, np);
    } else {
        kc = gather(k);
        vc = gather(v);
    }
    // Keep the original mask pointer so the scheduler reuses its device copy.
    auto * selected_mask = ggml_qsa_mask_select(ctx, mask, indices, np);''', '''    ggml_tensor * kc = nullptr;
    ggml_tensor * vc = nullptr;
    auto * ids2 = ggml_reshape_2d(ctx, ggml_cont(ctx, indices), ns, nq); // rows = n_tps*n_stream, stream-major
    if (!next_opt_flag("qsa_gather_unfused") || k->ne[3] > 1) {  // NEXT: fused gather+dequant+cast (see qsa.cu) is the default; the only path for several streams
        kc = ggml_qsa_gather_f16(ctx, k, ids2, np);
        vc = ggml_qsa_gather_f16(ctx, v, ids2, np);
    } else {
        kc = gather(k);
        vc = gather(v);
    }
    // Keep the original mask pointer so the scheduler reuses its device copy.
    auto * selected_mask = ggml_qsa_mask_select(ctx, mask, ids2, np);''', 'compact ids2')
wr('src/models/qwen4exp.cpp', q)

# ---------------------------------------------------------------- llama-kv-cache.h / .cpp
kh = rd('src/llama-kv-cache.h')
kh = rep(kh, '    std::vector<int32_t> get_cell_pos_host() const; // NEXT: debug (NEXT_DEVICE_MASK_CHECK)\n',
         '    std::vector<int32_t> get_cell_pos_host() const; // NEXT: debug (NEXT_DEVICE_MASK_CHECK)\n    uint32_t get_stream_of_seq(llama_seq_id seq_id) const { return seq_to_stream.at(seq_id); }\n', 'kv.h 1')
kh = rep(kh, '    std::vector<int32_t> get_cell_pos_host() const { return kv->get_cell_pos_host(); }\n',
         '    std::vector<int32_t> get_cell_pos_host() const { return kv->get_cell_pos_host(); }\n    uint32_t get_stream_of_seq(llama_seq_id seq_id) const { return kv->get_stream_of_seq(seq_id); }\n', 'kv.h 2')
wr('src/llama-kv-cache.h', kh)
kc = rd('src/llama-kv-cache.cpp')
kc = rep(kc, '    const bool ok = n_stream == 1 && n_seq_max == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi;',
         '    // one stream per sequence (or a single sequence): every cell of a stream belongs to its sequence\n    const bool ok = ((n_stream == 1 && n_seq_max == 1) || n_stream == n_seq_max) && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi;', 'device_mask_ok')
wr('src/llama-kv-cache.cpp', kc)

# ---------------------------------------------------------------- llama-graph.h / .cpp
gh = rd('src/llama-graph.h')
gh = rep(gh, '    ggml_tensor * self_pos = nullptr;\n', '    ggml_tensor * self_pos = nullptr;\n    ggml_tensor * self_kvs = nullptr; // I32 [n_streams of the ubatch]: mask stream -> KV cache stream\n', 'graph.h kvs')
wr('src/llama-graph.h', gh)
g = rd('src/llama-graph.cpp')
# build
g = rep(g, '''        if (mctx_cur->device_mask_ok(cparams.causal_attn) && (!ubatch.is_pos_2d() || device_mask_level >= 2) && (cparams.kv_unified || ubatch.n_seqs_unq == 1)) {
            // NEXT: GPU-generated causal masks, one per device holding KV layers (no host fill, no per-step upload)
            inp->self_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
            ggml_set_input(inp->self_pos);
            ggml_set_name(inp->self_pos, "attn_inp_mask_pos");
            const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
            const int64_t n_kv = mctx_cur->get_n_kv();
            for (const auto & e : mctx_cur->get_cell_pos_list()) {
                ggml_tensor * cpv = ggml_view_1d(ctx0, e.second, n_kv, 0);
                ggml_tensor * m = ggml_kq_mask_dev(ctx0, cpv, inp->self_pos, n_kv, ubatch.n_tokens, type);''', '''        if (mctx_cur->device_mask_ok(cparams.causal_attn) && (!ubatch.is_pos_2d() || device_mask_level >= 2) && (!cparams.kv_unified || ubatch.n_seqs_unq == 1)) {
            // NEXT: GPU-generated causal masks, one per device holding KV layers (no host fill, no per-step upload);
            // with one KV stream per sequence the mask carries one plane per stream of the ubatch
            const int64_t n_ns  = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
            const int64_t n_tps = ubatch.n_tokens / n_ns;
            inp->self_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
            ggml_set_input(inp->self_pos);
            ggml_set_name(inp->self_pos, "attn_inp_mask_pos");
            inp->self_kvs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ns);
            ggml_set_input(inp->self_kvs);
            ggml_set_name(inp->self_kvs, "attn_inp_mask_kvs");
            const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
            const int64_t n_kv = mctx_cur->get_n_kv();
            for (const auto & e : mctx_cur->get_cell_pos_list()) {
                ggml_tensor * m = ggml_kq_mask_dev(ctx0, e.second, inp->self_pos, inp->self_kvs, n_kv, n_tps, n_ns, type);''', 'graph build')
# set_input helper: fill kvs (attn_kv input)
g = rep(g, '''    if (self_pos && self_pos->buffer) {
        ggml_backend_tensor_set(self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        mctx->upload_cell_pos();
    }''', '''    if (self_pos && self_pos->buffer) {
        ggml_backend_tensor_set(self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        if (self_kvs && self_kvs->buffer) {
            std::vector<int32_t> kvs(self_kvs->ne[0], 0);
            for (int64_t s = 0; s < self_kvs->ne[0]; ++s) {
                kvs[s] = self_kvs->ne[0] == 1 && cparams.kv_unified ? 0 : (int32_t) mctx->get_stream_of_seq(ubatch->seq_id_unq[s]);
            }
            ggml_backend_tensor_set(self_kvs, kvs.data(), 0, kvs.size()*sizeof(int32_t));
        }
        mctx->upload_cell_pos();
    }''', 'set_input attn')
g = rep(g, '''    if (inp_attn->self_pos && inp_attn->self_pos->buffer) {
        ggml_backend_tensor_set(inp_attn->self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        mctx->get_attn()->upload_cell_pos();
    }''', '''    if (inp_attn->self_pos && inp_attn->self_pos->buffer) {
        ggml_backend_tensor_set(inp_attn->self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        if (inp_attn->self_kvs && inp_attn->self_kvs->buffer) {
            std::vector<int32_t> kvs(inp_attn->self_kvs->ne[0], 0);
            for (int64_t s = 0; s < inp_attn->self_kvs->ne[0]; ++s) {
                kvs[s] = inp_attn->self_kvs->ne[0] == 1 && cparams.kv_unified ? 0 : (int32_t) mctx->get_attn()->get_stream_of_seq(ubatch->seq_id_unq[s]);
            }
            ggml_backend_tensor_set(inp_attn->self_kvs, kvs.data(), 0, kvs.size()*sizeof(int32_t));
        }
        mctx->get_attn()->upload_cell_pos();
    }''', 'set_input hybrid')
# can_reuse: number of streams must match
g = rep(g, '    res &= self_pos == nullptr || self_pos->ne[0] == params.ubatch.n_tokens;',
        '    res &= self_pos == nullptr || self_pos->ne[0] == params.ubatch.n_tokens;\n    res &= self_kvs == nullptr || self_kvs->ne[0] == (params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq);', 'can_reuse attn')
g = rep(g, '    res &= inp_attn->self_pos == nullptr || inp_attn->self_pos->ne[0] == params.ubatch.n_tokens;',
        '    res &= inp_attn->self_pos == nullptr || inp_attn->self_pos->ne[0] == params.ubatch.n_tokens;\n    res &= inp_attn->self_kvs == nullptr || inp_attn->self_kvs->ne[0] == (params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq);', 'can_reuse hybrid')
# DM_CHECK readback: per stream
old_chk = g[g.index('int llm_graph_input_attn_kv::check_dev_masks('):g.index('ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_l(int il) const {')]
new_chk = '''int llm_graph_input_attn_kv::check_dev_masks(const llama_kv_cache_context * kvctx, const llama_ubatch * ubatch, int verbose) const {
    int bad_total = 0;
    if (self_kq_mask_dev.empty() || !self_pos) {
        return 0;
    }
    const std::vector<int32_t> cp = kvctx->get_cell_pos_host();
    const uint32_t n_tokens = ubatch->n_tokens;
    for (const auto & e : self_kq_mask_dev) {
        ggml_tensor * m = e.second;
        const int64_t n_kv = m->ne[0], n_tps = m->ne[1], n_ns = m->ne[3];
        const int64_t kv_size = (int64_t) cp.size() / std::max<int64_t>(1, (int64_t) (self_kvs ? kvctx->get_stream_of_seq(0) + 1 : 1)); // placeholder, recomputed below
        (void) kv_size;
        std::vector<uint8_t> buf(ggml_nbytes(m));
        ggml_backend_tensor_get(m, buf.data(), 0, buf.size());
        std::vector<int32_t> kvs(n_ns, 0);
        if (self_kvs && self_kvs->buffer) {
            ggml_backend_tensor_get(self_kvs, kvs.data(), 0, kvs.size()*sizeof(int32_t));
        }
        // the host table is [kv_size x n_stream]; kv_size = the stride of the device table
        const int64_t stride = e.second->src[0]->nb[1] / sizeof(int32_t);
        int bad = 0;
        for (int64_t s = 0; s < n_ns; ++s) {
            for (int64_t t = 0; t < n_tps; ++t) {
                const int64_t i = s*n_tps + t;
                const llama_pos p1 = i < (int64_t) n_tokens ? ubatch->pos[i] : -1;
                for (int64_t c = 0; c < n_kv; ++c) {
                    const int64_t ci = (int64_t) kvs[s]*stride + c;
                    const bool keep_exp = i < (int64_t) n_tokens && ci < (int64_t) cp.size() && cp[ci] >= 0 && cp[ci] <= p1;
                    float got;
                    if (m->type == GGML_TYPE_F16) {
                        got = ggml_fp16_to_fp32(((const ggml_fp16_t *) buf.data())[i*n_kv + c]);
                    } else {
                        got = ((const float *) buf.data())[i*n_kv + c];
                    }
                    const bool keep_got = got == 0.0f;
                    if (keep_got != keep_exp || (!keep_got && !(got == -INFINITY))) {
                        if (bad < 5 && verbose > 1) {
                            fprintf(stderr, "DM_CHECK mismatch %s: stream %lld row %lld cell %lld cell_pos %d tok_pos %d got %g expected %s\\n", ggml_backend_buft_name(e.first), (long long) s, (long long) t, (long long) c, ci < (int64_t) cp.size() ? cp[ci] : -2, (int) p1, got, keep_exp ? "0" : "-inf");
                        }
                        bad++;
                    }
                }
            }
        }
        if (bad || verbose > 1) {
            fprintf(stderr, "DM_CHECK %s: n_kv=%lld n_tps=%lld n_ns=%lld n_tokens=%u pos[0]=%d mismatches=%d\\n", ggml_backend_buft_name(e.first), (long long) n_kv, (long long) n_tps, (long long) n_ns, n_tokens, n_tokens ? (int) ubatch->pos[0] : -1, bad);
        }
        bad_total += bad;
    }
    return bad_total;
}

'''
g = g.replace(old_chk, new_chk, 1)
wr('src/llama-graph.cpp', g)
print('multi-stream patch applied')
