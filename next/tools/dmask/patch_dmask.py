# NEXT_DEVICE_MASK: GPU-generated causal KQ masks from device-resident cell positions (incrementally maintained)
import re
F = '/home/douya/src/llama.cpp-flashnext-20260913'
def rd(p): return open(F + p).read()
def wr(p, s): open(F + p, 'w').write(s)
def rep(s, old, new, count=1, path=''):
    c = s.count(old); assert c == count, (path, old[:70], c); return s.replace(old, new)

# ---------------------------------------------------------------- ggml-qsa.h
s = rd('/ggml/src/ggml-qsa.h')
if 'ggml_kq_mask_dev' not in s:
    s = rep(s, 'GGML_API int ggml_qsa_kind(const struct ggml_tensor * t);',
        '//   device-side causal KQ mask (kind 9): out[c,t] = (cell_pos[c] >= 0 && cell_pos[c] <= pos[t]) ? 0 : -inf\nGGML_API struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, int64_t n_kv, int64_t n_rows, enum ggml_type type);\nGGML_API int ggml_qsa_kind(const struct ggml_tensor * t);', path='qsa.h')
    wr('/ggml/src/ggml-qsa.h', s); print('ggml-qsa.h patched')

# ---------------------------------------------------------------- ggml-qsa.c
s = rd('/ggml/src/ggml-qsa.c')
if 'kind == 9' not in s:
    s = rep(s, '    } else if (kind == 5) {', '''    } else if (kind == 9) {
        const int32_t * cp = dst->src[0]->data; const int32_t * ps = dst->src[1]->data;
        const int64_t n_kv = dst->ne[0], n_rows = dst->ne[1], n_tok = dst->src[1]->ne[0];
        for (int64_t t = ith; t < n_rows; t += nth) {
            for (int64_t c = 0; c < n_kv; ++c) {
                const bool keep = t < n_tok && cp[c] >= 0 && cp[c] <= ps[t];
                if (dst->type == GGML_TYPE_F16) ((ggml_fp16_t *) dst->data)[t*n_kv + c] = ggml_fp32_to_fp16(keep ? 0.0f : -INFINITY);
                else ((float *) dst->data)[t*n_kv + c] = keep ? 0.0f : -INFINITY;
            }
        }
    } else if (kind == 5) {''', path='qsa.c dispatch')
    s = s.rstrip('\n') + '''
struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, int64_t n_kv, int64_t n_rows, enum ggml_type type) {
    GGML_ASSERT(cell_pos->type == GGML_TYPE_I32 && pos->type == GGML_TYPE_I32 && cell_pos->ne[0] >= n_kv && n_rows >= pos->ne[0]);
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_F32);
    struct ggml_tensor * args[] = {cell_pos, pos};
    return ggml_custom_4d(ctx, type, n_kv, n_rows, 1, 1, args, 2, qsa_cpu, GGML_N_TASKS_MAX, (void *)(intptr_t)9);
}
'''
    wr('/ggml/src/ggml-qsa.c', s); print('ggml-qsa.c patched')

# ---------------------------------------------------------------- qsa.cu
s = rd('/ggml/src/ggml-cuda/qsa.cu')
if 'kq_mask_dev' not in s:
    s = rep(s, 'void ggml_cuda_op_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {', '''// NEXT: causal KQ mask from device-resident cell positions: rows >= n_tok (padding) are fully masked
template <typename T>
static __global__ void kq_mask_dev(const int32_t * __restrict__ cell_pos, const int32_t * __restrict__ pos, T * __restrict__ out, int64_t n_kv, int64_t n_tok) {
    const int64_t c = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (c >= n_kv) return;
    const int cp = cell_pos[c];
    const bool keep = t < n_tok && cp >= 0 && cp <= pos[t];
    out[t*n_kv + c] = keep ? T(0.0f) : T(-INFINITY);
}
void ggml_cuda_op_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_qsa_kind(dst) == 9) {
        const auto * cp = dst->src[0]; const auto * ps = dst->src[1];
        const int64_t n_kv = dst->ne[0], n_rows = dst->ne[1], n_tok = ps->ne[0];
        const dim3 grid((unsigned)((n_kv + 255)/256), (unsigned) n_rows);
        if (dst->type == GGML_TYPE_F16) {
            kq_mask_dev<half><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (half *) dst->data, n_kv, n_tok);
        } else {
            kq_mask_dev<float><<<grid, 256, 0, ctx.stream()>>>((const int32_t *) cp->data, (const int32_t *) ps->data, (float *) dst->data, n_kv, n_tok);
        }
        return;
    }''', path='qsa.cu')
    wr('/ggml/src/ggml-cuda/qsa.cu', s); print('qsa.cu patched')

# ---------------------------------------------------------------- llama-kv-cache.h
s = rd('/src/llama-kv-cache.h')
if 'cell_pos_dev' not in s:
    s = rep(s, '    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;\n', '''    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    // NEXT: device-resident cell positions (I32 [kv_size, n_stream], one tensor per KV buffer type / device) so the
    // causal KQ mask can be generated on the GPU instead of being filled on the host and uploaded every step.
    // apply_ubatch() writes only the cells of the ubatch; structural changes mark it dirty -> full re-upload.
    bool device_mask_ok(bool causal_attn) const;
    ggml_tensor * get_cell_pos(int32_t il) const;
    const std::vector<std::pair<ggml_backend_buffer_type_t, ggml_tensor *>> & get_cell_pos_list() const { return cell_pos_dev; }
    void upload_cell_pos() const;
''', count=2, path='kv.h decl')  # both the cache and the context class declare set_input_kq_mask -> patched twice; fix the context copy below
    # the context class must forward instead of declaring the cache internals: repair the second insertion
    ctx_start = s.index('class llama_kv_cache_context')
    head, tail = s[:ctx_start], s[ctx_start:]
    tail = tail.replace('''
    // NEXT: device-resident cell positions (I32 [kv_size, n_stream], one tensor per KV buffer type / device) so the
    // causal KQ mask can be generated on the GPU instead of being filled on the host and uploaded every step.
    // apply_ubatch() writes only the cells of the ubatch; structural changes mark it dirty -> full re-upload.
    bool device_mask_ok(bool causal_attn) const;
    ggml_tensor * get_cell_pos(int32_t il) const;
    const std::vector<std::pair<ggml_backend_buffer_type_t, ggml_tensor *>> & get_cell_pos_list() const { return cell_pos_dev; }
    void upload_cell_pos() const;
''', '''
    // NEXT: GPU-generated KQ mask support (forwarders)
    bool device_mask_ok(bool causal_attn) const { return kv->device_mask_ok(causal_attn); }
    ggml_tensor * get_cell_pos(int32_t il) const { return kv->get_cell_pos(il); }
    const std::vector<std::pair<ggml_backend_buffer_type_t, ggml_tensor *>> & get_cell_pos_list() const { return kv->get_cell_pos_list(); }
    void upload_cell_pos() const { kv->upload_cell_pos(); }
''', 1)
    s = head + tail
    s = rep(s, '    std::vector<kv_layer> layers;\n', '''    std::vector<kv_layer> layers;

    // NEXT: see device_mask_ok()
    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_tensor *>> cell_pos_dev;
    mutable bool cell_pos_dirty = true;
    void update_cell_pos(const slot_info & sinfo, const llama_ubatch & ubatch);
''', path='kv.h members')
    wr('/src/llama-kv-cache.h', s); print('llama-kv-cache.h patched')

# ---------------------------------------------------------------- llama-kv-cache.cpp
s = rd('/src/llama-kv-cache.cpp')
if 'cell_pos_dev' not in s:
    # 1. create the tensors before the buffers are allocated
    s = rep(s, '    for (auto & [buft, ctx] : ctx_map) {\n        ggml_backend_buffer_t buf;\n', '''    // NEXT: one I32 [kv_size, n_stream] cell-position tensor per buffer type (device), see device_mask_ok()
    for (auto & [buft, ctx] : ctx_map) {
        ggml_tensor * cp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, kv_size, n_stream);
        ggml_format_name(cp, "cache_cellpos_%s", ggml_backend_buft_name(buft));
        cell_pos_dev.emplace_back(buft, cp);
    }
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
''', path='kv.cpp ctor')
    # 2. incremental update at the end of apply_ubatch: find the function and its closing brace
    a = s.index('void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {')
    b = s.index('\n}\n', a)
    s = s[:b] + '\n\n    update_cell_pos(sinfo, ubatch);' + s[b:]
    # 3. mark dirty in the mutators
    for sig in ['void llama_kv_cache::clear(bool data) {', 'bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {',
                'void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {',
                'void llama_kv_cache::seq_keep(llama_seq_id seq_id) {', 'void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {',
                'void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {']:
        s = rep(s, sig + '\n', sig + '\n    cell_pos_dirty = true;\n', path='kv.cpp mutator ' + sig[:40])
    for sig in ['void llama_kv_cache::state_read_sinfo(', 'bool llama_kv_cache::state_read_meta(']:
        i = s.index(sig); j = s.index('{\n', i) + 2
        s = s[:j] + '    cell_pos_dirty = true;\n' + s[j:]
    # 4. the new methods
    s = rep(s, 'void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {', '''bool llama_kv_cache::device_mask_ok(bool causal_attn) const {
    static const bool enabled = [] { const char * e = getenv("NEXT_DEVICE_MASK"); return e != nullptr && atoi(e) != 0; }();
    if (!enabled || !causal_attn || cell_pos_dev.empty()) {
        return false;
    }
    // the GPU rule is "cell used and cell.pos <= token.pos": one stream, one sequence, no SWA, no ALiBi
    return n_stream == 1 && n_seq_max == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi;
}

ggml_tensor * llama_kv_cache::get_cell_pos(int32_t il) const {
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    const auto & layer = layers[it->second];
    if (!layer.k || !layer.k->buffer) {
        return nullptr;
    }
    const auto buft = ggml_backend_buffer_get_type(layer.k->buffer);
    for (const auto & e : cell_pos_dev) {
        if (e.first == buft) {
            return e.second;
        }
    }
    return nullptr;
}

void llama_kv_cache::upload_cell_pos() const {
    if (!cell_pos_dirty) {
        return;
    }
    const uint32_t kv_size = v_cells[0].size();
    std::vector<int32_t> host((size_t) kv_size * n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t i = 0; i < kv_size; ++i) {
            host[(size_t) s * kv_size + i] = cells.is_empty(i) ? -1 : cells.pos_get(i);
        }
    }
    for (const auto & e : cell_pos_dev) {
        if (e.second->data) {
            ggml_backend_tensor_set(e.second, host.data(), 0, host.size() * sizeof(int32_t));
        }
    }
    cell_pos_dirty = false;
}

void llama_kv_cache::update_cell_pos(const slot_info & sinfo, const llama_ubatch & ubatch) {
    if (cell_pos_dirty || cell_pos_dev.empty() || cell_pos_dev[0].second->data == nullptr) {
        return; // the next upload_cell_pos() rewrites everything
    }
    const uint32_t kv_size = v_cells[0].size();
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];
        const auto & idxs = sinfo.idxs[s];
        // contiguous runs of cells -> one tensor_set per run
        size_t i = 0;
        while (i < idxs.size()) {
            size_t j = i + 1;
            while (j < idxs.size() && idxs[j] == idxs[j - 1] + 1) { j++; }
            std::vector<int32_t> vals(j - i);
            for (size_t k = i; k < j; ++k) { vals[k - i] = ubatch.pos[s * sinfo.size() + k]; }
            for (const auto & e : cell_pos_dev) {
                ggml_backend_tensor_set(e.second, vals.data(), ((size_t) strm * kv_size + idxs[i]) * sizeof(int32_t), vals.size() * sizeof(int32_t));
            }
            i = j;
        }
    }
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {''', path='kv.cpp methods')
    wr('/src/llama-kv-cache.cpp', s); print('llama-kv-cache.cpp patched')

# ---------------------------------------------------------------- llama-graph.h
s = rd('/src/llama-graph.h')
if 'self_kq_mask_dev' not in s:
    cls = s.index('class llm_graph_input_attn_kv : public llm_graph_input_i {')
    anchor = '    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }\n'
    i = s.index(anchor, cls)
    s = s[:i] + anchor + '''    // NEXT: the mask that lives on the device of layer il (GPU-generated masks), else the shared one
    ggml_tensor * get_kq_mask_l(int il) const;

    // NEXT: GPU-generated masks, one per KV buffer type (device); self_pos = I32 [n_batch] token positions feeding them
    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_tensor *>> self_kq_mask_dev;
    ggml_tensor * self_pos = nullptr;
''' + s[i + len(anchor):]
    wr('/src/llama-graph.h', s); print('llama-graph.h patched')

# ---------------------------------------------------------------- llama-graph.cpp
s = rd('/src/llama-graph.cpp')
if 'self_kq_mask_dev' not in s:
    m = re.search(r'([ \t]*)inp->self_kq_mask = build_attn_inp_kq_mask\(ctx0, mctx_cur, ubatch, cparams\);\s*\n[ \t]*inp->self_kq_mask_cnv = inp->self_kq_mask;\n', s)
    assert m, 'build_attn_inp_kv_impl mask lines not found'
    ind = m.group(1)
    block = (ind + 'if (mctx_cur->device_mask_ok(cparams.causal_attn) && !ubatch.is_pos_2d() && (cparams.kv_unified || ubatch.n_seqs_unq == 1)) {\n'
        + ind + '    // NEXT: GPU-generated causal masks, one per device holding KV layers (no host fill, no per-step upload)\n'
        + ind + '    inp->self_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);\n'
        + ind + '    ggml_set_input(inp->self_pos);\n'
        + ind + '    ggml_set_name(inp->self_pos, "attn_inp_mask_pos");\n'
        + ind + '    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;\n'
        + ind + '    const int64_t n_kv = mctx_cur->get_n_kv();\n'
        + ind + '    for (const auto & e : mctx_cur->get_cell_pos_list()) {\n'
        + ind + '        ggml_tensor * cpv = ggml_view_1d(ctx0, e.second, n_kv, 0);\n'
        + ind + '        ggml_tensor * m = ggml_kq_mask_dev(ctx0, cpv, inp->self_pos, n_kv, ubatch.n_tokens, type);\n'
        + ind + '        ggml_format_name(m, "attn_kq_mask_dev_%s", ggml_backend_buft_name(e.first));\n'
        + ind + '        inp->self_kq_mask_dev.emplace_back(e.first, m);\n'
        + ind + '    }\n'
        + ind + '    inp->self_kq_mask     = nullptr;\n'
        + ind + '    inp->self_kq_mask_cnv = inp->self_kq_mask_dev.front().second;\n'
        + ind + '} else {\n'
        + ind + '    inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);\n'
        + ind + '    inp->self_kq_mask_cnv = inp->self_kq_mask;\n'
        + ind + '}\n')
    s = s[:m.start()] + block + s[m.end():]
    s = rep(s, '''void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);
''', '''void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    if (self_pos && self_pos->buffer) {
        ggml_backend_tensor_set(self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        mctx->upload_cell_pos();
    }
''', path='graph.cpp set_input')
    s = rep(s, '''    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {''', '''    if (inp_attn->self_kq_mask) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }
    if (inp_attn->self_pos && inp_attn->self_pos->buffer) {
        ggml_backend_tensor_set(inp_attn->self_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        mctx->get_attn()->upload_cell_pos();
    }

    if (inp_attn->self_k_rot) {''', path='graph.cpp hybrid set_input')
    fn = s.index('bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {')
    anchor = '    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);\n'
    i = s.index(anchor, fn)
    s = s[:i] + '''    res &= can_reuse_kq_mask(self_kq_mask ? self_kq_mask : self_kq_mask_cnv, mctx, params.ubatch, params.cparams);
    res &= self_pos == nullptr || self_pos->ne[0] == params.ubatch.n_tokens;
''' + s[i + len(anchor):]
    s = s.rstrip('\n') + '''

ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_l(int il) const {
    if (self_kq_mask_dev.empty()) {
        return self_kq_mask_cnv;
    }
    ggml_tensor * cp = mctx->get_cell_pos(il);
    for (const auto & e : self_kq_mask_dev) {
        if (e.second->src[0]->view_src == cp) {
            return e.second;
        }
    }
    return self_kq_mask_cnv;
}
'''
    if '#include "ggml-qsa.h"' not in s:
        s = s.replace('#include "llama-graph.h"\n', '#include "llama-graph.h"\n#include "../ggml/src/ggml-qsa.h"\n', 1)
    wr('/src/llama-graph.cpp', s); print('llama-graph.cpp patched')

# ---------------------------------------------------------------- qwen4exp.cpp
s = rd('/src/models/qwen4exp.cpp')
if 'get_kq_mask_l(il)' not in s:
    s = rep(s, '    ggml_tensor * kq_mask = inp->get_kq_mask();\n', '    ggml_tensor * kq_mask = inp->get_kq_mask_l(il);\n', path='qwen4exp 1143')
    s = rep(s, 'build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il)', 'build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask_l(il), sections, il)', path='qwen4exp 1215')
    wr('/src/models/qwen4exp.cpp', s); print('qwen4exp.cpp patched')
print('ALL PATCHES OK')
