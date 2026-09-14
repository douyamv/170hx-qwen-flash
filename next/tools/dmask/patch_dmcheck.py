# NEXT_DEVICE_MASK_CHECK=N: after every graph compute, read the GPU-generated KQ masks back and compare them cell by
# cell with the host rule (non-empty cell && cell.pos <= token.pos). Prints DM_CHECK lines to stderr; N>1 prints details.
import sys
S = '/home/douya/src/llama.cpp-flashnext-20260913/src/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new in s: continue
        assert s.count(old) == 1, (path, old[:60])
        s = s.replace(old, new, 1)
    open(path, 'w').write(s)

# 1. kv cache: host copy of the cell positions
patch(S + 'llama-kv-cache.h', [
    ("    void upload_cell_pos() const;\n",
     "    void upload_cell_pos() const;\n    std::vector<int32_t> get_cell_pos_host() const; // NEXT: debug (NEXT_DEVICE_MASK_CHECK)\n"),
    ("    void upload_cell_pos() const { kv->upload_cell_pos(); }\n",
     "    void upload_cell_pos() const { kv->upload_cell_pos(); }\n    std::vector<int32_t> get_cell_pos_host() const { return kv->get_cell_pos_host(); }\n"),
])
patch(S + 'llama-kv-cache.cpp', [
    ("void llama_kv_cache::upload_cell_pos() const {\n",
     '''std::vector<int32_t> llama_kv_cache::get_cell_pos_host() const {
    const uint32_t kv_size = v_cells[0].size();
    std::vector<int32_t> host((size_t) kv_size * n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t i = 0; i < kv_size; ++i) {
            host[(size_t) s * kv_size + i] = cells.is_empty(i) ? -1 : cells.pos_get(i);
        }
    }
    return host;
}

void llama_kv_cache::upload_cell_pos() const {
'''),
])
# 2. attn input: check method
patch(S + 'llama-graph.h', [
    ("    ggml_tensor * get_kq_mask_l(int il) const;\n",
     "    ggml_tensor * get_kq_mask_l(int il) const;\n    // NEXT: debug — compare the GPU masks with the host rule, returns the number of mismatching entries\n    int check_dev_masks(const llama_ubatch * ubatch, int verbose) const;\n"),
])
patch(S + 'llama-graph.cpp', [
    ("ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_l(int il) const {\n",
     '''int llm_graph_input_attn_kv::check_dev_masks(const llama_ubatch * ubatch, int verbose) const {
    int bad_total = 0;
    if (self_kq_mask_dev.empty() || !self_pos) {
        return 0;
    }
    const std::vector<int32_t> cp = mctx->get_cell_pos_host();
    const uint32_t n_tokens = ubatch->n_tokens;
    for (const auto & e : self_kq_mask_dev) {
        ggml_tensor * m = e.second;
        const int64_t n_kv = m->ne[0], n_rows = m->ne[1];
        std::vector<uint8_t> buf(ggml_nbytes(m));
        ggml_backend_tensor_get(m, buf.data(), 0, buf.size());
        int bad = 0;
        for (int64_t t = 0; t < n_rows; ++t) {
            const llama_pos p1 = t < (int64_t) n_tokens ? ubatch->pos[t] : -1;
            for (int64_t c = 0; c < n_kv; ++c) {
                const bool keep_exp = t < (int64_t) n_tokens && c < (int64_t) cp.size() && cp[c] >= 0 && cp[c] <= p1;
                float got;
                if (m->type == GGML_TYPE_F16) {
                    got = ggml_fp16_to_fp32(((const ggml_fp16_t *) buf.data())[t*n_kv + c]);
                } else {
                    got = ((const float *) buf.data())[t*n_kv + c];
                }
                const bool keep_got = got == 0.0f;
                if (keep_got != keep_exp || (!keep_got && !(got == -INFINITY))) {
                    if (bad < 5 && verbose > 1) {
                        fprintf(stderr, "DM_CHECK mismatch %s: row %lld cell %lld cell_pos %d tok_pos %d got %g expected %s\\n", ggml_backend_buft_name(e.first), (long long) t, (long long) c, c < (int64_t) cp.size() ? cp[c] : -2, (int) p1, got, keep_exp ? "0" : "-inf");
                    }
                    bad++;
                }
            }
        }
        if (bad || verbose > 1) {
            fprintf(stderr, "DM_CHECK %s: n_kv=%lld rows=%lld n_tokens=%u pos[0]=%d mismatches=%d\\n", ggml_backend_buft_name(e.first), (long long) n_kv, (long long) n_rows, n_tokens, n_tokens ? (int) ubatch->pos[0] : -1, bad);
        }
        bad_total += bad;
    }
    return bad_total;
}

ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_l(int il) const {
'''),
])
# 3. context: run the check after every compute when NEXT_DEVICE_MASK_CHECK is set
patch(S + 'llama-context.cpp', [
    ("    ret = GGML_STATUS_SUCCESS;\n\n    return res;\n}\n\nint llama_context::encode(const llama_batch & batch_inp) {",
     '''    // NEXT: debug — verify the GPU-generated KQ masks against the host rule (NEXT_DEVICE_MASK_CHECK=1: summary on
    // mismatch only, =2: every ubatch + details)
    static const int dm_check = [] { const char * e = getenv("NEXT_DEVICE_MASK_CHECK"); return e ? atoi(e) : 0; }();
    if (dm_check) {
        ggml_backend_sched_synchronize(sched.get());
        static int64_t n_checked = 0, n_bad_ubatches = 0;
        int bad = 0;
        for (const auto & inp : res->inputs) {
            if (auto * a = dynamic_cast<llm_graph_input_attn_kv *>(inp.get())) {
                bad += a->check_dev_masks(&ubatch, dm_check);
            } else if (auto * h = dynamic_cast<llm_graph_input_mem_hybrid *>(inp.get())) {
                if (h->inp_attn) bad += h->inp_attn->check_dev_masks(&ubatch, dm_check);
            }
        }
        n_checked++;
        if (bad) n_bad_ubatches++;
        if (bad || dm_check > 1 || n_checked % 500 == 0) {
            fprintf(stderr, "DM_CHECK ubatch #%lld n_tokens=%u: %d mismatching entries (bad ubatches so far %lld)\\n", (long long) n_checked, ubatch.n_tokens, bad, (long long) n_bad_ubatches);
        }
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {'''),
])
print('dmcheck patched')
