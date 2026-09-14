# fix: the nested attn input of a hybrid memory keeps a stale mctx after graph reuse (only the hybrid's own mctx is
# refreshed by can_reuse) -> pass the KV-cache context explicitly to the check
S = '/home/douya/src/llama.cpp-flashnext-20260913/src/'
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new in s: continue
        assert s.count(old) == 1, (path, old[:70])
        s = s.replace(old, new, 1)
    open(path, 'w').write(s)
patch(S + 'llama-graph.h', [
    ("    int check_dev_masks(const llama_ubatch * ubatch, int verbose) const;\n",
     "    int check_dev_masks(const llama_kv_cache_context * kvctx, const llama_ubatch * ubatch, int verbose) const;\n"),
])
patch(S + 'llama-graph.cpp', [
    ("int llm_graph_input_attn_kv::check_dev_masks(const llama_ubatch * ubatch, int verbose) const {\n",
     "int llm_graph_input_attn_kv::check_dev_masks(const llama_kv_cache_context * kvctx, const llama_ubatch * ubatch, int verbose) const {\n"),
    ("    const std::vector<int32_t> cp = mctx->get_cell_pos_host();\n",
     "    const std::vector<int32_t> cp = kvctx->get_cell_pos_host();\n"),
])
patch(S + 'llama-context.cpp', [
    ("                bad += a->check_dev_masks(&ubatch, dm_check);\n",
     "                bad += a->check_dev_masks(a->mctx, &ubatch, dm_check);\n"),
    ("                if (h->inp_attn) bad += h->inp_attn->check_dev_masks(&ubatch, dm_check);\n",
     "                if (h->inp_attn) bad += h->inp_attn->check_dev_masks(h->mctx->get_attn(), &ubatch, dm_check);\n"),
])
patch(S + 'llama-context.cpp', [
    ('#include "llama-memory.h"\n', '#include "llama-memory.h"\n#include "llama-memory-hybrid.h"\n'),
])
print('dmcheck2 patched')
