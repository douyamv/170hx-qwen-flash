# expand the mask ops into the graph right after the attention input is built (in the llm_graph_context callers, which
# own gf), so they are ordered before the layers instead of being pulled in by the DFS right before their first
# consumer, where they split node pairs that ggml-cuda fuses by adjacency and change the layer's numerics
S = '/home/douya/src/llama.cpp-flashnext-20260913/src/'
p = S + 'llama-graph.cpp'; s = open(p).read()
bad = "                ggml_build_forward_expand(gf, m); // keep the layers' node order (and ggml-cuda fusion pairs) intact\n"
if bad in s:
    s = s.replace(bad, ''); print('removed misplaced expand')
old1 = "    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);\n"
new1 = old1 + "    for (const auto & e : inp->self_kq_mask_dev) { ggml_build_forward_expand(gf, e.second); } // NEXT: before the layers (keeps fusion pairs intact)\n"
old2 = "    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());\n"
new2 = old2 + "    for (const auto & e : inp_attn->self_kq_mask_dev) { ggml_build_forward_expand(gf, e.second); } // NEXT: before the layers (keeps fusion pairs intact)\n"
for old, new in ((old1, new1), (old2, new2)):
    if new not in s:
        assert s.count(old) == 1, old; s = s.replace(old, new, 1)
open(p, 'w').write(s); print('early expand in callers')
