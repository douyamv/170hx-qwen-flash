# the second hybrid input (V-less llm_graph_input_attn_k) never takes the device path: restore its original can_reuse line
F = '/home/douya/src/llama.cpp-flashnext-20260913'
p = F + '/src/llama-graph.cpp'; s = open(p).read()
new = '''    res &= can_reuse_kq_mask(inp_attn->self_kq_mask ? inp_attn->self_kq_mask : inp_attn->self_kq_mask_cnv, mctx->get_attn(), params.ubatch, params.cparams);
    res &= inp_attn->self_pos == nullptr || inp_attn->self_pos->ne[0] == params.ubatch.n_tokens;
'''
orig = '    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);\n'
lines = s.split('\n')
# find each occurrence and check the class of inp_attn in the enclosing function: look back for 'llm_graph_input_attn_k *' vs 'llm_graph_input_attn_kv *' in the class definition is not local; instead use the function name
import re
occ = [m.start() for m in re.finditer(re.escape(new), s)]
print('occurrences:', len(occ))
if len(occ) == 2:
    # the second one is in the attn_k-based hybrid (compile error at that site) -> restore
    i = occ[1]; s = s[:i] + orig + s[i + len(new):]
    open(p, 'w').write(s); print('restored the second site')
