# hybrid input: can_reuse must not dereference the (null) host mask when the GPU mask is used
F = '/home/douya/src/llama.cpp-flashnext-20260913'
p = F + '/src/llama-graph.cpp'; s = open(p).read()
old = '    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);\n'
new = '''    res &= can_reuse_kq_mask(inp_attn->self_kq_mask ? inp_attn->self_kq_mask : inp_attn->self_kq_mask_cnv, mctx->get_attn(), params.ubatch, params.cparams);
    res &= inp_attn->self_pos == nullptr || inp_attn->self_pos->ne[0] == params.ubatch.n_tokens;
'''
c = s.count(old)
if c:
    s = s.replace(old, new); open(p, 'w').write(s); print('hybrid can_reuse patched (%d site)' % c)
else:
    print('already patched' if 'inp_attn->self_kq_mask ? inp_attn->self_kq_mask' in s else 'ANCHOR NOT FOUND')
# any other raw dereference of inp_attn->self_kq_mask->...?
import re
for m in re.finditer(r'inp_attn->self_kq_mask->', s): print('WARNING raw deref at', s[:m.start()].count('\n') + 1)
