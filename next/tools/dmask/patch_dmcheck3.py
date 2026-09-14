# the readback must keep the mask tensors alive: gallocr reuses an intermediate's memory as soon as its last consumer
# ran, so reading it after the graph gives garbage -> mark the masks as graph outputs while the check is enabled
S = '/home/douya/src/llama.cpp-flashnext-20260913/src/'
p = S + 'llama-graph.cpp'; s = open(p).read()
old = '                ggml_format_name(m, "attn_kq_mask_dev_%s", ggml_backend_buft_name(e.first));\n'
new = old + '''                static const bool dm_check = getenv("NEXT_DEVICE_MASK_CHECK") != nullptr;
                if (dm_check) {
                    ggml_set_output(m); // keep the mask alive for the readback check (no memory reuse by later nodes)
                }
'''
if new not in s:
    assert s.count(old) == 1; s = s.replace(old, new, 1); open(p, 'w').write(s); print('set_output added')
else: print('already')
