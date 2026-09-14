# ggml-backend.cpp: run the device-side KQ mask op (GGML_OP_CUSTOM, qsa kind 9) on the device that owns its cell_pos
# table. Pass 1 of the scheduler only honours the node's own buffer, INPUT flags and WEIGHTS sources; this op has
# none of them, so the expansion passes placed it next to the token embedding lookup (wrong device in a layer split).
import re, sys
p = '/home/douya/src/llama.cpp-flashnext-20260913/ggml/src/ggml-backend.cpp'
s = open(p).read()
if 'ggml_qsa_kind(tensor) == 9' in s:
    print('already patched'); sys.exit(0)
inc_anchor = '#include "ggml-impl.h"\n'
assert s.count(inc_anchor) == 1, 'include anchor'
s = s.replace(inc_anchor, inc_anchor + '#include "ggml-qsa.h"\n', 1)
anchor = '    // operations with weights are preferably run on the same backend as the weights\n'
assert s.count(anchor) == 1, 'anchor'
block = '''    // NEXT: the device-side KQ mask (custom op kind 9) reads the KV cache's cell-position table, which lives on one
    // device; run the op there so that neither the table nor the mask has to cross devices in a layer split
    if (tensor->op == GGML_OP_CUSTOM && ggml_qsa_kind(tensor) == 9 && tensor->src[0] != NULL) {
        const struct ggml_tensor * base = tensor->src[0]->view_src ? tensor->src[0]->view_src : tensor->src[0];
        if (base->buffer != NULL && !ggml_backend_buffer_is_host(base->buffer)) {
            cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, base, tensor);
            if (cur_backend_id != -1) {
                SET_CAUSE(tensor, "1.mask");
                return cur_backend_id;
            }
        }
    }

'''
s = s.replace(anchor, block + anchor, 1)
open(p, 'w').write(s)
print('patched', p)
