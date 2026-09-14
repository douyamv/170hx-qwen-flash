# markers in the MTP draft driver (common/speculative.cpp): positional insertion after the signatures inside the mtp impl
p = '/home/douya/src/llama.cpp-flashnext-20260913/common/speculative.cpp'
lines = open(p).read().split('\n')
src = '\n'.join(lines)
if 'SPEC_PROCESS' in src:
    print('speculative.cpp already patched'); raise SystemExit
start = next(i for i, l in enumerate(lines) if l.startswith('struct common_speculative_impl_draft_mtp'))
def find_after(prefix, frm):
    return next(i for i in range(frm, len(lines)) if lines[i].strip().startswith(prefix))
ip = find_after('bool process(const llama_batch & batch_in) override {', start)
idr = find_after('void draft(common_speculative_draft_params_vec & dparams) override {', start)
isamp = next(i for i in range(start, len(lines)) if 'common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);' in lines[i])
assert ip < idr < isamp, (ip, idr, isamp)
lines[isamp] = lines[isamp].replace('common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);', '{ llama_trace_local trace_s("SPEC_DRAFT_SAMPLE"); common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true); }')
lines.insert(idr + 1, '        llama_trace_local trace_d("SPEC_DRAFT");')
lines.insert(ip + 1, '        llama_trace_local trace_p("SPEC_PROCESS");')
src = '\n'.join(lines)
inc = '#include "../src/llama-trace-local.h"'
if inc not in src:
    src = src.replace('#include "../src/llama-ext.h"', inc + '\n#include "../src/llama-ext.h"', 1)
open(p, 'w').write(src); print('speculative.cpp patched: process@%d draft@%d sample@%d' % (ip + 1, idr + 2, isamp + 3))
