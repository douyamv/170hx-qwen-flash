# host-timeline markers (RAII llama_trace_local, active only while the tracer collects): llama-context, server, speculative
import re, sys
F = '/home/douya/src/llama.cpp-flashnext-20260913'
def patch(path, pairs, include=None):
    s = open(path).read(); n = 0
    if include and include not in s:
        # after the first #include line
        i = s.index('#include'); s = s[:i] + include + '\n' + s[i:]
    for old, new in pairs:
        c = s.count(old)
        assert 1 <= c <= 2, (path, old[:60], c)
        s = s.replace(old, new); n += c
    open(path, 'w').write(s); print(path.split('/')[-1], 'patched', n)
# 1. llama-context.cpp
patch(F + '/src/llama-context.cpp', [
    ('        res->set_inputs(&ubatch);\n', '        { llama_trace_local trace_si("SET_INPUTS"); res->set_inputs(&ubatch); }\n'),
    ('    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);\n', '    ggml_status status;\n    { llama_trace_local trace_gc("GRAPH_COMPUTE"); status = graph_compute(res->get_gf(), ubatch.n_tokens > 1); }\n'),
    ('void llama_context::synchronize() {\n    if (!sched) {\n        return;\n    }\n', 'void llama_context::synchronize() {\n    if (!sched) {\n        return;\n    }\n    llama_trace_local trace_sync("SYNC");\n'),
])
# 2. server-context.cpp
patch(F + '/tools/server/server-context.cpp', [
    ('                    if (do_checkpoint) {\n                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);\n                    }\n',
     '                    if (do_checkpoint) {\n                        llama_trace_local trace_ck("SRV_CKPT");\n                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);\n                    }\n'),
    ('        int ret = 0;\n        queue_tasks.yield_to_queue([&]() {\n            ret = llama_decode(ctx_tgt, batch_view);\n            if (ret == 0 && has_output) {\n                llama_synchronize(ctx_tgt);\n            }\n        });\n',
     '        int ret = 0;\n        {\n        llama_trace_local trace_dec("SRV_DECODE");\n        queue_tasks.yield_to_queue([&]() {\n            ret = llama_decode(ctx_tgt, batch_view);\n            if (ret == 0 && has_output) {\n                llama_synchronize(ctx_tgt);\n            }\n        });\n        }\n'),
    ('            // verify and try to accept the draft\n            {\n                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));\n',
     '            // verify and try to accept the draft\n            {\n                llama_trace_local trace_acc("SRV_ACCEPT");\n                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));\n'),
    ('        if (!drafting.empty()) {\n            queue_tasks.yield_to_queue([&]() {\n                common_speculative_draft(spec.get());\n            });\n        }\n',
     '        if (!drafting.empty()) {\n            llama_trace_local trace_drf("SRV_DRAFT");\n            queue_tasks.yield_to_queue([&]() {\n                common_speculative_draft(spec.get());\n            });\n        }\n'),
    ('    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {\n',
     '    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {\n        llama_trace_local trace_send("SRV_SEND");\n'),
    ('                scoped_timer timer(t_sampl, n_sampl);\n                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);\n',
     '                scoped_timer timer(t_sampl, n_sampl);\n                llama_trace_local trace_smp("SRV_SAMPLE");\n                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);\n'),
], include='#include "../../src/llama-trace-local.h"')
# 3. speculative.cpp (MTP driver)
patch(F + '/common/speculative.cpp', [
    ('    bool process(const llama_batch & batch_in) override {\n        if (batch_in.n_tokens <= 0) {\n',
     '    bool process(const llama_batch & batch_in) override {\n        llama_trace_local trace_p("SPEC_PROCESS");\n        if (batch_in.n_tokens <= 0) {\n'),
    ('    void draft(common_speculative_draft_params_vec & dparams) override {\n        auto & ctx_dft = params.ctx_dft;\n        common_batch_clear(batch);\n        int n_drafting = 0;\n',
     '    void draft(common_speculative_draft_params_vec & dparams) override {\n        llama_trace_local trace_d("SPEC_DRAFT");\n        auto & ctx_dft = params.ctx_dft;\n        common_batch_clear(batch);\n        int n_drafting = 0;\n'),
    ('                auto * smpl = smpls[seq_id].get();\n                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);\n',
     '                auto * smpl = smpls[seq_id].get();\n                { llama_trace_local trace_s("SPEC_DRAFT_SAMPLE"); common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true); }\n'),
], include='#include "../src/llama-trace-local.h"')
