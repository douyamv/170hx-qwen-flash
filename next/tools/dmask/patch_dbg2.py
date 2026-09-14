F = '/home/douya/src/llama.cpp-flashnext-20260913'
p = F + '/src/llama-graph.cpp'; s = open(p).read()
if 'dmask cond:' not in s:
    old = '        if (mctx_cur->device_mask_ok(cparams.causal_attn) && !ubatch.is_pos_2d() && (cparams.kv_unified || ubatch.n_seqs_unq == 1)) {'
    assert s.count(old) == 1
    new = '''        if (getenv("NEXT_DEVICE_MASK_VERBOSE")) { static int pr = 0; if (pr++ < 6) fprintf(stderr, "dmask cond: ok=%d pos2d=%d unified=%d n_seqs_unq=%u n_tokens=%u causal=%d\\n", (int) mctx_cur->device_mask_ok(cparams.causal_attn), (int) ubatch.is_pos_2d(), (int) cparams.kv_unified, ubatch.n_seqs_unq, ubatch.n_tokens, (int) cparams.causal_attn); }
''' + old
    s = s.replace(old, new); open(p, 'w').write(s); print('graph: condition diagnostics added')
