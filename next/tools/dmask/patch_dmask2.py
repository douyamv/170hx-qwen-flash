# NEXT_DEVICE_MASK=2: also accept ubatches with multi-component (M-RoPE) positions. For text tokens the host rule is
# exactly cell.pos <= token.pos on the first component (the extra x/y check only matters for image tokens sharing a position).
F = '/home/douya/src/llama.cpp-flashnext-20260913'
p = F + '/src/llama-graph.cpp'; s = open(p).read()
old = '        if (mctx_cur->device_mask_ok(cparams.causal_attn) && !ubatch.is_pos_2d() && (cparams.kv_unified || ubatch.n_seqs_unq == 1)) {'
new = '''        static const int device_mask_level = [] { const char * e = getenv("NEXT_DEVICE_MASK"); return e ? atoi(e) : 0; }();
        // level 2 also accepts multi-component (M-RoPE) positions: for text tokens the host rule is exactly
        // cell.pos <= token.pos on the first component (the x/y check only matters for image tokens sharing a position)
        if (mctx_cur->device_mask_ok(cparams.causal_attn) && (!ubatch.is_pos_2d() || device_mask_level >= 2) && (cparams.kv_unified || ubatch.n_seqs_unq == 1)) {'''
if old in s:
    s = s.replace(old, new); open(p, 'w').write(s); print('graph: level-2 condition patched')
else:
    print('already patched' if 'device_mask_level' in s else 'ANCHOR NOT FOUND')
