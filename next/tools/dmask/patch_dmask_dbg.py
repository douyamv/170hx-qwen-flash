# one-time stderr diagnostics for the device-mask path (NEXT_DEVICE_MASK_VERBOSE=1)
F = '/home/douya/src/llama.cpp-flashnext-20260913'
p = F + '/src/llama-kv-cache.cpp'; s = open(p).read()
if 'device_mask_ok:' not in s:
    old = '''    // the GPU rule is "cell used and cell.pos <= token.pos": one stream, one sequence, no SWA, no ALiBi
    return n_stream == 1 && n_seq_max == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi;
}'''
    new = '''    // the GPU rule is "cell used and cell.pos <= token.pos": one stream, one sequence, no SWA, no ALiBi
    const bool ok = n_stream == 1 && n_seq_max == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi;
    static const bool verbose = getenv("NEXT_DEVICE_MASK_VERBOSE") != nullptr;
    if (verbose) {
        static int printed = 0;
        if (printed++ < 4) fprintf(stderr, "device_mask_ok: ok=%d n_stream=%u n_seq_max=%u swa=%d alibi=%d tensors=%zu\\n", (int) ok, n_stream, n_seq_max, (int) swa_type, (int) hparams.use_alibi, cell_pos_dev.size());
    }
    return ok;
}'''
    assert s.count(old) == 1; s = s.replace(old, new); open(p, 'w').write(s); print('kv-cache: device_mask_ok diagnostics added')
p = F + '/src/llama-graph.cpp'; s = open(p).read()
if 'device mask: %zu' not in s:
    old = '            inp->self_kq_mask     = nullptr;\n            inp->self_kq_mask_cnv = inp->self_kq_mask_dev.front().second;\n'
    assert s.count(old) == 1
    new = old + '            if (getenv("NEXT_DEVICE_MASK_VERBOSE")) { static int printed = 0; if (printed++ < 4) fprintf(stderr, "device mask: %zu masks, n_kv=%lld, n_tokens=%u\\n", inp->self_kq_mask_dev.size(), (long long) n_kv, ubatch.n_tokens); }\n'
    s = s.replace(old, new); open(p, 'w').write(s); print('graph: device mask diagnostics added')
