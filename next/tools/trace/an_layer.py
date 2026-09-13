import csv, sys, collections, bisect, re
fn = sys.argv[1]; dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cpu = []; ker = []
for row in csv.reader(open(fn, newline='')):
    if len(row) < 9: continue
    try: s = int(row[2]); e = int(row[3])
    except: continue
    if row[0] == 'CPU' and row[8] == 'DECODE_TARGET': cpu.append((s, e))
    elif row[0] == 'KERNEL' and int(row[1]) == dev: ker.append((s, e, row[8]))
cpu.sort(); ker.sort(); ks = [k[0] for k in ker]
def short(n):
    n = re.sub(r'^_Z\d+', '', n); n = re.sub(r'ggml_cuda_mm_fusion_args_device.*', '', n)
    for pat, rep in [(r'mul_mat_vec_qIL9ggml_type(\d+)ELi(\d+)ELb(\d)ELb(\d)ELb(\d)E.*', r'mmvq<t\1,n\2,g\3,b\4,f\5>'), (r'mul_mat_vec_q_moeIL9ggml_type(\d+)ELi(\d+)ELb(\d)E.*', r'mmvq_moe<t\1,n\2,\3>'),
                     (r'k_bin_bcastI.*?op_(\w+?)E.*', r'binbcast_\1'), (r'unary_op_kernelI.*?op_(\w+?)E.*', r'unary_\1'), (r'quantize_q8_1.*', 'quantize_q8_1'), (r'rms_norm_f32.*', 'rms_norm'), (r'scale_f32.*', 'scale'),
                     (r'k_get_rows.*dequantize_(\w+?)E.*', r'get_rows_\1'), (r'k_get_rows.*', 'get_rows'), (r'cpy_scalar_contiguous.*', 'cpy_cast'), (r'cpy_.*', 'cpy'), (r'concat.*', 'concat'), (r'pad_f32.*', 'pad'), (r'gated_delta_net.*', 'gdn'), (r'mul_mat_fIf.*', 'mul_mat_f'), (r'mul_mat_vec_fIf.*', 'mmvf'),
                     (r'top_k_.*?E', 'topk_radix'), (r'flash_attn.*', 'fattn'), (r'rope_multi.*', 'rope'), (r'cutlass.*|.*cublas.*|.*gemm.*', 'cublas'), (r'qsa_(\w+?)P.*', r'qsa_\1'), (r'set_rows.*', 'set_rows'), (r'k_argsort.*|.*RadixSort.*', 'sort'), (r'(ssm_conv|conv).*', 'ssm_conv'), (r'softmax.*|soft_max.*', 'softmax'), (r'argmax.*', 'argmax'), (r'(\w+?)[IP].*', r'\1')]:
        m = re.match(pat, n)
        if m: return re.sub(pat, rep, n)[:28]
    return n[:28]
# steady steps: use steps 5..-2
steps = cpu[5:-2]
tot = collections.Counter(); cnt = collections.Counter(); N = 0
for i, (s, e) in enumerate(steps):
    nxt = cpu[cpu.index((s, e)) + 1][0]
    if nxt - s > 400e6: continue
    N += 1
    for k in ker[bisect.bisect_left(ks, s):bisect.bisect_left(ks, nxt)]:
        nm = short(k[2]); tot[nm] += k[1] - k[0]; cnt[nm] += 1
print(f'dev{dev}: {N} steps; kernels/step {sum(cnt.values())/N:.0f}; busy {sum(tot.values())/N/1e6:.2f} ms/step')
print('%-30s %8s %8s %8s' % ('kernel', 'n/step', 'ms/step', 'avg us'))
for nm, t in sorted(tot.items(), key=lambda x: -x[1])[:34]:
    print('%-30s %8.1f %8.3f %8.1f' % (nm, cnt[nm]/N, t/N/1e6, t/cnt[nm]/1e3))
# one layer's kernel sequence: take step 8, find the kernels between the 1st and 2nd 'gdn' occurrences (a GDN layer) 
s, e = steps[3]; nxt = cpu[cpu.index((s, e)) + 1][0]
seq = [short(k[2]) for k in ker[bisect.bisect_left(ks, s):bisect.bisect_left(ks, nxt)]]
idx = [i for i, n in enumerate(seq) if n == 'gdn']
if len(idx) >= 2:
    lay = seq[idx[0]:idx[1]]
    print(f'\none GDN layer = {len(lay)} kernels:')
    out = []; prev = None; c = 0
    for n in lay + [None]:
        if n == prev: c += 1; continue
        if prev is not None: out.append(f'{prev}x{c}' if c > 1 else prev)
        prev = n; c = 1
    print(' > '.join(out))
