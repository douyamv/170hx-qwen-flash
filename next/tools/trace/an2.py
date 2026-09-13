import csv, sys, collections, bisect, re
fn = sys.argv[1]
cpu = []; api = []; ker = []; cpy = []
with open(fn, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 9: continue
        k = row[0]
        try: s = int(row[2]); e = int(row[3])
        except: continue
        if k == 'CPU': cpu.append((s, e, row[8]))
        elif k == 'API': api.append((s, e, row[8].split('_v')[0], int(row[5])))
        elif k == 'KERNEL': ker.append((s, e, int(row[1]), row[8], int(row[5])))
        elif k == 'COPY': cpy.append((s, e, int(row[1]), int(row[4]), int(row[5]), row[8]))
cpu.sort(); api.sort(); ker.sort(); cpy.sort()
tg = [c for c in cpu if c[2] == 'DECODE_TARGET']
def cat(n):
    n = n.lower()
    rules = [('mmvq_moe|mul_mat_vec_q.*id|mmvq.*ids|moe', 'moe_mmvq'), ('mul_mat_vec_q|mmvq', 'dense_mmvq'), ('mul_mat_q|mmq', 'mmq'),
             ('quantize_q8_1|quantize_mmq', 'quantize_act'), ('gated_delta|gdn|delta_net', 'gdn'), ('ssm_conv|conv1d|ssm', 'ssm_conv'),
             ('qsa|lightning|indexer|topk|top_k|argsort', 'qsa/topk'), ('flash|fattn', 'fattn'), ('rms_norm|norm', 'norm'), ('rope', 'rope'),
             ('soft_max|softmax', 'softmax'), ('get_rows', 'get_rows'), ('cpy|copy|dup', 'cpy'), ('bin_bcast|binbcast', 'binbcast'),
             ('unary|silu|sigmoid|gelu|swiglu|softplus|exp|tanh|relu', 'unary'), ('scale', 'scale'), ('concat', 'concat'), ('hc|hyper', 'hc'),
             ('cublas|gemm|sgemm|ampere|cutlass', 'cublas'), ('argmax', 'argmax'), ('sum|mean', 'sum/mean'), ('fill|memset|set', 'set')]
    for pat, c in rules:
        if re.search(pat, n): return c
    return 'other'
cats = {}
ks = [x[0] for x in ker]
steady = tg[3:-1]
print('steady steps', len(steady))
per_dev_busy = collections.Counter(); per_dev_cnt = collections.Counter(); per_cat = collections.Counter(); per_cat_cnt = collections.Counter()
per_name = collections.Counter(); per_name_cnt = collections.Counter()
period_tot = 0; span = collections.Counter()
for i, (s, e, _) in enumerate(steady):
    nxt = tg[tg.index((s, e, _)) + 1][0]
    period_tot += nxt - s
    a = bisect.bisect_left(ks, s); b = bisect.bisect_left(ks, nxt)
    first = {}; last = {}
    for kk in ker[a:b]:
        d = kk[2]; dur = kk[1] - kk[0]
        per_dev_busy[d] += dur; per_dev_cnt[d] += 1
        first.setdefault(d, kk[0]); last[d] = max(last.get(d, 0), kk[1])
        c = cats.get(kk[3]) or cats.setdefault(kk[3], cat(kk[3]))
        per_cat[(d, c)] += dur; per_cat_cnt[(d, c)] += 1
        per_name[(d, kk[3][:90])] += dur; per_name_cnt[(d, kk[3][:90])] += 1
    for d in first: span[d] += last[d] - first[d]
N = len(steady)
print('traced step period %.2f ms' % (period_tot / N / 1e6))
for d in sorted(per_dev_busy):
    print('dev%d kernels/step %.0f busy %.2f ms span %.2f ms' % (d, per_dev_cnt[d] / N, per_dev_busy[d] / N / 1e6, span[d] / N / 1e6))
print('total GPU busy per step %.2f ms' % (sum(per_dev_busy.values()) / N / 1e6))
print('\nby category (ms/step, kernels/step)')
for d in sorted(per_dev_busy):
    items = sorted(((v, c) for (dd, c), v in per_cat.items() if dd == d), reverse=True)
    print('dev%d: ' % d + ', '.join('%s %.2f/%d' % (c, v / N / 1e6, per_cat_cnt[(d, c)] / N) for v, c in items[:14]))
print('\ntop kernels (ms/step, n/step, avg us)')
for (d, n), v in sorted(per_name.items(), key=lambda x: -x[1])[:45]:
    print('dev%d %7.3f %6.1f %7.1f  %s' % (d, v / N / 1e6, per_name_cnt[(d, n)] / N, v / per_name_cnt[(d, n)] / 1e3, n))
# D2H copies per step
ps = [x[0] for x in cpy]
bytes_dir = collections.Counter(); t_dir = collections.Counter(); n_dir = collections.Counter()
for (s, e, _) in steady:
    nxt = tg[tg.index((s, e, _)) + 1][0]
    a = bisect.bisect_left(ps, s); b = bisect.bisect_left(ps, nxt)
    for c in cpy[a:b]:
        key = (c[2], c[5]); bytes_dir[key] += c[3]; t_dir[key] += c[1] - c[0]; n_dir[key] += 1
print('\ncopies per step (dev, kind): n, KB, ms')
for key in sorted(bytes_dir):
    print(key, round(n_dir[key] / N, 1), round(bytes_dir[key] / N / 1024, 1), round(t_dir[key] / N / 1e6, 3))
