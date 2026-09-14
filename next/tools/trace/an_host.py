import csv, sys, collections, bisect
fn = sys.argv[1]
cpu = []; api = []; ker = []; cpy = []
with open(fn, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 9: continue
        k = row[0]
        try: s = int(row[2]); e = int(row[3])
        except: continue
        if k == 'CPU': cpu.append((s, e, row[8]))
        elif k == 'API': api.append((s, e, row[8].split('_v')[0], row[6]))
        elif k == 'KERNEL': ker.append((s, e, int(row[1]), row[8]))
        elif k == 'COPY': cpy.append((s, e, int(row[1]), int(row[4]), row[8]))
cpu.sort(); api.sort(); ker.sort(); cpy.sort()
names = collections.Counter(c[2] for c in cpu)
print('CPU marker kinds:', dict(names.most_common(12)))
tg = [c for c in cpu if c[2] == 'DECODE_TARGET']
steady = tg[3:-1]; N = len(steady)
ks = [x[0] for x in ker]; as_ = [x[0] for x in api]; cs = [x[0] for x in cpu]
per_api = collections.Counter(); per_api_n = collections.Counter(); per_cpu = collections.Counter(); per_cpu_n = collections.Counter()
gap_turn = 0; gap01 = 0; gap12 = 0; first0 = 0; period = 0; tgt_dur = 0; dev_busy = collections.Counter(); dev_span = collections.Counter()
for (s, e, _) in steady:
    nxt = tg[tg.index((s, e, _)) + 1][0]
    period += nxt - s; tgt_dur += e - s
    a = bisect.bisect_left(as_, s); b = bisect.bisect_left(as_, nxt)
    for x in api[a:b]: per_api[x[2]] += x[1] - x[0]; per_api_n[x[2]] += 1
    a = bisect.bisect_left(cs, s); b = bisect.bisect_left(cs, nxt)
    for x in cpu[a:b]: per_cpu[x[2]] += x[1] - x[0]; per_cpu_n[x[2]] += 1
    a = bisect.bisect_left(ks, s); b = bisect.bisect_left(ks, nxt)
    first = {}; last = {}
    for kk in ker[a:b]:
        first.setdefault(kk[2], kk[0]); last[kk[2]] = max(last.get(kk[2], 0), kk[1]); dev_busy[kk[2]] += kk[1] - kk[0]
    if 0 in first: first0 += first[0] - s
    if 0 in last and 1 in first: gap01 += first[1] - last[0]
    if 1 in last and 2 in first: gap12 += first[2] - last[1]
    if 2 in last: gap_turn += nxt - last[2]
    for d in first: dev_span[d] += last[d] - first[d]
print('steps %d  period %.2f ms  DECODE_TARGET marker dur %.2f ms' % (N, period / N / 1e6, tgt_dur / N / 1e6))
print('avg: step start -> first dev0 kernel %.2f ms | dev0 last -> dev1 first %.2f ms | dev1 last -> dev2 first %.2f ms | dev2 last kernel -> next step %.2f ms' % (first0 / N / 1e6, gap01 / N / 1e6, gap12 / N / 1e6, gap_turn / N / 1e6))
for d in sorted(dev_busy): print('  dev%d busy %.2f span %.2f ms' % (d, dev_busy[d] / N / 1e6, dev_span[d] / N / 1e6))
print('\nCPU markers per step (ms, n):')
for k, v in sorted(per_cpu.items(), key=lambda x: -x[1])[:12]: print('  %-28s %7.2f %6.1f' % (k, v / N / 1e6, per_cpu_n[k] / N))
print('\nAPI per step (ms, n):')
for k, v in sorted(per_api.items(), key=lambda x: -x[1])[:16]: print('  %-34s %7.2f %6.1f' % (k, v / N / 1e6, per_api_n[k] / N))
# one typical step timeline of CPU markers
s, e, _ = steady[N // 2]; nxt = tg[tg.index((s, e, _)) + 1][0]
a = bisect.bisect_left(cs, s); b = bisect.bisect_left(cs, nxt)
print('\ntimeline of one step (offset ms, dur ms, marker):')
for x in cpu[a:b][:40]: print('  %7.2f %7.2f %s' % ((x[0] - s) / 1e6, (x[1] - x[0]) / 1e6, x[2]))
a = bisect.bisect_left(ks, s); b = bisect.bisect_left(ks, nxt)
print('kernel activity in that step (offset of first/last kernel per dev):')
first = {}; last = {}
for kk in ker[a:b]: first.setdefault(kk[2], kk[0]); last[kk[2]] = max(last.get(kk[2], 0), kk[1])
for d in sorted(first): print('  dev%d %7.2f .. %7.2f ms' % (d, (first[d] - s) / 1e6, (last[d] - s) / 1e6))
# gaps > 0.3 ms inside the step on dev2 (largest span)
print('gaps > 0.3 ms between consecutive kernels on dev2 in that step:')
prev = None
for kk in ker[a:b]:
    if kk[2] != 2: continue
    if prev is not None and kk[0] - prev[1] > 3e5: print('  %7.2f -> %7.2f (%.2f ms) after %s' % ((prev[1] - s) / 1e6, (kk[0] - s) / 1e6, (kk[0] - prev[1]) / 1e6, prev[3][:60]))
    prev = kk
