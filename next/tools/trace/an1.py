import csv, sys, collections, bisect
fn = sys.argv[1]
cpu = []; api = []; ker = []; cpy = []
with open(fn, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 9: continue
        k = row[0]
        try:
            s = int(row[2]); e = int(row[3])
        except: continue
        if k == 'CPU': cpu.append((s, e, row[8]))
        elif k == 'API': api.append((s, e, row[8].replace('_v', ' ').split()[0], int(row[5])))
        elif k == 'KERNEL': ker.append((s, e, int(row[1]), row[8], int(row[5])))
        elif k == 'COPY': cpy.append((s, e, int(row[1]), int(row[4]), int(row[5]), row[8]))
cpu.sort(); api.sort(); ker.sort()
print('rows cpu', len(cpu), 'api', len(api), 'kernel', len(ker), 'copy', len(cpy))
corr_api = {a[3]: a[2] for a in api}
api_s = [a[0] for a in api]
ker_corr = collections.defaultdict(list)
for kk in ker: ker_corr[kk[4]].append(kk)
seq = [c for c in cpu if c[2] in ('DECODE_TARGET', 'DECODE_MTP')]
print('markers', collections.Counter(c[2] for c in seq))
def summarize(s, e):
    i = bisect.bisect_left(api_s, s); j = bisect.bisect_right(api_s, e)
    cnt = collections.Counter(); tm = collections.Counter(); kern_dev = collections.Counter(); kern_dev_graph = collections.Counter()
    for a in api[i:j]:
        cnt[a[2]] += 1; tm[a[2]] += a[1] - a[0]
        if a[2] in ('cudaLaunchKernel', 'cudaGraphLaunch'):
            for kk in ker_corr.get(a[3], []):
                kern_dev[kk[2]] += 1
                if a[2] == 'cudaGraphLaunch': kern_dev_graph[kk[2]] += 1
    return cnt, tm, kern_dev, kern_dev_graph
# print a timeline of the first ~40 markers after the first target decode
print('\n%-14s %8s %6s %6s %6s %6s %6s  %s' % ('marker', 'ms', 'launch', 'glaunch', 'capt', 'sync', 'memcpy', 'kernels per dev (graph)'))
agg = collections.defaultdict(lambda: collections.Counter())
n_by = collections.Counter()
for idx, (s, e, name) in enumerate(seq):
    cnt, tm, kd, kg = summarize(s, e)
    key = name
    n_by[key] += 1
    agg[key]['ms'] += (e - s) / 1e6
    for a in ('cudaLaunchKernel', 'cudaGraphLaunch', 'cudaStreamBeginCapture', 'cudaStreamSynchronize', 'cudaMemcpyAsync'):
        agg[key][a] += cnt[a]
    if 1 <= idx <= 45:
        print('%-14s %8.2f %6d %6d %6d %6d %6d  %s' % (name, (e - s) / 1e6, cnt['cudaLaunchKernel'], cnt['cudaGraphLaunch'], cnt['cudaStreamBeginCapture'], cnt['cudaStreamSynchronize'], cnt['cudaMemcpyAsync'],
              ' '.join('d%d:%d(%d)' % (d, kd[d], kg[d]) for d in sorted(kd))))
print('\nAGG')
for k in agg:
    print(k, n_by[k], {a: round(v / n_by[k], 1) for a, v in agg[k].items()})
