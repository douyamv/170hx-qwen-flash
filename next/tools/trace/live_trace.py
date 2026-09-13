import os, time, csv, bisect, collections, subprocess, re
P = '/home/douya/qwen-3.8-next-opt'; FLAG = P + '/profile.flag'; PROF = P + '/logs/profile.csv'; LOG = P + '/logs/server.log'
def last_activity():
    out = subprocess.run(['bash', '-c', f'tail -c 200000 {LOG} | grep -a "slot" | tail -1'], capture_output=True, text=True).stdout.strip()
    return out[:120]
print('last slot line:', last_activity())
off = os.path.getsize(PROF) if os.path.exists(PROF) else 0
open(FLAG, 'w').close(); t0 = time.time()
time.sleep(16)
if os.path.exists(FLAG): os.remove(FLAG)
time.sleep(1.5)
out = '/home/douya/tests/trace-live-%s.csv' % time.strftime('%H%M%S')
with open(PROF, 'rb') as f, open(out, 'wb') as g:
    f.seek(off); g.write(f.read())
print('saved', out, os.path.getsize(out), 'bytes; last slot line now:', last_activity())
cpu = []; api = []; ker = []
with open(out, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 9: continue
        try: s = int(row[2]); e = int(row[3])
        except: continue
        if row[0] == 'CPU': cpu.append((s, e, row[8]))
        elif row[0] == 'API': api.append((s, e, row[8].split('_v')[0]))
        elif row[0] == 'KERNEL': ker.append((s, e, int(row[1]), row[8]))
cpu.sort(); api.sort(); ker.sort()
tg = [c for c in cpu if c[2] == 'DECODE_TARGET']
print('markers', collections.Counter(c[2] for c in cpu), 'kernels', len(ker), 'api', len(api))
if len(tg) > 5:
    ks = [k[0] for k in ker]
    # long target markers = prompt evals
    for (s, e, _) in tg:
        if e - s > 150e6: print('  long DECODE_TARGET (prompt eval?) dur %.0f ms' % ((e - s) / 1e6))
    rows = []
    for i in range(1, len(tg) - 1):
        s, e, _ = tg[i]; nxt = tg[i + 1][0]
        if e - s > 150e6 or nxt - s > 400e6: continue
        a, b = bisect.bisect_left(ks, s), bisect.bisect_left(ks, nxt)
        busy = collections.Counter()
        for k in ker[a:b]: busy[k[2]] += k[1] - k[0]
        rows.append((nxt - s, e - s, busy))
    N = len(rows)
    if N:
        print('decode steps %d: period %.1f ms, target marker %.1f ms, GPU busy per dev: %s' % (N, sum(r[0] for r in rows) / N / 1e6, sum(r[1] for r in rows) / N / 1e6,
              {d: round(sum(r[2][d] for r in rows) / N / 1e6, 1) for d in range(4)}))
subprocess.run(['python3', '/home/douya/tests/an2.py', out])
