# after deploy_monitor2 finishes: long-context sweep, TTFT at 70K, a live decode trace, then the mini block-cache equality test
P=/home/douya/qwen-3.8-next-opt
while ! grep -q "MONITOR DONE\|SERVICE DIED" /home/douya/tests/deploy-monitor4.out 2>/dev/null; do sleep 15; done
grep -q "SERVICE DIED" /home/douya/tests/deploy-monitor4.out && { echo "service died, skipping"; exit 1; }
echo "[post3] start $(date +%T)"
python3 /home/douya/tests/prod_sweep.py long 2>&1
python3 /home/douya/tests/ttft_probe.py 70000 2>&1
echo "[post3] live trace during a 70K decode $(date +%T)"
python3 - << 'PY'
import json, time, urllib.request, glob, os, threading
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=3600).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:70000] + post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数。\n'}, '/tokenize')['tokens']
post(dict(prompt=toks, n_predict=1, cache_prompt=True))
P = '/home/douya/qwen-3.8-next-opt'; off = os.path.getsize(P + '/logs/profile.csv')
res = {}
def gen(): res['r'] = post(dict(prompt=toks, n_predict=400, cache_prompt=True, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=7))
th = threading.Thread(target=gen); th.start(); time.sleep(2.0)
open(P + '/profile.flag', 'w').close(); time.sleep(12); os.remove(P + '/profile.flag'); th.join(); time.sleep(1)
out = '/home/douya/tests/trace-v4-70k.csv'
with open(P + '/logs/profile.csv', 'rb') as f, open(out, 'wb') as g: f.seek(off); g.write(f.read())
t = res['r']['timings']; print('70K sampled decode during trace:', round(t['predicted_per_second'], 1), 'tok/s (traced part slower)', 'draft', t['draft_n'], t['draft_n_accepted'])
PY
python3 /home/douya/tests/live_trace.py 2>/dev/null | head -0; python3 - << 'PY'
import csv, bisect, collections
fn = '/home/douya/tests/trace-v4-70k.csv'
tg = []; cp = []; cpu = collections.defaultdict(list); ker = []
for row in csv.reader(open(fn, newline='')):
    if len(row) < 9: continue
    try: s = int(row[2]); e = int(row[3])
    except: continue
    if row[0] == 'CPU':
        cpu[row[8]].append((e - s) / 1e6)
        if row[8] == 'DECODE_TARGET': tg.append((s, e))
    elif row[0] == 'COPY': cp.append((s, e, int(row[1]), int(row[4]), row[8]))
    elif row[0] == 'KERNEL': ker.append((s, e, int(row[1])))
tg.sort(); cp.sort(); ker.sort(); cs = [c[0] for c in cp]; ks = [k[0] for k in ker]
vol = collections.Counter(); busy = collections.Counter(); n = 0; per = []
for i in range(3, len(tg) - 1):
    s = tg[i][0]; nxt = tg[i + 1][0]
    if nxt - s > 400e6: continue
    n += 1; per.append(nxt - s)
    for c in cp[bisect.bisect_left(cs, s):bisect.bisect_left(cs, nxt)]: vol[(c[2], c[4])] += c[3]
    for k in ker[bisect.bisect_left(ks, s):bisect.bisect_left(ks, nxt)]: busy[k[2]] += k[1] - k[0]
print('markers:', {k: (len(v), round(sum(v) / len(v), 2)) for k, v in cpu.items()})
print('steps', n, 'period %.1f ms' % (sum(per) / n / 1e6), 'GPU busy ms/step', {d: round(v / n / 1e6, 1) for d, v in sorted(busy.items())})
print('MB/step (dev,kind):', {k: round(v / n / 1e6, 2) for k, v in sorted(vol.items())})
PY
echo "POST3 DONE $(date +%T)"
