#!/bin/bash
# after opt-v6 deploy: short sweep, 70K sweep + TTFT, moea A/B at 70K (runtime kill file), live decode trace
P=/home/douya/qwen-3.8-next-opt; T=/home/douya/tests
echo "[post10] start $(date +%T)"
python3 $T/prod_sweep.py short 2>&1 | tail -n 12
python3 $T/prod_sweep.py long 2>&1
python3 $T/ttft_probe.py 70000 2>&1 | tail -n 6
echo "[post10] moea A/B at 70K (greedy 200 tokens, same prompt) $(date +%T)"
python3 - << 'PY'
import json, time, urllib.request, glob, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=3600).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:70000] + post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数。\n'}, '/tokenize')['tokens']
post(dict(prompt=toks, n_predict=1, cache_prompt=True))
P = '/home/douya/qwen-3.8-next-opt'
res = {}
for tag in ('moea_on', 'moea_off', 'moea_on2'):
    if tag == 'moea_off': open(P + '/opt/moea_off', 'w').close(); time.sleep(2.5)
    if tag == 'moea_on2': os.remove(P + '/opt/moea_off'); time.sleep(2.5)
    r = post(dict(prompt=toks, n_predict=200, cache_prompt=True, ignore_eos=True, temperature=0.0, seed=1))
    t = r['timings']; res[tag] = r
    print('%-9s %6.1f tok/s  draft %d accepted %d  %s' % (tag, t['predicted_per_second'], t['draft_n'], t['draft_n_accepted'], r['content'][:60].replace('\n', ' ')))
a = post({'content': res['moea_on']['content']}, '/tokenize')['tokens']; b = post({'content': res['moea_off']['content']}, '/tokenize')['tokens']; c = post({'content': res['moea_on2']['content']}, '/tokenize')['tokens']
pref = lambda x, y: next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
print('token prefix on/off = %d/%d, on/on2 = %d/%d' % (pref(a, b), len(a), pref(a, c), len(a)))
PY
echo "[post10] live trace during a 70K decode $(date +%T)"
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
out = '/mnt/slowdisk/tests-archive/traces/trace-v10-70k.csv'
with open(P + '/logs/profile.csv', 'rb') as f, open(out, 'wb') as g: f.seek(off); g.write(f.read())
t = res['r']['timings']; print('70K sampled decode during trace:', round(t['predicted_per_second'], 1), 'tok/s (traced part slower)', 'draft', t['draft_n'], t['draft_n_accepted'])
PY
: > $P/logs/profile.csv
python3 $T/an2.py /mnt/slowdisk/tests-archive/traces/trace-v10-70k.csv 2>&1 | head -n 40
echo "[post10] done $(date +%T)"
