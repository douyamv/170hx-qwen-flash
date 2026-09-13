#!/usr/bin/env python3
"""200K baseline on production v4: prefill time, decode tok/s (sampled x3 + greedy), TTFT, plus a 15 s trace during one decode."""
import json, time, urllib.request, glob, os, threading, subprocess
URL = 'http://127.0.0.1:8093'; P = '/home/douya/qwen-3.8-next-opt'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request(URL + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=7200).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
all_t = post({'content': text}, '/tokenize')['tokens']
instr = post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数，并说明它们之间的调用关系。\n'}, '/tokenize')['tokens']
base = all_t[:200000] + instr
print('tokens', len(base), flush=True)
t0 = time.time(); r = post(dict(prompt=base, n_predict=1, cache_prompt=True)); t = r['timings']
print('200K prefill: %.1f s, %.0f tok/s' % (time.time() - t0, t['prompt_per_second']), flush=True)
open(P + '/opt/spec_n_max', 'w').write('4'); time.sleep(1.2)
def run(tag, **kw):
    t0 = time.time(); r = post(dict(prompt=base + [271] * kw.pop('pad', 2), n_predict=kw.pop('n', 300), cache_prompt=True, **kw)); t = r['timings']
    print(f'{tag}: gen {t["predicted_per_second"]:.1f} tok/s  prompt_ms {t["prompt_ms"]:.0f}  acc {t["draft_n_accepted"]}/{t["draft_n"]}  head {r["content"][:40]!r}', flush=True); return r
run('200K greedy 4/q4', temperature=0, pad=2)
for i, seed in enumerate((7, 11, 13)):
    run(f'200K sampled seed{seed}', temperature=1.0, top_k=20, top_p=0.95, seed=seed, pad=3 + i)
# trace during a sampled run
off = os.path.getsize(P + '/logs/profile.csv'); res = {}
def gen(): res['r'] = post(dict(prompt=base + [271] * 7, n_predict=400, cache_prompt=True, temperature=1.0, top_k=20, top_p=0.95, seed=17))
th = threading.Thread(target=gen); th.start(); time.sleep(2.5)
open(P + '/profile.flag', 'w').close(); time.sleep(12); os.remove(P + '/profile.flag'); th.join(); time.sleep(1)
out = '/home/douya/tests/trace-v5-200k.csv'
with open(P + '/logs/profile.csv', 'rb') as f, open(out, 'wb') as g: f.seek(off); g.write(f.read())
t = res['r']['timings']; print('traced run: %.1f tok/s (slower under trace), acc %d/%d' % (t['predicted_per_second'], t['draft_n_accepted'], t['draft_n']), flush=True)
subprocess.run(['python3', '/home/douya/tests/an2.py', out])
