#!/usr/bin/env python3
"""TTFT probe on production: after a cached long prompt, send chat-like continuations and report prompt_ms + checkpoint log lines."""
import json, sys, time, glob, urllib.request, subprocess
URL = 'http://127.0.0.1:8093'; LOG = '/home/douya/qwen-3.8-next-opt/logs/server.log'
n = int(sys.argv[1]) if len(sys.argv) > 1 else 70000
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request(URL + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=3600).read())
def tok(s): return post({'content': s, 'add_special': False}, '/tokenize')['tokens']
def logtail(pat, k=3): return subprocess.run(['bash', '-c', f'tail -c 300000 {LOG} | grep -a "{pat}" | tail -{k} | cut -c30-170'], capture_output=True, text=True).stdout.strip()
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
base = tok(text)[:n]
t0 = time.time(); r = post(dict(prompt=base, n_predict=64, cache_prompt=True, temperature=0)); print(f'prefill+64: {time.time()-t0:.1f}s prompt_n {r["timings"]["prompt_n"]}', flush=True)
gen = r.get('tokens') or tok(r['content'])
for i in range(3):
    turn = base + gen + tok(f'\n\n用户：请再补充第{i+1}点，简短一些。\n助手：')
    t0 = time.time(); r = post(dict(prompt=turn, n_predict=48, cache_prompt=True, temperature=0, return_tokens=True)); wall = time.time() - t0
    t = r['timings']; print(f'turn {i+1}: wall {wall:.2f}s prompt_n {t["prompt_n"]} prompt_ms {t["prompt_ms"]:.0f} cache_n {t.get("cache_n")} gen_tps {t["predicted_per_second"]:.1f}', flush=True)
    gen = gen + tok(f'\n\n用户：请再补充第{i+1}点，简短一些。\n助手：') + (r.get('tokens') or tok(r['content']))
print('checkpoint log:', logtail('context checkpoint', 4))
print('errors:', logtail('ERR\\|abort\\|mismatch', 3) or 'none')
