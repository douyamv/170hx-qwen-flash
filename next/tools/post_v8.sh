#!/bin/bash
# after the 3-slot deploy: single-request 2K sweep, 3 concurrent streams (per-request + aggregate tok/s), one 70K request
P=/home/douya/qwen-3.8-next-opt; T=/home/douya/tests
echo "[post8] start $(date +%T)"
python3 $T/prod_sweep.py short 2>&1 | grep -E "short_greedy|EQUAL" | cut -c1-140
python3 - <<'PY'
import json, urllib.request, glob, threading, time
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=3600).read())
files = sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/ggml/src/ggml-cuda/*.cu'))
text = ''.join(open(p, errors='ignore').read() for p in files)
toks = post({'content': text}, '/tokenize')['tokens']
q = post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数。\n'}, '/tokenize')['tokens']
def run(tag, prompts, n_predict=300):
    res = {}
    def worker(i, p):
        t0 = time.time(); r = post(dict(prompt=p, n_predict=n_predict, cache_prompt=True, ignore_eos=True, temperature=0.0)); t = r['timings']
        res[i] = (time.time() - t0, t['prompt_n'], t['prompt_ms'], t['predicted_n'], t['predicted_per_second'], t.get('draft_n'), t.get('draft_n_accepted'))
    ths = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
    tot = sum(r[3] for r in res.values())
    for i in sorted(res): print('%s req %d: prompt %d (%.1f s) gen %d @ %.1f tok/s draft %s/%s wall %.1f s' % ((tag, i) + (res[i][1], res[i][2]/1000, res[i][3], res[i][4], res[i][5], res[i][6], res[i][0])))
    print('%s aggregate: %d tokens in %.1f s = %.1f tok/s' % (tag, tot, wall, tot/wall), flush=True)
# 3 concurrent ~3K-token contexts: prefill first (n_predict=1), then the timed generation round (cache hits)
P3 = [toks[0:3000] + q, toks[60000:63000] + q, toks[120000:123000] + q]
run('3x3K prefill', P3, 1)
run('3x3K concurrent', P3)
run('1x3K alone', [P3[0]])
run('2x3K concurrent', P3[:2])
# one 70K request (slot 0 gets a new context; prefill measured)
P70 = [toks[:70000] + q]
run('1x70K prefill', P70, 1)
run('1x70K alone', P70)
run('70K + 2x3K concurrent', [P70[0], P3[1], P3[2]])
PY
echo "[post8] done $(date +%T)"
