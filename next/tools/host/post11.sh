#!/bin/bash
# after the v6.1 deploy: quick 2K check, then markers-only host timelines at 70K and 200K (+ decode speeds)
P=/home/douya/qwen-3.8-next-opt; T=/home/douya/tests
echo "[post11] start $(date +%T)"
python3 $T/prod_sweep.py short 2>&1 | grep -E "short_greedy|EQUAL" | cut -c1-140
cd $T && python3 - <<'PY'
import json, time, urllib.request, glob, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=3600).read())
P = '/home/douya/qwen-3.8-next-opt'
files = sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/ggml/src/ggml-cuda/*.cu') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/tools/server/*.cpp') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/common/*.cpp'))
text = ''.join(open(p, errors='ignore').read() for p in files)
alltoks = post({'content': text}, '/tokenize')['tokens']
q = post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数。\n'}, '/tokenize')['tokens']
for N in (70000, 200000):
    toks = alltoks[:N] + q
    t0 = time.time(); post(dict(prompt=toks, n_predict=1, cache_prompt=True)); print('prefill %d: %.1f s' % (N, time.time() - t0))
    r = post(dict(prompt=toks, n_predict=300, cache_prompt=True, ignore_eos=True, temperature=0.0)); t = r['timings']
    print('%dK greedy 4/q4: %.1f tok/s  draft %d acc %d' % (N // 1000, t['predicted_per_second'], t['draft_n'], t['draft_n_accepted']))
    open(P + '/logs/profile.csv', 'w').close()
    open(P + '/profile.flag.cpu', 'w').close(); time.sleep(0.3)
    r = post(dict(prompt=toks, n_predict=300, cache_prompt=True, ignore_eos=True, temperature=0.0)); t = r['timings']
    time.sleep(0.5); os.remove(P + '/profile.flag.cpu'); time.sleep(0.6)
    print('%dK greedy with CPU markers: %.1f tok/s' % (N // 1000, t['predicted_per_second']))
    out = '/mnt/slowdisk/tests-archive/traces/cpu-v61-%dk.csv' % (N // 1000)
    os.rename(P + '/logs/profile.csv', out) if False else open(out, 'wb').write(open(P + '/logs/profile.csv', 'rb').read())
    print('saved', out)
PY
: > $P/logs/profile.csv
for n in 70 200; do echo "== host timeline ${n}K =="; python3 $T/host/an_cpu.py /mnt/slowdisk/tests-archive/traces/cpu-v61-${n}k.csv 1 2>&1 | head -n 48 | cut -c1-120; done
echo "[post11] done $(date +%T)"
