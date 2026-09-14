#!/bin/bash
# DM6: rewind (seq_rm -> dirty cell_pos -> full re-upload) must give the same tokens with the GPU mask as with the host mask
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_DM5 DONE" /home/douya/tests/dmask/val_dm5.out 2>/dev/null; do sleep 10; done
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
for dm in 1 0; do
  NEXT_DEVICE_MASK=$dm TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm6_$dm $A1 --spec-draft-backend-sampling || exit 1
  python3 - $dm <<'PY'
import json, urllib.request, glob, sys
dm = sys.argv[1]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
files = sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp'))
text = ''.join(open(p, errors='ignore').read() for p in files)
toks = post({'content': text}, '/tokenize')['tokens']
A = toks[:16384]
alt = toks[30000:30300]                      # a different 300-token tail for the rewound prompt
out = {}
r1 = post(dict(prompt=A, n_predict=24, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none'))
out['A'] = r1.get('tokens', r1.get('err'))
A2 = A[:16084] + alt                          # rewind 300 tokens, then 300 new ones
r2 = post(dict(prompt=A2, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none'))
out['A2'] = r2.get('tokens', r2.get('err')); out['A2_prompt_n'] = (r2.get('timings') or {}).get('prompt_n')
A3 = A2[:15000] + toks[40000:40100]           # a second, deeper rewind
r3 = post(dict(prompt=A3, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none'))
out['A3'] = r3.get('tokens', r3.get('err')); out['A3_prompt_n'] = (r3.get('timings') or {}).get('prompt_n')
json.dump(out, open(f'/home/douya/tests/mini/rewind-dm6_{dm}.json', 'w'))
print('dm=%s A2 prompt_n=%s A3 prompt_n=%s lens %s' % (dm, out['A2_prompt_n'], out['A3_prompt_n'], [len(out[k]) if isinstance(out[k], list) else out[k] for k in ('A', 'A2', 'A3')]))
PY
  grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/dm6_$dm.log | grep -v CORS | head -n 2 | cut -c1-140
  pkill -f "llama-server.*--port 8094"; sleep 3
done
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/rewind-dm6_1.json')); b = json.load(open('/home/douya/tests/mini/rewind-dm6_0.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a[k], b[k]
    if isinstance(x, list) and isinstance(y, list):
        p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))); print('%s: gpu-mask vs host-mask common prefix %d / %d %d' % (k, p, len(x), len(y)))
    else: print(k, 'error', x if not isinstance(x, list) else '', y if not isinstance(y, list) else '')
PY
echo "VAL_DM6 DONE $(date +%T)"
