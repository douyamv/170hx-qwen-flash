#!/bin/bash
# same as validate_par3.sh but with a unified KV (one stream, 3 sequences): the fork's single-stream fast paths stay on
T=/home/douya/tests; cd $T/mini || exit 1
until grep -q "P3 DONE" $T/validate_par3.out 2>/dev/null; do sleep 10; done
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new28/libnext-trace.so LIBDIR=lib-new28 NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_VERBOSE=1
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
echo "########## P3U: mini, --parallel 3 --kv-unified, ctx 18432 shared ##########"
bash mini_run4.sh p3u --ctx-size 18432 --parallel 3 --kv-unified --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' --spec-draft-backend-sampling --model $H/mini/Huihui-Qwen3.8-Flash-Next-mini4-noPLE.gguf --spec-draft-model $H/mini/mtp-huihui-mini4-shared-Q8_0-q4head.gguf || exit 1
grep -a "dmask cond\|device mask\|n_slots\|kv_unified" logs/p3u.log | head -n 4 | cut -c20-150
python3 - <<'PY'
import json, urllib.request, glob, threading, time
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens']
res = {}
def worker(i, off, n):
    t0 = time.time(); r = post(dict(prompt=toks[off:off + n], n_predict=64, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none', return_tokens=True))
    t = r.get('timings') or {}; res[i] = (round(time.time() - t0, 1), t.get('prompt_n'), t.get('predicted_n'), t.get('predicted_per_second'), t.get('draft_n'), t.get('draft_n_accepted'), r.get('err', ''), r.get('tokens', [])[:12])
for rnd in (1, 2):
    res.clear(); ths = [threading.Thread(target=worker, args=(i, off, n)) for i, (off, n) in enumerate([(0, 4000), (10000, 5000), (20000, 3000)])]
    [t.start() for t in ths]; [t.join() for t in ths]
    for i in sorted(res): print('round%d req %d: wall %ss prompt_n %s predicted %s tok/s %s draft %s/%s %s head %s' % ((rnd, i) + res[i]))
# reference: the same 3 prompts one after another (only one slot busy) -> compare heads with the concurrent run
res.clear()
for i, (off, n) in enumerate([(0, 4000), (10000, 5000), (20000, 3000)]): worker(i, off, n)
for i in sorted(res): print('serial req %d: predicted %s tok/s %s head %s' % (i, res[i][2], res[i][3], res[i][7]))
PY
grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT\|failed" logs/p3u.log | grep -av CORS | head -n 4 | cut -c1-160
pkill -f "llama-server.*--port 8094"; sleep 3; echo "P3U DONE $(date +%T)"
