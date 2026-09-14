#!/bin/bash
# lib-new30 = lib-new29 + incremental cell_pos uploads: W1 one slot with rewinds (seq_rm paths) under DM_CHECK, W2 three
# slots concurrent under DM_CHECK (any stale cell shows as a mismatch)
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913; cd $T/mini || exit 1
rm -rf lib-new30 && cp -a lib-new29 lib-new30 && cp -a $S/build-sm80/bin/libllama.so* lib-new30/ && echo "lib-new30 ready"
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new30/libnext-trace.so LIBDIR=lib-new30 NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_CHECK=1
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
M="--model $H/mini/Huihui-Qwen3.8-Flash-Next-mini4-noPLE.gguf --spec-draft-model $H/mini/mtp-huihui-mini4-shared-Q8_0-q4head.gguf"
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT\|failed" logs/$1.log | grep -av CORS | head -n 3 | cut -c1-160; }
dm() { echo "DM_CHECK $1: ubatches $(grep -ac '^DM_CHECK ubatch' logs/$1.log) | $(grep -a '^DM_CHECK ubatch' logs/$1.log | tail -n 1 | grep -o 'bad ubatches so far [0-9]*') | mismatch lines: $(grep -a 'DM_CHECK' logs/$1.log | grep -avc ' 0 mismatching')"; }
touch opt/spec_adaptive_off; echo 4096 > opt/qsa_min_kv
echo "########## W1: 1 slot, drafts on, rewinds A/A2/A3 + 2K stream, DM_CHECK ##########"
MINI_DEVS=$G3 bash mini_run5.sh w1 $A1 $M || exit 1
python3 stream_md5.py w1_2048 2048 48 2>/dev/null | tail -1 | cut -c1-100
python3 $T/rewind_sse.py w1 3
dm w1; echo "errors: $(errs w1)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## W2: 3 slots, concurrent rounds + rewinds on slot 0, DM_CHECK ##########"
MINI_DEVS=$G3 bash mini_run5.sh w2 $A1 $M --ctx-size 18432 --parallel 3 || exit 1
python3 - <<'PY'
import json, urllib.request, glob, threading, time
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens']
P = [(0, 5000), (12000, 5200), (30000, 4800)]; res = {}
def worker(i, off, n, npred=48):
    r = post(dict(prompt=toks[off:off + n], n_predict=npred, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none', return_tokens=True))
    t = r.get('timings') or {}; res[i] = (t.get('prompt_n'), t.get('predicted_n'), r.get('err', ''), r.get('tokens', [])[:8])
for rnd in (1, 2, 3):
    res.clear(); ths = [threading.Thread(target=worker, args=(i, off, n)) for i, (off, n) in enumerate(P)]
    [t.start() for t in ths]; [t.join() for t in ths]
    for i in sorted(res): print('w2 round%d req %d: prompt_n %s predicted %s %s head %s' % ((rnd, i) + res[i]))
    if rnd == 2: P = [(0, 4700), (12000, 5200), (30000, 4900)]  # rewinds on two slots
PY
dm w2; echo "errors: $(errs w2)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "VALIDATE_INC DONE $(date +%T)"
