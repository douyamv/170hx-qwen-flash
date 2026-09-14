#!/bin/bash
# lib-new29 = lib-new28 + multi-stream fast paths. V1 single-slot regression, V2 3 slots with the device mask readback
# check and the fast paths on, V3 the same prompts with the fast paths off (reference), plus a serial cache-hit round.
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913; cd $T/mini || exit 1
rm -rf lib-new29 && cp -a lib-new28 lib-new29 && cp -a $S/build-sm80/bin/libggml-base.so* $S/build-sm80/bin/libggml-cuda.so* $S/build-sm80/bin/libllama.so* lib-new29/ && echo "lib-new29 ready $(ls -la --time-style=+%T lib-new29/libggml-cuda.so.0.22.0 | awk '{print $6}')"
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new29/libnext-trace.so LIBDIR=lib-new29
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
M="--model $H/mini/Huihui-Qwen3.8-Flash-Next-mini4-noPLE.gguf --spec-draft-model $H/mini/mtp-huihui-mini4-shared-Q8_0-q4head.gguf"
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT\|failed" logs/$1.log | grep -av CORS | head -n 3 | cut -c1-160; }
touch opt/spec_adaptive_off; echo 4096 > opt/qsa_min_kv
echo "########## V1: 1 slot, dm=2, lib-new29 (regression vs lib-new28 = hh1: 992bbd351c) ##########"
NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_CHECK=1 MINI_DEVS=$G3 bash mini_run4.sh v1 $A1 $M || exit 1
python3 stream_md5.py v1_2048 2048 48 2>/dev/null | tail -1 | cut -c1-100
python3 stream_md5.py v1_16384 16384 24 2>/dev/null | tail -1 | cut -c1-100
echo "DM_CHECK: $(grep -ac '^DM_CHECK ubatch' logs/v1.log) ubatches, bad: $(grep -a '^DM_CHECK ubatch' logs/v1.log | tail -n 1 | grep -o 'bad ubatches so far [0-9]*')"; echo "errors: $(errs v1)"; pkill -f "llama-server.*--port 8094"; sleep 3
CONC='
import json, urllib.request, glob, threading, time, sys
tag = sys.argv[1]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path="/completion"):
    r = urllib.request.Request("http://127.0.0.1:8094" + path, data=json.dumps(o).encode(), headers={"Content-Type": "application/json"})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {"err": str(e)[:80]}
text = "".join(open(p, errors="ignore").read() for p in sorted(glob.glob("/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp")))
toks = post({"content": text}, "/tokenize")["tokens"]
P = [(0, 5000), (12000, 5200), (30000, 4800)]
res = {}; out = {}
def worker(i, off, n):
    t0 = time.time(); r = post(dict(prompt=toks[off:off + n], n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format="none", return_tokens=True, stream=False))
    t = r.get("timings") or {}; res[i] = (round(time.time() - t0, 1), t.get("prompt_n"), t.get("predicted_n"), t.get("predicted_per_second"), r.get("err", ""), r.get("tokens", []))
for rnd in (1, 2):
    res.clear(); ths = [threading.Thread(target=worker, args=(i, off, n)) for i, (off, n) in enumerate(P)]
    [t.start() for t in ths]; [t.join() for t in ths]
    for i in sorted(res): print("%s round%d req %d: wall %ss prompt_n %s predicted %s tok/s %s %s head %s" % ((tag, rnd, i) + res[i][:5] + (res[i][5][:10],)))
    if rnd == 1: out = {i: res[i][5] for i in res}
json.dump(out, open("/home/douya/tests/mini/conc-%s.json" % tag, "w"))
'
echo "########## V2: 3 slots (one stream each), dm=2 + DM_CHECK, fast paths ON ##########"
NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_CHECK=1 MINI_DEVS=$G3 bash mini_run4.sh v2 $A1 $M --ctx-size 18432 --parallel 3 || exit 1
grep -a "device mask:\|device_mask_ok" logs/v2.log | head -n 2 | cut -c20-140
python3 -c "$CONC" v2
echo "DM_CHECK: $(grep -ac '^DM_CHECK ubatch' logs/v2.log) ubatches, bad: $(grep -a '^DM_CHECK ubatch' logs/v2.log | tail -n 1 | grep -o 'bad ubatches so far [0-9]*')"; grep -a "DM_CHECK" logs/v2.log | grep -av " 0 mismatching" | head -n 4 | cut -c1-180
echo "errors: $(errs v2)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## V3: 3 slots, fast paths OFF (qsa_min_kv=1e8, host mask) - reference ##########"
echo 100000000 > opt/qsa_min_kv
NEXT_DEVICE_MASK=0 MINI_DEVS=$G3 bash mini_run4.sh v3 $A1 $M --ctx-size 18432 --parallel 3 || exit 1
python3 -c "$CONC" v3
echo 4096 > opt/qsa_min_kv
echo "errors: $(errs v3)"; pkill -f "llama-server.*--port 8094"; sleep 3
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/conc-v2.json')); b = json.load(open('/home/douya/tests/mini/conc-v3.json'))
for i in sorted(a, key=int):
    x, y = a[i], b.get(i, [])
    p = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    print('req %s: fast-paths-on vs off common prefix %d / %d %d %s' % (i, p, len(x), len(y), 'IDENTICAL' if x == y and x else 'DIFF'))
PY
echo "VALIDATE_MS DONE $(date +%T)"
