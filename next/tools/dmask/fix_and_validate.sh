#!/bin/bash
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913
python3 $T/dmask/patch_dmask2.py || exit 1
export TMPDIR=/mnt/slowdisk/tmp
cd $S && systemd-run --user --scope -p MemoryMax=3000M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target llama -j1 > $T/dmask/build_fix.log 2>&1; echo "FIXBUILD_EXIT=$? $(date +%T)"; grep -nE " error" $T/dmask/build_fix.log | head -n 3
cd $T/mini && cp $S/build-sm80/bin/libllama.so.0.3.0 lib-new25/ && echo "lib-new25 updated"
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -v CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
echo "########## X1: NEXT_DEVICE_MASK=2, 1 GPU: equality + kernel proof + decode timeline ##########"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_VERBOSE=1 TRACER=$T/mini/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh x1 $A1 --spec-draft-backend-sampling || exit 1
for n in 2048 16384; do python3 stream_md5.py x1_$n $n 96 2>/dev/null | tail -1 | cut -c1-90; pref dm0_$n x1_$n; done
grep -a "device mask:" logs/x1.log | head -n 2 | cut -c1-120
python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:60]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:16384]
post(dict(prompt=toks, n_predict=1, cache_prompt=True))
open('/home/douya/tests/mini/profile.flag.cpu', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=200, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none')); t = r.get('timings', {})
time.sleep(0.5); os.remove('/home/douya/tests/mini/profile.flag.cpu'); time.sleep(0.5)
print('decode with GPU mask: %s tok/s' % t.get('predicted_per_second'))
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
post(dict(prompt=toks, n_predict=40, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none')); time.sleep(0.8); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
PY
echo "-- host timeline:"; python3 /home/douya/tests/host/an_cpu.py profile.csv 0 2>&1 | grep -E "steps|SET_INPUTS|QSA_CPU|DECODE_TARGET" | head -n 6
echo "-- mask kernels:"; grep -a "^KERNEL" profile.csv | awk -F'"' '{print $2}' | grep -iE "kq_mask_dev|qsa_mask" | sed 's/^_Z[0-9]*//' | cut -c1-30 | sort | uniq -c
echo "errors: $(errs x1)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## X2: NEXT_DEVICE_MASK=2, 2 GPUs ##########"
NEXT_DEVICE_MASK=2 TRACER=$T/mini/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new25 bash mini_run4.sh x2 $A2 --spec-draft-backend-sampling || exit 1
python3 stream_md5.py x2_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j2_2048 x2_2048
echo "errors: $(errs x2)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## X3: rewind equality (GPU mask level 2 vs host mask) ##########"
for dm in 2 0; do
  NEXT_DEVICE_MASK=$dm TRACER=$T/mini/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh x3_$dm $A1 --spec-draft-backend-sampling || exit 1
  python3 - $dm <<'PY'
import json, urllib.request, glob, sys
dm = sys.argv[1]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:60]}
toks = post({'content': ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))}, '/tokenize')['tokens']
A = toks[:16384]; out = {}
r = post(dict(prompt=A, n_predict=24, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none')); out['A'] = r.get('tokens', r.get('err'))
A2 = A[:16084] + toks[30000:30300]
r = post(dict(prompt=A2, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none')); out['A2'] = r.get('tokens', r.get('err')); out['A2_prompt_n'] = (r.get('timings') or {}).get('prompt_n')
A3 = A2[:12000] + toks[40000:40200]
r = post(dict(prompt=A3, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none')); out['A3'] = r.get('tokens', r.get('err')); out['A3_prompt_n'] = (r.get('timings') or {}).get('prompt_n')
json.dump(out, open(f'/home/douya/tests/mini/rewind-x3_{dm}.json', 'w')); print('dm=%s prompt_n A2=%s A3=%s' % (dm, out['A2_prompt_n'], out['A3_prompt_n']))
PY
  echo "errors: $(errs x3_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/rewind-x3_2.json')); b = json.load(open('/home/douya/tests/mini/rewind-x3_0.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a[k], b[k]
    if isinstance(x, list) and isinstance(y, list):
        p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))); print('%s: gpu-mask vs host-mask common prefix %d / %d %d' % (k, p, len(x), len(y)))
    else: print(k, 'error:', x if not isinstance(x, list) else '', y if not isinstance(y, list) else '')
PY
echo "FIX_VALIDATE DONE $(date +%T)"
