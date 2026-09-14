#!/bin/bash
# DM5: is the GPU mask kernel really launched? full-generation CUPTI kernel trace with NEXT_DEVICE_MASK=1
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_DM4 DONE" /home/douya/tests/dmask/val_dm4.out 2>/dev/null; do sleep 10; done
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=1 TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm5 $A1 --spec-draft-backend-sampling || exit 1
python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:4096]
print('warm:', str(post(dict(prompt=toks, n_predict=1, cache_prompt=True)))[:60])
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.5)
r = post(dict(prompt=toks, n_predict=40, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none'))
print('gen:', str(r.get('timings', r))[:80])
time.sleep(1.0); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
PY
echo "-- kernel names containing mask/qsa/hc in the trace:"; grep -a "^KERNEL" profile.csv | awk -F'"' '{print $2}' | grep -iE "mask|qsa|hc_|kq" | sed 's/^_Z[0-9]*//' | cut -c1-40 | sort | uniq -c | sort -rn | head -n 8
echo "-- total kernels:"; grep -ac "^KERNEL" profile.csv
grep -a "CUDA error\|ERR\|abort" logs/dm5.log | grep -v CORS | head -n 2 | cut -c1-140
pkill -f "llama-server.*--port 8094"; echo "VAL_DM5 DONE $(date +%T)"
