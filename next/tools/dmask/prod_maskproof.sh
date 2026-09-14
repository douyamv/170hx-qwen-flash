#!/bin/bash
# production placement proof: full CUPTI trace of a short decode, count kq_mask_dev launches per device
P=/home/douya/qwen-3.8-next-opt; T=/home/douya/tests
: > $P/logs/profile.csv; rm -f $P/profile.flag $P/profile.flag.cpu
python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=600).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:4000]
post(dict(prompt=toks, n_predict=1, cache_prompt=True))
open('/home/douya/qwen-3.8-next-opt/profile.flag', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=30, cache_prompt=True, ignore_eos=True, temperature=0.0)); time.sleep(0.8); os.remove('/home/douya/qwen-3.8-next-opt/profile.flag'); time.sleep(2)
print('traced 4K-context decode: %s tokens, %.1f tok/s' % (r['timings']['predicted_n'], r['timings']['predicted_per_second']))
PY
echo "-- kq_mask_dev launches per device (column 2 = CUPTI deviceId):"; grep -a "^KERNEL" $P/logs/profile.csv | grep -a kq_mask_dev | cut -d, -f2 | sort | uniq -c | tr '\n' ' '; echo
echo "-- all kernels per device:"; grep -a "^KERNEL" $P/logs/profile.csv | cut -d, -f2 | sort | uniq -c | tr '\n' ' '; echo
echo "-- H2D memcpy count in the window:"; grep -a -c "^MEMCPY" $P/logs/profile.csv
echo "-- runtime API counts (cudaGraphLaunch = CUDA-graph-captured splits per window; 30 tokens):"; grep -a "^RUNTIME\|^API" $P/logs/profile.csv | awk -F'"' '{print $2}' | grep -E "cudaGraphLaunch|cudaStreamSynchronize|cudaMemcpyAsync|cudaLaunchKernel" | sort | uniq -c | tr '\n' ' '; echo
echo "-- record kinds:"; cut -d, -f1 $P/logs/profile.csv | sort | uniq -c | tr '\n' ' '; echo
cp $P/logs/profile.csv /mnt/slowdisk/tests-archive/traces/trace-v62-4k.csv && echo "saved /mnt/slowdisk/tests-archive/traces/trace-v62-4k.csv"
: > $P/logs/profile.csv
echo "MASKPROOF DONE $(date +%T)"
