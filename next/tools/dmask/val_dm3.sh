#!/bin/bash
# DM3: prove the GPU mask path is active (CUPTI kernel names) and measure decode-phase set_inputs with a long generation
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_DMASK DONE" /home/douya/tests/dmask/val_dmask.out 2>/dev/null; do sleep 10; done
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
for dm in 1 0; do
  rm -f profile.flag profile.flag.cpu; : > profile.csv
  NEXT_DEVICE_MASK=$dm TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm3_$dm $A1 --spec-draft-backend-sampling || exit 1
  # warm the 16K prompt, then a long ignore_eos generation while the markers-only flag is on
  python3 - <<PY
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=600).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:16384]
post(dict(prompt=toks, n_predict=1, cache_prompt=True))
open('/home/douya/tests/mini/profile.flag.cpu', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=200, cache_prompt=True, ignore_eos=True, temperature=0.0)); t = r['timings']
time.sleep(0.5); os.remove('/home/douya/tests/mini/profile.flag.cpu'); time.sleep(0.5)
print('NEXT_DEVICE_MASK=$dm decode: %.1f tok/s (%d tokens)' % (t['predicted_per_second'], t['predicted_n']))
# CUPTI kernel trace for 3 s to list kernel names
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
post(dict(prompt=toks, n_predict=60, cache_prompt=True, ignore_eos=True, temperature=0.0)); time.sleep(0.5); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.0)
PY
  echo "-- host timeline (decode, NEXT_DEVICE_MASK=$dm):"; python3 /home/douya/tests/host/an_cpu.py profile.csv 0 2>&1 | grep -E "steps|SET_INPUTS|QSA_CPU|DECODE_TARGET|GRAPH_COMPUTE|SYNC " | head -n 8
  echo "-- mask kernels in the CUPTI trace:"; grep -a "KERNEL" profile.csv | grep -aoE "kq_mask_dev[^,\"]*|set_input|k_bin_bcast[^\"]{0,10}" | sort | uniq -c | head -n 3
  pkill -f "llama-server.*--port 8094"; sleep 3
done
echo "VAL_DM3 DONE $(date +%T)"
