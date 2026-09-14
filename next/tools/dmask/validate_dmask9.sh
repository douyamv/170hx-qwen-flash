#!/bin/bash
# lib-new28 = lib-new26 + early-expanded mask ops (+ inert DM_CHECK code): equality vs the host-mask references
# c2_0 (no speculation, 16K stream + rewinds) and s6 (drafts on, fresh history rewinds), plus the kernel-sequence check
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new28/libnext-trace.so LIBDIR=lib-new28
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off; echo 4096 > opt/qsa_min_kv
echo "########## E1: lib-new28 dm=2, no speculation: 16K stream + rewinds vs c2_0 (host mask) ##########"
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh e1 $A1 --spec-type none || exit 1
python3 stream_md5.py e1_16384 16384 48 2>/dev/null | tail -1 | cut -c1-100
python3 $T/rewind_sse.py e1 5
echo "errors: $(errs e1)"; pkill -f "llama-server.*--port 8094"; sleep 3
python3 -c "
import json; a=json.load(open('stream-e1_16384.json')); b=json.load(open('stream-c2_0_16384.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('16K no-spec stream e1 vs c2_0:', 'IDENTICAL' if a==b and len(a)>0 else 'DIFF', 'common prefix', p, '/', len(a), len(b))"
python3 $T/cmp_rewind.py e1 c2_0
echo "########## E2: lib-new28 dm=2, drafts on, fresh history: rewinds vs s6 / r4_0 (host mask) ##########"
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh e2 $A1 --spec-draft-backend-sampling || exit 1
python3 $T/rewind_sse.py e2 3
echo "errors: $(errs e2)"; pkill -f "llama-server.*--port 8094"; sleep 3
python3 $T/cmp_rewind.py e2 s6; python3 $T/cmp_rewind.py e2 r4_0
echo "########## E3: lib-new28 dm=2, no speculation: kernel sequence vs kseq_dm0 ##########"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh e3 $A1 --spec-type none || exit 1
python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:2048]
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=12, cache_prompt=False, ignore_eos=True, temperature=0.0, reasoning_format='none')); time.sleep(0.8); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
print('traced: prompt_n', (r.get('timings') or {}).get('prompt_n'), 'predicted', (r.get('timings') or {}).get('predicted_n'), r.get('err', ''))
PY
grep -a "^KERNEL" profile.csv | awk -F'"' '{print $2}' | sed 's/^_Z[0-9]*//' > kseq_e3.txt; grep -av kq_mask_dev kseq_e3.txt > kseq_e3_nomask.txt
echo "kernels: $(wc -l < kseq_e3.txt) (kq_mask_dev: $(grep -ac kq_mask_dev kseq_e3.txt))"; cmp kseq_dm0.txt kseq_e3_nomask.txt && echo "KERNEL SEQUENCE IDENTICAL to host mask (ignoring kq_mask_dev)"
echo "errors: $(errs e3)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "VALIDATE9 DONE $(date +%T)"
