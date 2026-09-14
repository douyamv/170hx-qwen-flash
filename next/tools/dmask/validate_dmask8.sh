#!/bin/bash
# is the discrepancy in the compact QSA decode path? dense path forced via opt/qsa_min_kv=100000000 (runtime toggle),
# no speculation, dm=2 vs dm=0 (lib-new26): 16K stream + rewinds (CPU sampling, logprobs)
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new26/libnext-trace.so LIBDIR=lib-new26
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-type none"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
OLDMIN=$(cat opt/qsa_min_kv 2>/dev/null); echo 100000000 > opt/qsa_min_kv; echo "qsa_min_kv: $OLDMIN -> $(cat opt/qsa_min_kv)"
for dm in 2 0; do
  echo "########## D$dm: dm=$dm, dense QSA path (qsa_min_kv=1e8), no speculation ##########"
  NEXT_DEVICE_MASK=$dm MINI_DEVS=$G3 bash mini_run4.sh d$dm $A1 || exit 1
  python3 stream_md5.py d${dm}_16384 16384 48 2>/dev/null | tail -1 | cut -c1-100
  python3 $T/rewind_sse.py d$dm 5
  echo "errors: $(errs d$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
echo "${OLDMIN:-4096}" > opt/qsa_min_kv; echo "qsa_min_kv restored: $(cat opt/qsa_min_kv)"
python3 -c "
import json; a=json.load(open('stream-d2_16384.json')); b=json.load(open('stream-d0_16384.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('16K dense stream d2 vs d0: common prefix', p, '/', len(a), len(b), 'IDENTICAL' if a==b and len(a)>0 else 'DIFF')"
python3 $T/cmp_rewind.py d2 d0
echo "VALIDATE8 DONE $(date +%T)"
