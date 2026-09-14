#!/bin/bash
# is the residual difference a ggml-cuda fusion-decision effect? GGML_CUDA_DISABLE_FUSION=1 for both paths
# (lib-new28, no speculation, identical fresh histories): 16K stream + rewinds
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new28/libnext-trace.so LIBDIR=lib-new28 GGML_CUDA_DISABLE_FUSION=1
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-type none"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off; echo 4096 > opt/qsa_min_kv
for dm in 0 2; do
  echo "########## F$dm: dm=$dm, GGML_CUDA_DISABLE_FUSION=1, no speculation ##########"
  NEXT_DEVICE_MASK=$dm MINI_DEVS=$G3 bash mini_run4.sh f$dm $A1 || exit 1
  python3 stream_md5.py f${dm}_16384 16384 48 2>/dev/null | tail -1 | cut -c1-100
  python3 $T/rewind_sse.py f$dm 5
  echo "errors: $(errs f$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
python3 -c "
import json; a=json.load(open('stream-f2_16384.json')); b=json.load(open('stream-f0_16384.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('16K no-fusion stream f2 vs f0:', 'IDENTICAL' if a==b and len(a)>0 else 'DIFF', 'common prefix', p, '/', len(a), len(b))"
python3 $T/cmp_rewind.py f2 f0
echo "VALIDATE10 DONE $(date +%T)"
