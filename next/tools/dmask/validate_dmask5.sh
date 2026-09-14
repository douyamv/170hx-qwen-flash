#!/bin/bash
# lib-new27 (= lib-new26 + NEXT_DEVICE_MASK_CHECK debug readback): run the rewind scenario with the check on and count
# mask mismatches; also with drafts off. Diagnosis only — lib-new27 is not a deployment candidate.
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new27/libnext-trace.so LIBDIR=lib-new27
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
echo "########## C1: 1 GPU, dm=2 + NEXT_DEVICE_MASK_CHECK=1, drafts on: streams + rewinds ##########"
NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_CHECK=1 MINI_DEVS=$G3 bash mini_run4.sh c1 $A1 || exit 1
python3 stream_md5.py c1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100
python3 $T/rewind_sse.py c1 3
echo "-- DM_CHECK summary (c1):"; grep -a -c "^DM_CHECK ubatch" logs/c1.log; grep -a "DM_CHECK" logs/c1.log | grep -av " 0 mismatching" | head -n 12 | cut -c1-200; echo "-- last DM_CHECK line:"; grep -a "^DM_CHECK ubatch" logs/c1.log | tail -n 1 | cut -c1-160
echo "errors: $(errs c1)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## C2/C3: 1 GPU, NO speculation (--spec-type none): dm=2 (+CHECK) vs dm=0, probs at every position ##########"
for dm in 2 0; do
  NEXT_DEVICE_MASK=$dm NEXT_DEVICE_MASK_CHECK=1 MINI_DEVS=$G3 bash mini_run4.sh c2_$dm $A1 --spec-type none || exit 1
  python3 stream_md5.py c2_${dm}_16384 16384 48 2>/dev/null | tail -1 | cut -c1-100
  python3 $T/rewind_sse.py c2_$dm 5
  echo "-- DM_CHECK summary (c2_$dm):"; grep -a -c "^DM_CHECK ubatch" logs/c2_$dm.log; grep -a "DM_CHECK" logs/c2_$dm.log | grep -av " 0 mismatching" | head -n 8 | cut -c1-200
  echo "errors: $(errs c2_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
python3 -c "
import json; a=json.load(open('stream-c2_2_16384.json')); b=json.load(open('stream-c2_0_16384.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('16K no-spec stream c2_2 vs c2_0: common prefix', p, '/', len(a), len(b), 'IDENTICAL' if a==b and len(a)>0 else 'DIFF')"
python3 $T/cmp_rewind.py c2_2 c2_0
python3 $T/cmp_rewind.py c1 s5
echo "VALIDATE5 DONE $(date +%T)"
