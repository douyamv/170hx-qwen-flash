#!/bin/bash
# lib-new27 + set_output: valid DM_CHECK readback (C4 drafts on: 2K stream + rewinds; C5 no speculation 16K + rewinds)
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new27/libnext-trace.so LIBDIR=lib-new27
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
for cfg in "c4 " "c5 --spec-type none"; do
  set -- $cfg; tag=$1; shift
  echo "########## $tag: 1 GPU, dm=2 + NEXT_DEVICE_MASK_CHECK=1 (masks kept as outputs) $* ##########"
  NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_CHECK=1 MINI_DEVS=$G3 bash mini_run4.sh $tag $A1 "$@" || exit 1
  python3 stream_md5.py ${tag}_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100
  python3 $T/rewind_sse.py $tag 3
  echo "-- DM_CHECK ($tag): ubatches checked / bad:"; grep -a -c "^DM_CHECK ubatch" logs/$tag.log; grep -a "^DM_CHECK ubatch" logs/$tag.log | tail -n 1 | cut -c1-160
  grep -a "DM_CHECK" logs/$tag.log | grep -av " 0 mismatching" | head -n 6 | cut -c1-200
  echo "errors: $(errs $tag)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
echo "VALIDATE6 DONE $(date +%T)"
