#!/bin/bash
# device-mask validation on the mini: NEXT_DEVICE_MASK=1 must be bit-identical to the host mask, 1 GPU and 2 GPUs, rewind OK
T=/home/douya/tests/mini; S=/home/douya/src/llama.cpp-flashnext-20260913; cd $T
while ! grep -q "BUILD_DMASK DONE" /home/douya/tests/dmask/build_dmask.out 2>/dev/null; do sleep 10; done
grep -q "DMASKBUILD_EXIT=0" /home/douya/tests/dmask/build_dmask.out || { echo "BUILD FAILED"; exit 1; }
rm -rf lib-new25 && mkdir lib-new25 && cp -a lib-new22/. lib-new25/ && cp $S/build-sm80/bin/libggml-base.so.0.22.0 $S/build-sm80/bin/libggml-cuda.so.0.22.0 $S/build-sm80/bin/libllama.so.0.3.0 lib-new25/
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -v CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
echo "########## DM0: lib-new25, host mask (reference) ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm0 $A1 --spec-draft-backend-sampling || exit 1
for n in 2048 16384; do python3 stream_md5.py dm0_$n $n 96 2>/dev/null | tail -1 | cut -c1-90; done; pref mk22b_2048 dm0_2048; echo "errors: $(errs dm0)"
echo "########## DM1: lib-new25, NEXT_DEVICE_MASK=1 (1 GPU) ##########"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=1 TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm1 $A1 --spec-draft-backend-sampling || exit 1
python3 stream_md5.py dm1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref dm0_2048 dm1_2048
touch profile.flag.cpu; sleep 0.3; python3 stream_md5.py dm1_16384 16384 96 2>/dev/null | tail -1 | cut -c1-90; sleep 0.5; rm -f profile.flag.cpu; sleep 0.5; pref dm0_16384 dm1_16384
python3 mini_ckpt_test.py dm1 2>&1 | tail -3
grep -a "attn_kq_mask_dev\|cache_cellpos" logs/dm1.log | head -n 2 | cut -c1-120
python3 /home/douya/tests/host/an_cpu.py profile.csv 0 2>&1 | grep -E "steps|SET_INPUTS|QSA_CPU|DECODE_TARGET|GRAPH_COMPUTE" | head -n 8
echo "errors: $(errs dm1)"
echo "########## DM2: NEXT_DEVICE_MASK=1, 2 GPUs (CUDA0 + GPU3) ##########"
NEXT_DEVICE_MASK=1 TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new25 bash mini_run4.sh dm2 $A2 --spec-draft-backend-sampling || exit 1
python3 stream_md5.py dm2_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j2_2048 dm2_2048; pref k1_2048 dm2_2048
python3 mini_ckpt_test.py dm2 2>&1 | tail -3; echo "errors: $(errs dm2)"
pkill -f "llama-server.*--port 8094"; echo "VAL_DMASK DONE $(date +%T)"
