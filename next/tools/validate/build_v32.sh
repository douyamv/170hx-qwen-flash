#!/bin/bash
# after val_e4: moea v3.2 (fusion-invariant), rebuild, lib-new21, then H pair (1 vs 2 GPUs, moea on, sort on) again
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913
while ! grep -q "VAL_E4 DONE\|DIED\|TIMEOUT" $T/mini/logs/val_e4.out 2>/dev/null; do sleep 10; done
cp $T/moea_v32.cu $S/ggml/src/ggml-cuda/moea.cu
export TMPDIR=/mnt/slowdisk/tmp
cd $S && systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-cuda -j1 > $T/moea32-build.log 2>&1; echo "MOEA32BUILD_EXIT=$? $(date +%T)"; grep -nE " error" $T/moea32-build.log | head -n 3
mkdir -p $T/mini/lib-new21 && cp -a $T/mini/lib-new20/. $T/mini/lib-new21/ && cp $S/build-sm80/bin/libggml-cuda.so.0.22.0 $T/mini/lib-new21/libggml-cuda.so.0.22.0
export LD_LIBRARY_PATH=$S/build-sm80/bin CUDA_VISIBLE_DEVICES=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
cd $T && echo "== moea_test (v3.2) =="; timeout 300 ./moea_test 2>&1 | grep -E "^q[45]_" | cut -c1-120
cd $T/mini
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
rm -f opt/moea_off
echo "########## J1: 1 GPU, v3.2, moea ON, nosort ON ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new21 bash mini_run3.sh j1 $A1 || exit 1; for n in 2048 16384; do python3 stream_md5.py j1_$n $n 96 2>/dev/null | tail -1 | cut -c1-90; done
echo "########## J2: 2 GPUs, v3.2, moea ON, nosort ON ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new21 bash mini_run3.sh j2 $A2 || exit 1; for n in 2048 16384; do python3 stream_md5.py j2_$n $n 96 2>/dev/null | tail -1 | cut -c1-90; pref j1_$n j2_$n; done
python3 mini_ckpt_test.py j2 2>&1 | tail -3
pkill -f "llama-server.*--port 8094"; echo "BUILD_V32 DONE $(date +%T)"
