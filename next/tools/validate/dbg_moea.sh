#!/bin/bash
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913
cp $T/moea_dbg.cu $S/ggml/src/ggml-cuda/moea.cu
export TMPDIR=/mnt/slowdisk/tmp
cd $S && systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-cuda -j1 > $T/moeadbg-build.log 2>&1; echo "DBGBUILD_EXIT=$?"; grep -nE " error" $T/moeadbg-build.log | head -n 3
mkdir -p $T/mini/lib-dbg && cp -a $T/mini/lib-new21/. $T/mini/lib-dbg/ && cp $S/build-sm80/bin/libggml-cuda.so.0.22.0 $T/mini/lib-dbg/libggml-cuda.so.0.22.0
cd $T/mini
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
NEXT_MOEA_VERBOSE=1 NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-dbg bash mini_run3.sh d1 $A1 || exit 1; python3 stream_md5.py d1_2048 2048 32 2>/dev/null | tail -1 | cut -c1-80
NEXT_MOEA_VERBOSE=1 NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-dbg bash mini_run3.sh d2 $A2 || exit 1; python3 stream_md5.py d2_2048 2048 32 2>/dev/null | tail -1 | cut -c1-80
pkill -f "llama-server.*--port 8094"
echo "== moea decisions, 1 GPU =="; grep -a "^moea:\|moea: " logs/d1.log | sed 's/.*moea: /moea: /' | sort | uniq -c | cut -c1-230
echo "== moea decisions, 2 GPUs =="; grep -a "^moea:\|moea: " logs/d2.log | sed 's/.*moea: /moea: /' | sort | uniq -c | cut -c1-230
echo "DBG_MOEA DONE $(date +%T)"
