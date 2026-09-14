#!/bin/bash
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913; D=$T/flashnext-udq4-20260913/deployment-v2
cd $S && for f in src/llama-context.cpp tools/server/server-context.cpp common/speculative.cpp; do cp -n $f $T/opt-20260913/orig6/$(basename $f).v6; done
cp -n $D/next-trace.cpp $T/opt-20260913/orig6/next-trace.cpp.v6
python3 $T/host/patch_spec.py && python3 $T/host/patch_tracer.py || { echo PATCH_FAILED; exit 1; }
grep -c 'llama_trace_local trace_' $S/src/llama-context.cpp $S/tools/server/server-context.cpp $S/common/speculative.cpp
echo "== tracer build =="; CUPTI_INC=$(dirname $(find /usr/include /usr/local/cuda* -name cupti.h 2>/dev/null | head -n 1)); echo "cupti.h in $CUPTI_INC"
ldd /home/douya/qwen-3.8-next-opt/libnext-trace.so | grep -i cupti | head -n 1
g++ -O2 -shared -fPIC -std=c++17 -o $T/host/libnext-trace.so $D/next-trace.cpp -I$CUPTI_INC -lcupti -lpthread 2>&1 | head -n 5 && ls -la $T/host/libnext-trace.so && nm -D $T/host/libnext-trace.so | grep -E "next_trace_time|next_trace_cpu"
export TMPDIR=/mnt/slowdisk/tmp
echo "== llama + common + server build $(date +%T) =="
systemd-run --user --scope -p MemoryMax=3000M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target llama llama-common llama-server -j1 > $T/host/build.log 2>&1; echo "MARKERBUILD_EXIT=$? $(date +%T)"; grep -nE " error|Error " $T/host/build.log | head -n 5
mkdir -p $T/mini/lib-new22 && cp -a $T/mini/lib-new21/. $T/mini/lib-new22/ && cp $S/build-sm80/bin/libllama.so.0.3.0 $T/mini/lib-new22/ && cp $S/build-sm80/bin/libllama-common.so.0.3.0 $T/mini/lib-new22/ && cp $S/build-sm80/bin/llama-server $T/mini/lib-new22/llama-server && cp $T/host/libnext-trace.so $T/mini/lib-new22/libnext-trace.so && ls -la $T/mini/lib-new22 | grep -E "llama-server|libllama|libnext" | cut -c30-120
echo "BUILD_MARKERS DONE $(date +%T)"
