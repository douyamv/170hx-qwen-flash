#!/bin/bash
T=/home/douya/tests; S=/home/douya/src/llama.cpp-flashnext-20260913
mkdir -p $T/opt-20260913/orig7; cd $S && for f in ggml/src/ggml-qsa.h ggml/src/ggml-qsa.c ggml/src/ggml-cuda/qsa.cu src/llama-kv-cache.h src/llama-kv-cache.cpp src/llama-graph.h src/llama-graph.cpp src/models/qwen4exp.cpp; do cp -n $f $T/opt-20260913/orig7/$(basename $f).v61 2>/dev/null; done
python3 $T/dmask/patch_dmask.py || { echo PATCH_FAILED; exit 1; }
grep -n "include" $S/src/llama-graph.cpp | head -n 5
export TMPDIR=/mnt/slowdisk/tmp
echo "== build ggml-base ggml-cuda llama $(date +%T) =="
systemd-run --user --scope -p MemoryMax=3000M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-base ggml-cuda llama -j1 > $T/dmask/build.log 2>&1; echo "DMASKBUILD_EXIT=$? $(date +%T)"; grep -nE " error|error:" $T/dmask/build.log | head -n 8
echo "BUILD_DMASK DONE $(date +%T)"
