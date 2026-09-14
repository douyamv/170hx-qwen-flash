#!/bin/bash
# after sweep2: build the measured q8a plan table into q8a.cu, rebuild, verify, snapshot lib-new20, run the mini validation
while pgrep -f "sweep2.sh|sweep_q8a.sh" > /dev/null; do sleep 15; done
S=/home/douya/src/llama.cpp-flashnext-20260913; T=/home/douya/tests
cd $T
echo "== plan table from the sweep =="
python3 $T/gen_table.py $T/prof/q8a_sweep.csv > $T/prof/q8a_table.txt 2>&1; cat $T/prof/q8a_table.txt
python3 $T/fill_table.py $T/q8a_v21.cu $T/prof/q8a_table.txt
cp $T/q8a_v21.cu $S/ggml/src/ggml-cuda/q8a.cu
export TMPDIR=/mnt/slowdisk/tmp
cd $S && systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-cuda -j1 > $T/q8a21-build.log 2>&1
echo "Q8A21BUILD_EXIT=$? $(date +%T)"; grep -nE " error" $T/q8a21-build.log | head -n 5
export LD_LIBRARY_PATH=$S/build-sm80/bin CUDA_VISIBLE_DEVICES=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
cd $T
echo "== q8a_test with the table (auto) =="; NEXT_Q8A_VERBOSE=1 timeout 300 ./q8a_test 2>&1 | grep -E "^q8_0|^q4_0|^q8a:" | cut -c1-120
echo "== q8a_test v5 rule (forced R=2) =="; NEXT_Q8A_RPW=2 timeout 300 ./q8a_test 2>&1 | grep -E "^q8_0|^q4_0" | cut -c1-60
echo "== moea_test sanity =="; timeout 300 ./moea_test 2>&1 | grep -E "^q[45]_" | cut -c1-120
echo "== snapshot lib-new20 =="
mkdir -p $T/mini/lib-new20 && cp -a $T/mini/lib-new17/. $T/mini/lib-new20/ && cp $S/build-sm80/bin/libggml-cuda.so.0.22.0 $T/mini/lib-new20/libggml-cuda.so.0.22.0 && cp $S/build-sm80/bin/libggml-base.so.0.22.0 $T/mini/lib-new20/libggml-base.so.0.22.0
md5sum $T/mini/lib-new20/libggml-cuda.so.0.22.0 $T/mini/lib-new20/libggml-base.so.0.22.0 $S/build-sm80/bin/libggml-cuda.so.0.22.0 $S/build-sm80/bin/libggml-base.so.0.22.0 | cut -c1-8,34-
echo "== validate_v6 (mini) =="; LIB=lib-new20 bash $T/mini/validate_v6.sh 2>&1 | cut -c1-200
echo "FINALIZE_V6 DONE $(date +%T)"
