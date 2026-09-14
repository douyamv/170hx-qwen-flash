#!/bin/bash
# experiment: software-pipelined q8a_gemv (q8a_pf.cu) vs the v2.1 table build; restores v2.1 afterwards
while pgrep -f "finalize_v6.sh|sweep2.sh" > /dev/null; do sleep 15; done
S=/home/douya/src/llama.cpp-flashnext-20260913; T=/home/douya/tests
export TMPDIR=/mnt/slowdisk/tmp LD_LIBRARY_PATH=$S/build-sm80/bin CUDA_VISIBLE_DEVICES=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
cd $T
echo "== baseline (v2.1 table build) =="; for sh in "2560 6144 5" "2560 10240 5" "2560 12288 4" "6144 2560 5" "2560 6144 1" "2560 2560 3"; do ./q8a_test2 $sh; done
python3 - <<'PY'
# carry the measured table into the prefetch variant
s = open('/home/douya/tests/q8a_pf.cu').read(); t = open('/home/douya/tests/q8a_v21.cu').read()
a = t.index('//Q8A_TABLE_BEGIN'); b = t.index('//Q8A_TABLE_END'); table = t[a:b]
a2 = s.index('//Q8A_TABLE_BEGIN'); b2 = s.index('//Q8A_TABLE_END'); s = s[:a2] + table + s[b2:]
open('/home/douya/tests/q8a_pf.cu', 'w').write(s); print('table carried over')
PY
cp $T/q8a_pf.cu $S/ggml/src/ggml-cuda/q8a.cu
cd $S && systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-cuda -j1 > $T/q8apf-build.log 2>&1; echo "PFBUILD_EXIT=$?"; grep -nE " error" $T/q8apf-build.log | head -n 3
cd $T
echo "== prefetch variant =="; for sh in "2560 6144 5" "2560 10240 5" "2560 12288 4" "6144 2560 5" "2560 6144 1" "2560 2560 3"; do ./q8a_test2 $sh; done
echo "== prefetch correctness =="; timeout 300 ./q8a_test 2>&1 | grep -E "^q8_0" | cut -c1-110 | head -n 5
cuobjdump --dump-resource-usage $S/build-sm80/bin/libggml-cuda.so.0.22.0 2>/dev/null | grep -A1 -E "q8a_gemvILi5ELi2E" | grep REG | sed 's/^ *//' | cut -c1-40
echo "== restore v2.1 =="; cp $T/q8a_v21.cu $S/ggml/src/ggml-cuda/q8a.cu
cd $S && systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 -q taskset -c 20-39 nice -n 10 cmake --build build-sm80 --target ggml-cuda -j1 > $T/q8a21-build2.log 2>&1; echo "RESTORE_EXIT=$?"
md5sum $S/build-sm80/bin/libggml-cuda.so.0.22.0 $T/mini/lib-new20/libggml-cuda.so.0.22.0 | cut -c1-8,34-
echo "EXP_PF DONE $(date +%T)"
