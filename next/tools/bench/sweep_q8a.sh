#!/bin/bash
# per-shape sweep of the q8a plan space (rows per warp R x K-splits s); runs after after_v6.sh releases the build dir/GPU3
while pgrep -f "after_v6.sh|build_v6.sh" > /dev/null; do sleep 10; done
S=/home/douya/src/llama.cpp-flashnext-20260913
export LD_LIBRARY_PATH=$S/build-sm80/bin CUDA_VISIBLE_DEVICES=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
cd /home/douya/tests
echo "cases: 0:2560x6144 B1 | 1:2560x6144 B5 | 2:2560x10240 B5 | 3:2560x12288 B4 | 4:6144x2560 B5 | 5:2560x640 B5 | 6:10240x320 B5 | 7:320x10240 B5 | 8:2560x2560 B8 | 9:head B5(mmvq) | 10-12: q4_0"
printf "%-6s" "R,s"; for i in 0 1 2 3 4 5 6 7 8; do printf "%8s" "c$i"; done; echo
for R in 1 2 3 4; do for s in 1 2 4; do
  out=$(NEXT_Q8A_RPW=$R NEXT_Q8A_SPLITK=$s timeout 300 ./q8a_test 2>&1 | grep -E "^q8_0" | head -n 9 | awk '{for (i=1;i<=NF;i++) if ($i=="us") print $(i-1)}')
  printf "%-6s" "$R,$s"; for v in $out; do printf "%8.1f" "$v"; done; echo
done; done
echo "== v5 reference plan (R=2, s=1 for K<4096; K>=4096 -> s to reach 2048 warps: case4 s=2, case7 s=8) =="
echo "== correctness (auto plan) =="; timeout 300 ./q8a_test 2>&1 | grep -E "^q8_0|^q4_0" | cut -c1-110 | head -n 13
echo "SWEEP_Q8A DONE $(date +%T)"
