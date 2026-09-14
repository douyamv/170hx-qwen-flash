#!/bin/bash
# full q8a plan sweep: production dense shapes x B in {1,3,4,5} x R in 1..4 x s in {1,2,4}; CSV -> /home/douya/tests/prof/q8a_sweep.csv
while pgrep -f "sweep_q8a.sh" > /dev/null; do sleep 10; done
S=/home/douya/src/llama.cpp-flashnext-20260913
export LD_LIBRARY_PATH=$S/build-sm80/bin CUDA_VISIBLE_DEVICES=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
cd /home/douya/tests
OUT=/home/douya/tests/prof/q8a_sweep.csv; echo "K,N,B,R,s,us" > $OUT
for shape in "2560 6144" "2560 10240" "2560 12288" "6144 2560" "2560 2560" "2560 640" "10240 320" "320 10240" "2560 320" "640 2560"; do
  set -- $shape; K=$1; N=$2
  for B in 1 3 4 5; do
    for R in 1 2 3 4; do for s in 1 2 4; do
      r=$(NEXT_Q8A_RPW=$R NEXT_Q8A_SPLITK=$s timeout 120 ./q8a_test2 $K $N $B 2>/dev/null | tail -n 1)
      [ -n "$r" ] && echo "$K,$N,$B,$R,$s,$(echo $r | awk '{print $4}')" >> $OUT
    done; done
  done
  echo "shape $K x $N done $(date +%T)"
done
echo "SWEEP2 DONE $(date +%T)"
