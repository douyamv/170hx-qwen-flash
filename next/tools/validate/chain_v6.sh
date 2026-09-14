#!/bin/bash
# wait for the complete sweep, then finalize (table -> build -> snapshot -> validate), then the prefetch experiment
T=/home/douya/tests
while ! grep -q "SWEEP2 DONE" $T/sweep2.out 2>/dev/null; do sleep 15; done
sed -i 's/^while pgrep -f "sweep2.sh|sweep_q8a.sh" > \/dev\/null; do sleep 15; done$/true/' $T/finalize_v6.sh
bash $T/finalize_v6.sh > $T/finalize_v6.out 2>&1
sed -i 's/^while pgrep -f "finalize_v6.sh|sweep2.sh" > \/dev\/null; do sleep 15; done$/true/' $T/exp_pf.sh
bash $T/exp_pf.sh > $T/exp_pf.out 2>&1
echo "CHAIN_V6 DONE $(date +%T)"
