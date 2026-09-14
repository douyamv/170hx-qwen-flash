#!/bin/bash
# probability-level comparison of the two placements (moea on): near-tie flip vs real deviation
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_E6 DONE\|DIED\|TIMEOUT" $T/logs/val_e6.out 2>/dev/null; do sleep 10; done
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
CTX="--ctx-size 20480 --batch-size 512 --ubatch-size 512"; rm -f opt/moea_off
echo "########## P1: all on GPU3 (like j1) ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new21 bash mini_run3.sh p1 --spec-draft-device CUDA0 --override-tensor "^token_embd\.weight$=CUDA0" $CTX || exit 1
python3 /home/douya/tests/probe_probs.py p1 14
echo "########## P2: target on GPU3, token_embd + draft on CUDA0 (like k1) ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3,$C0 LIBDIR=lib-new21 bash mini_run3.sh p2 --tensor-split 4,0 --spec-draft-device CUDA1 --override-tensor "^token_embd\.weight$=CUDA1" $CTX || exit 1
python3 /home/douya/tests/probe_probs.py p2 14
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/probs-p1.json')); b = json.load(open('/home/douya/tests/mini/probs-p2.json'))
print('tokens p1:', a['tokens'][:14]); print('tokens p2:', b['tokens'][:14])
for i, (pa, pb) in enumerate(zip(a['probs'], b['probs'])):
    da = {t: p for t, p in pa}; db = {t: p for t, p in pb}
    common = set(da) & set(db); maxd = max((abs(da[t] - db[t]) for t in common), default=float('nan'))
    print('pos %2d top1 p1=%s(%.4f) p2=%s(%.4f)  max|dp| over common top5 = %.2e  p1 top2 gap=%.2e' % (i, pa[0][0], pa[0][1], pb[0][0], pb[0][1], maxd, (pa[0][1] - pa[1][1]) if len(pa) > 1 else float('nan')))
PY
pkill -f "llama-server.*--port 8094"; echo "VAL_E7 DONE $(date +%T)"
