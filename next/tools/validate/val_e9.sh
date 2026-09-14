#!/bin/bash
# re-probe P1 (same device) with the fixed parser and compare with P2's probabilities
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_E8 DONE" $T/logs/val_e8.out 2>/dev/null; do sleep 10; done
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
CTX="--ctx-size 20480 --batch-size 512 --ubatch-size 512"; rm -f opt/moea_off
echo "########## P1 (re-probe): all on GPU3 ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new21 bash mini_run3.sh p1b --spec-draft-device CUDA0 --override-tensor "^token_embd\.weight$=CUDA0" $CTX || exit 1
python3 /home/douya/tests/probe_probs.py p1b 14
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/probs-p1b.json')); b = json.load(open('/home/douya/tests/mini/probs-p2.json'))
print('tokens p1:', a['tokens'][:14]); print('tokens p2:', b['tokens'][:14])
for i, (pa, pb) in enumerate(zip(a['probs'], b['probs'])):
    da = {t: p for t, p in pa}; db = {t: p for t, p in pb}
    common = set(da) & set(db); maxd = max((abs(da[t] - db[t]) for t in common), default=float('nan'))
    gap = (pa[0][1] - pa[1][1]) if len(pa) > 1 else float('nan')
    print('pos %2d  p1 top1 %s (%.5f)  p2 top1 %s (%.5f)  max|dp| common top5 %.2e  p1 top1-top2 gap %.2e' % (i, pa[0][0], pa[0][1], pb[0][0], pb[0][1], maxd, gap))
PY
pkill -f "llama-server.*--port 8094"; echo "VAL_E9 DONE $(date +%T)"
