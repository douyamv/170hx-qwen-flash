#!/bin/bash
# which placement matters with moea on: M1 = only token_embd remote; M2 = only the draft remote (target all on GPU3)
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_E5 DONE\|DIED\|TIMEOUT" $T/logs/val_e5.out 2>/dev/null; do sleep 10; done
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
CTX="--ctx-size 20480 --batch-size 512 --ubatch-size 512"; rm -f opt/moea_off
echo "########## M1: 2 devices, target+draft on GPU3 (CUDA0), token_embd on CUDA1 ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3,$C0 LIBDIR=lib-new21 bash mini_run3.sh m1 --tensor-split 4,0 --spec-draft-device CUDA0 --override-tensor "^token_embd\.weight$=CUDA1" $CTX || exit 1
python3 stream_md5.py m1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j1_2048 m1_2048; pref k1_2048 m1_2048
echo "########## M2: 2 devices, target + token_embd on GPU3 (CUDA0), draft on CUDA1 ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3,$C0 LIBDIR=lib-new21 bash mini_run3.sh m2 --tensor-split 4,0 --spec-draft-device CUDA1 --override-tensor "^token_embd\.weight$=CUDA0" $CTX || exit 1
python3 stream_md5.py m2_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j1_2048 m2_2048; pref k1_2048 m2_2048
echo "########## M3: 1 device, no speculative draft, moea ON ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new21 bash mini_run3.sh m3 --spec-draft-device CUDA0 --override-tensor "^token_embd\.weight$=CUDA0" $CTX --spec-type none || exit 1
python3 stream_md5.py m3_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j1_2048 m3_2048; pref k1_2048 m3_2048
pkill -f "llama-server.*--port 8094"; echo "VAL_E6 DONE $(date +%T)"
