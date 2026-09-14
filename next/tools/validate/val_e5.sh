#!/bin/bash
# is the 1-vs-2-GPU token difference a moea bug or generic split numerics?
# K1: all layers on GPU3, draft on CUDA0 (moea on)  -> vs j1 (everything on GPU3)
# L1/L2: moea OFF, 1 GPU vs 2 GPUs, a different prompt (4096) with 96 tokens
T=/home/douya/tests/mini; cd $T
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
CTX="--ctx-size 20480 --batch-size 512 --ubatch-size 512"
echo "########## K1: target all on GPU3, draft + token_embd on CUDA0 (2 devices, split 4,0), moea ON ##########"
rm -f opt/moea_off
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3,$C0 LIBDIR=lib-new21 bash mini_run3.sh k1 --tensor-split 4,0 --spec-draft-device CUDA1 --override-tensor "^token_embd\.weight$=CUDA1" $CTX || exit 1
python3 stream_md5.py k1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref j1_2048 k1_2048; pref j2_2048 k1_2048
echo "########## L1: 1 GPU, moea OFF, prompt 4096 ##########"; touch opt/moea_off
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new21 bash mini_run3.sh l1 --spec-draft-device CUDA0 --override-tensor "^token_embd\.weight$=CUDA0" $CTX || exit 1
python3 stream_md5.py l1_4096 4096 96 2>/dev/null | tail -1 | cut -c1-90; python3 stream_md5.py l1_8192 8192 96 2>/dev/null | tail -1 | cut -c1-90
echo "########## L2: 2 GPUs split 2,2, moea OFF, prompt 4096 ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new21 bash mini_run3.sh l2 --tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor "^token_embd\.weight$=CUDA1" $CTX || exit 1
python3 stream_md5.py l2_4096 4096 96 2>/dev/null | tail -1 | cut -c1-90; pref l1_4096 l2_4096; python3 stream_md5.py l2_8192 8192 96 2>/dev/null | tail -1 | cut -c1-90; pref l1_8192 l2_8192
rm -f opt/moea_off
pkill -f "llama-server.*--port 8094"; echo "VAL_E5 DONE $(date +%T)"
