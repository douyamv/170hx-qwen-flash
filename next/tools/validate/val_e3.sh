#!/bin/bash
# isolate the 1-GPU vs 2-GPU divergence at 2K: E'' = v6 libs, 2 GPUs, async inputs OFF; F = v5 libs, 2 GPUs
T=/home/douya/tests/mini; cd $T
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
ARGS="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
echo "########## E'': v6 libs, 2 GPUs, async inputs OFF (nosort + moea on) ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new20 bash mini_run3.sh v6_2gb $ARGS || exit 1
python3 stream_md5.py v6_2gb_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100; pref v6_2g_2048 v6_2gb_2048; pref v6a_2048 v6_2gb_2048
echo "########## E''': v6 libs, 2 GPUs, async ON, moea OFF, sort ON (numerics like v5) ##########"
touch opt/moea_off
NEXT_SCHED_ASYNC_INPUTS=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new20 bash mini_run3.sh v6_2gc $ARGS || exit 1
python3 stream_md5.py v6_2gc_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100; pref v6c_2048 v6_2gc_2048; pref v5r_2048 v6_2gc_2048
rm -f opt/moea_off
echo "########## F: v5 libs (lib-new17), 2 GPUs ##########"
MINI_DEVS=$C0,$G3 LIBDIR=lib-new17 bash mini_run3.sh v5_2g $ARGS || exit 1
python3 stream_md5.py v5_2g_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100; pref v5r_2048 v5_2g_2048; pref v6_2gc_2048 v5_2g_2048
pkill -f "llama-server.*--port 8094"; echo "VAL_E3 DONE $(date +%T)"
