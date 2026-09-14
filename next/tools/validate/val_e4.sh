#!/bin/bash
# which v6 feature is split-dependent? pairs (1 GPU vs 2 GPUs) with a single feature on: G = nosort only, H = moea only
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_E3 DONE\|DIED\|TIMEOUT" $T/logs/val_e3.out 2>/dev/null; do sleep 10; done
C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512"
echo "########## G1: 1 GPU, moea OFF, nosort ON ##########"; touch opt/moea_off
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new20 bash mini_run3.sh g1 $A1 || exit 1; python3 stream_md5.py g1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90
echo "########## G: 2 GPUs, moea OFF, nosort ON ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$G3 LIBDIR=lib-new20 bash mini_run3.sh g2 $A2 || exit 1; python3 stream_md5.py g2_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref g1_2048 g2_2048
rm -f opt/moea_off
echo "########## H1: 1 GPU, moea ON, sort ON ##########"
MINI_DEVS=$G3 LIBDIR=lib-new20 bash mini_run3.sh h1 $A1 || exit 1; python3 stream_md5.py h1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90
echo "########## H: 2 GPUs, moea ON, sort ON ##########"
MINI_DEVS=$C0,$G3 LIBDIR=lib-new20 bash mini_run3.sh h2 $A2 || exit 1; python3 stream_md5.py h2_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90; pref h1_2048 h2_2048
pkill -f "llama-server.*--port 8094"; echo "VAL_E4 DONE $(date +%T)"
