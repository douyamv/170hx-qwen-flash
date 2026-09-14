#!/bin/bash
# pre-deployment validation of the opt-v6 libs (lib-new20) on the mini with production-equivalent flags
T=/home/douya/tests/mini; cd $T
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; C1=GPU-3ba5a344-6eac-6688-805c-c2adcbaa93c3; C2=GPU-fbc3ea3b-e64d-a2cd-3b2e-47b3ed0aa5c2
LIB=${LIB:-lib-new20}
echo 4096 > opt/qsa_min_kv; echo 4 > opt/spec_n_max; rm -f opt/qsa_topk_rows opt/draft_head_target opt/spec_p_min opt/moea_off
tl() { grep -a "eval time =" logs/$1.log | grep -v prompt | tail -1 | sed 's/.*eval time = *//' | cut -c1-90; }
errs() { grep -a "ERR\|abort\|GGML_ASSERT\|mismatch\|segfault\|CUDA error" logs/$1.log | grep -v CORS | head -3 | cut -c1-200; }
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b))"; }
G3ARGS="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
echo "########## A: v6 all features (GPU3) ##########"
NEXT_SCHED_ASYNC_INPUTS=1 NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=$LIB bash mini_run3.sh v6a $G3ARGS || exit 1
for n in 2048 16384; do python3 stream_md5.py v6a_$n $n 96 | tail -1 | cut -c1-120; echo "  eval: $(tl v6a)"; done
python3 mini_ckpt_test.py v6a 2>&1 | tail -4; echo "errors: $(errs v6a)"
echo "########## B: v6 libs, async inputs OFF (host-only change -> must be identical to A) ##########"
NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=$LIB bash mini_run3.sh v6b $G3ARGS || exit 1
for n in 2048 16384; do python3 stream_md5.py v6b_$n $n 96 | tail -1 | cut -c1-120; echo "  eval: $(tl v6b)"; pref v6a_$n v6b_$n; done
echo "errors: $(errs v6b)"
echo "########## C: v6 libs, top-k sort ON + moea OFF (numerics like v5) ##########"
touch opt/moea_off
NEXT_SCHED_ASYNC_INPUTS=1 MINI_DEVS=$G3 LIBDIR=$LIB bash mini_run3.sh v6c $G3ARGS || exit 1
for n in 2048 16384; do python3 stream_md5.py v6c_$n $n 96 | tail -1 | cut -c1-120; echo "  eval: $(tl v6c)"; pref v6a_$n v6c_$n; done
echo "errors: $(errs v6c)"; rm -f opt/moea_off
echo "########## D: v5 libs (lib-new17) same config, for speed reference ##########"
MINI_DEVS=$G3 LIBDIR=lib-new17 bash mini_run3.sh v5r $G3ARGS || exit 1
for n in 2048 16384; do python3 stream_md5.py v5r_$n $n 96 | tail -1 | cut -c1-120; echo "  eval: $(tl v5r)"; pref v6c_$n v5r_$n; done
echo "########## E: 3 GPUs split 2,1,1 (draft + token_embd on CUDA2), all v6 features, 262K ctx ##########"
NEXT_SCHED_ASYNC_INPUTS=1 NEXT_TOPK_NOSORT=1 MINI_DEVS=$C0,$C1,$C2 LIBDIR=$LIB bash mini_run3.sh v6_3g --tensor-split 2,1,1 --spec-draft-device CUDA2 --override-tensor '^token_embd\.weight$=CUDA2' || exit 1
for n in 2048 16384 70000; do python3 stream_md5.py v6_3g_$n $n 96 | tail -1 | cut -c1-120; echo "  eval: $(tl v6_3g)"; done
python3 mini_ckpt_test.py v6_3g 2>&1 | tail -4; echo "errors: $(errs v6_3g)"
nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | tr '\n' ' '; echo
pkill -f "llama-server.*--port 8094"; echo "VALIDATE_V6 DONE $(date +%T)"
