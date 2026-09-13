# full pre-deployment validation of lib-new9 with production-equivalent flags (mini 4-layer model)
T=/home/douya/tests/mini; cd $T
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8; C1=GPU-3ba5a344-6eac-6688-805c-c2adcbaa93c3; C2=GPU-fbc3ea3b-e64d-a2cd-3b2e-47b3ed0aa5c2
echo 4096 > opt/qsa_min_kv; echo 4 > opt/spec_n_max; rm -f opt/qsa_topk_rows opt/draft_head_target opt/spec_p_min
tl() { grep -a "eval time =\|acceptance" logs/$1.log | tail -2 | cut -c40-150 | tr '\n' ' '; echo; }
errs() { grep -a "ERR\|abort\|GGML_ASSERT\|mismatch\|segfault" logs/$1.log | head -3 | cut -c1-200; }
echo "########## V1 single GPU (GPU3), block cache ON ##########"
MINI_DEVS=$G3 LIBDIR=lib-new9 bash mini_run3.sh v9 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' || exit 1
for n in 2048 16384 70000; do python3 stream_md5.py v9_$n $n 96 | tail -1; tl v9; done
python3 mini_ckpt_test.py v9 2>&1 | tail -6; echo "errors: $(errs v9)"
nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "########## V2 single GPU, block cache OFF (NEXT_QSA_NO_BLKCACHE=1) ##########"
NEXT_QSA_NO_BLKCACHE=1 MINI_DEVS=$G3 LIBDIR=lib-new9 bash mini_run3.sh v9nb --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' || exit 1
for n in 2048 16384 70000; do python3 stream_md5.py v9nb_$n $n 96 | tail -1; tl v9nb; done
echo "errors: $(errs v9nb)"
echo "########## V3 three GPUs (CUDA0,CUDA1,CUDA2 split 2,1,1; draft + token_embd on CUDA2) ##########"
MINI_DEVS=$C0,$C1,$C2 LIBDIR=lib-new9 bash mini_run3.sh v9_3g --tensor-split 2,1,1 --spec-draft-device CUDA2 --override-tensor '^token_embd\.weight$=CUDA2' || exit 1
for n in 2048 16384 70000 200000; do python3 stream_md5.py v9_3g_$n $n 96 | tail -1; tl v9_3g; done
python3 mini_ckpt_test.py v9_3g 2>&1 | tail -6; echo "errors: $(errs v9_3g)"
nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "########## md5 summary ##########"
for n in 2048 16384 70000; do echo "$n: blk=$(python3 -c "import json,hashlib;print(hashlib.md5(open('stream-v9_$n.json').read().encode()).hexdigest()[:10])" 2>/dev/null) noblk=$(python3 -c "import json,hashlib;print(hashlib.md5(open('stream-v9nb_$n.json').read().encode()).hexdigest()[:10])" 2>/dev/null) 3gpu=$(python3 -c "import json,hashlib;print(hashlib.md5(open('stream-v9_3g_$n.json').read().encode()).hexdigest()[:10])" 2>/dev/null)"; done
python3 - << 'PY'
import json
for n in (2048, 16384, 70000):
    try:
        a = json.load(open(f'/home/douya/tests/mini/stream-v9_{n}.json')); b = json.load(open(f'/home/douya/tests/mini/stream-v9nb_{n}.json')); c = json.load(open(f'/home/douya/tests/mini/stream-v9_3g_{n}.json'))
        def pref(x, y): return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        print(f'{n}: common prefix blk/noblk = {pref(a,b)}/{len(a)} tokens, blk/3gpu = {pref(a,c)}/{len(a)}')
    except Exception as e: print(n, 'compare failed', e)
PY
pkill -f "llama-server.*--port 8094"; echo "VALIDATE_ALL DONE $(date +%T)"
