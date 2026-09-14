#!/bin/bash
T=/home/douya/tests/mini; cd $T
S=/home/douya/src/llama.cpp-flashnext-20260913
mkdir -p lib-new19 && cp -a lib-new17/. lib-new19/ && cp $S/build-sm80/bin/libggml-cuda.so.0.22.0 lib-new19/libggml-cuda.so.0.22.0
ls -la lib-new19/libggml-cuda.so.0.22.0
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
echo 4096 > opt/qsa_min_kv; echo 4 > opt/spec_n_max
tl() { grep -a "eval time =\|acceptance" logs/$1.log | tail -2 | cut -c40-150 | tr '\n' ' '; echo; }
errs() { grep -a "ERR\|abort\|GGML_ASSERT\|mismatch\|segfault" logs/$1.log | head -3 | cut -c1-200; }
for m in 1 0; do
  echo "########## NEXT_MOEA=$m ##########"
  NEXT_MOEA=$m MINI_DEVS=$G3 LIBDIR=lib-new19 bash mini_run3.sh m19_$m --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' || exit 1
  for n in 2048 16384; do python3 stream_md5.py m19_${m}_$n $n 96 | tail -1; tl m19_$m; done
  echo "errors: $(errs m19_$m)"
done
python3 - << 'PY'
import json
for n in (2048, 16384):
    a = json.load(open(f'/home/douya/tests/mini/stream-m19_1_{n}.json')); b = json.load(open(f'/home/douya/tests/mini/stream-m19_0_{n}.json'))
    pref = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
    print(f'{n}: common prefix moea/mmvq = {pref}/{len(a)} tokens')
PY
pkill -f "llama-server.*--port 8094"; echo "VAL_MOEA DONE $(date +%T)"
