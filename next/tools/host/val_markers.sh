#!/bin/bash
# lib-new22 (markers) on the mini: streams unchanged vs dbs1, and the markers-only tracer yields a host timeline
T=/home/douya/tests/mini; S=/home/douya/src/llama.cpp-flashnext-20260913; cd $T
cp $S/build-sm80/bin/libllama-server-impl.so $T/lib-new22/libllama-server-impl.so && md5sum $T/lib-new22/libllama-server-impl.so $S/build-sm80/bin/libllama-server-impl.so | cut -c1-8,34-
sed 's|export LD_PRELOAD=/home/douya/qwen-3.8-next-opt/libnext-trace.so|export LD_PRELOAD=${TRACER:-/home/douya/qwen-3.8-next-opt/libnext-trace.so}|' mini_run3.sh > mini_run4.sh && grep -c 'TRACER' mini_run4.sh
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
CTX="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
rm -f profile.flag profile.flag.cpu; : > profile.csv
TRACER=$T/lib-new22/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new22 bash mini_run4.sh mk22 $CTX --spec-draft-backend-sampling || exit 1
python3 stream_md5.py mk22_2048 2048 96 2>/dev/null | tail -1 | cut -c1-90
touch profile.flag.cpu; sleep 0.3
python3 stream_md5.py mk22_16384 16384 96 2>/dev/null | tail -1 | cut -c1-90
sleep 0.5; rm -f profile.flag.cpu; sleep 0.5
python3 -c "
import json
for n in (2048, 16384):
    a=json.load(open(f'stream-dbs1_{n}.json')); b=json.load(open(f'stream-mk22_{n}.json'))
    p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print(n, 'dbs1 vs mk22 prefix', p, len(a), len(b))"
echo "== markers-only CSV =="; wc -l profile.csv; cut -d, -f1 profile.csv | sort | uniq -c | head; grep -a "START_CPU\|STOP_CPU" profile.csv | cut -c1-80
python3 /home/douya/tests/host/an_cpu.py profile.csv 1 2>&1 | head -n 60
grep -a "ERR\|abort\|GGML_ASSERT" logs/mk22.log | grep -v CORS | head -n 3 | cut -c1-160
pkill -f "llama-server.*--port 8094"; echo "VAL_MARKERS DONE $(date +%T)"
