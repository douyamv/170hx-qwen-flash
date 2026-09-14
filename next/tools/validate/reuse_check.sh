#!/bin/bash
T=/home/douya/tests/mini; cd $T
while pgrep -f val_moea.sh > /dev/null; do sleep 10; done
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
LLAMA_GRAPH_RESULT_DEBUG=1 NEXT_MOEA=1 MINI_DEVS=$G3 LIBDIR=lib-new19 bash mini_run3.sh reuse --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' --verbose || exit 1
python3 stream_md5.py reuse_2048 2048 96 | tail -1
echo "reuse=1: $(grep -ac 'can reuse graph = 1' logs/reuse.log)  reuse=0: $(grep -ac 'can reuse graph = 0' logs/reuse.log)"
grep -a 'can reuse graph' logs/reuse.log | tail -n 60 | sed 's/.*= //' | tr '\n' ' '; echo
grep -a 'cannot reuse graph due to' logs/reuse.log | sort | uniq -c | head -n 5
pkill -f "llama-server.*--port 8094"; echo "REUSE_CHECK DONE $(date +%T)"
