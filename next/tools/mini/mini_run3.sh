#!/usr/bin/env bash
# generic mini launcher: MINI_DEVS (uuid list), LIBDIR, tag, extra args (later flags override earlier ones)
set -u
S=/home/douya/src/llama.cpp-flashnext-20260913
M=/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/mini
T=/home/douya/tests/mini
LIB=$T/${LIBDIR:-lib-new9}
TAG=$1; shift
pkill -f "llama-server.*--port 8094" 2>/dev/null; pkill -f "gdb -q -batch" 2>/dev/null
# wait until the previous instance has really released its GPU memory (a 12 GB context takes a few seconds to tear down)
for i in $(seq 1 30); do pgrep -f "llama-server.*--port 8094" >/dev/null || break; sleep 1; done
for i in $(seq 1 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1); [ -n "$used" ] || break; [ "$used" -lt 29000 ] && break; sleep 1; done
sleep 1
export CUDA_VISIBLE_DEVICES=${MINI_DEVS:-GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3}
export LD_LIBRARY_PATH=$LIB NEXT_QSA_OPT=1 NEXT_OPT_DIR=$T/opt LLAMA_NO_MMAP_PREFETCH=1
export LD_PRELOAD=/home/douya/qwen-3.8-next-opt/libnext-trace.so NEXT_TRACE_FLAG=$T/profile.flag NEXT_TRACE_LOG=$T/profile.csv
rm -f $T/profile.flag
nohup taskset -c 2-39 $LIB/llama-server -m $M/Qwen3.8-Flash-Next-mini4-noPLE.gguf \
  --host 127.0.0.1 --port 8094 --ctx-size 262144 --parallel 1 --gpu-layers all --split-mode layer --fit off --lazy-mode off --load-mode mmap \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 1024 --ubatch-size 1024 --threads 8 --threads-batch 16 \
  --no-context-shift --reasoning-format none --cache-ram 0 --ctx-checkpoints 8 --metrics --backend-sampling \
  --spec-type draft-mtp --spec-draft-model $M/mtp-mini4-shared-Q8_0-q4head.gguf --spec-draft-n-max 5 --spec-draft-n-min 0 --spec-draft-ngl all --spec-draft-type-k f16 --spec-draft-type-v f16 \
  "$@" > $T/logs/$TAG.log 2>&1 &
PID=$!
for i in $(seq 1 200); do sleep 2; if grep -aq "listening on" $T/logs/$TAG.log 2>/dev/null; then echo "READY $TAG after $((i*2))s"; exit 0; fi; if ! kill -0 $PID 2>/dev/null; then echo "DIED $TAG"; grep -a "error\|ERR\|abort\|failed" $T/logs/$TAG.log | grep -v CORS | tail -5 | cut -c1-200; exit 1; fi; done
echo "TIMEOUT $TAG"; exit 1
