#!/bin/bash
# opt-v6 deployment: libs from $L (default lib-new20), launch.json.v4; backups bin.prev7 / launch.json.prev7 (= opt-v5)
set -u
P=/home/douya/qwen-3.8-next-opt
L=${L:-/home/douya/tests/mini/lib-new20}
[ -f $L/libggml-cuda.so.0.22.0 ] && [ -f $P/launch.json.v4 ] || { echo "missing $L or launch.json.v4"; exit 1; }
echo "[0] backups"; rm -rf $P/bin.prev7; cp -a $P/bin $P/bin.prev7 && cp -a $P/launch.json $P/launch.json.prev7 && echo ok
echo "[1] stop $(date +%T)"; sudo -n systemctl stop qwen38-flashnext-opt-262k.service && echo stopped
for i in $(seq 1 90); do pgrep -f "llama-server.*--port 8093" >/dev/null || break; sleep 1; done
pgrep -fa "llama-server" | cut -c1-70 | head -3; nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "[2] install libs ($L)"; for f in $L/*.so*; do b=$(basename $f); cp -a $f $P/bin/$b.new && mv -f $P/bin/$b.new $P/bin/$b; done
for f in libggml-cuda.so.0.22.0 libggml-base.so.0.22.0 libllama.so.0.3.0 libllama-common.so.0.3.0; do echo "$f prod=$(md5sum $P/bin/$f | cut -c1-8) new=$(md5sum $L/$f | cut -c1-8)"; done
echo "[3] launch.json v4"; cp -a $P/launch.json.v4 $P/launch.json && python3 -c "
import json; j=json.load(open('$P/launch.json')); a=j['command']; e=j['environment']
print(' '.join(a[a.index('--batch-size'):a.index('--batch-size')+4]), '| bs', '--backend-sampling' in a, '| n_max', a[a.index('--spec-draft-n-max')+1], '| async', e.get('NEXT_SCHED_ASYNC_INPUTS'), 'nosort', e.get('NEXT_TOPK_NOSORT'))"
echo 4 > $P/opt/spec_n_max; rm -f $P/opt/draft_head_target $P/opt/qsa_topk_rows $P/opt/spec_p_min $P/opt/moea_off; ls $P/opt | tr '\n' ' '; echo
echo "[4] start $(date +%T)"; sudo -n systemctl start qwen38-flashnext-opt-262k.service && echo started; sleep 12
systemctl is-active qwen38-flashnext-opt-262k.service; systemctl show qwen38-flashnext-opt-262k.service -p MainPID --value
tail -2 $P/logs/server.log | cut -c1-150
