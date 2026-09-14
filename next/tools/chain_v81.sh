#!/bin/bash
# stop production (frees GPU3 for the mini and the old shard-2 inode on the slow disk), validate lib-new30 on the mini,
# deploy v8.1 (lib-new30 + launch.json.v8) if clean, wait for the server, benchmark, then defrag shard 3
T=/home/douya/tests; P=/home/douya/qwen-3.8-next-opt
echo "[v81] stop production $(date +%T)"; sudo -n systemctl stop qwen38-flashnext-opt-262k.service; for i in $(seq 1 90); do pgrep -f "llama-server.*--port 8093" >/dev/null || break; sleep 1; done; sleep 3
df -h /mnt/slowdisk | tail -n 1; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
bash $T/dmask/validate_inc.sh > $T/dmask/validate_inc.out 2>&1
grep -a -E "DM_CHECK|errors:|got .* tokens|prompt_n|req [0-9]|DIED|TIMEOUT" $T/dmask/validate_inc.out | cut -c1-160
ok=1
grep -q "DM_CHECK w1: .*bad ubatches so far 0" $T/dmask/validate_inc.out || ok=0
grep -q "DM_CHECK w2: .*bad ubatches so far 0" $T/dmask/validate_inc.out || ok=0
grep -a "^errors: " $T/dmask/validate_inc.out | grep -v "^errors: $" | grep -v "peg_parse" | grep -q . && ok=0
grep -q "DIED\|TIMEOUT" $T/dmask/validate_inc.out && ok=0
if [ $ok -ne 1 ]; then echo "[v81] VALIDATION NOT CLEAN - restarting production with the previous libs (lib-new29)"; sudo -n systemctl start qwen38-flashnext-opt-262k.service; exit 1; fi
echo "[v81] validation clean, deploying lib-new30 $(date +%T)"
L=$T/mini/lib-new30 LJ=launch.json.v8 bash $T/deploy8.sh
for i in $(seq 1 400); do sleep 15; if curl -s -m 3 http://127.0.0.1:8093/health 2>/dev/null | grep -q '"ok"'; then echo "[chain] READY $(date +%T)"; break; fi; systemctl is-active --quiet qwen38-flashnext-opt-262k.service || { echo "[chain] SERVICE DIED $(date +%T)"; grep -a 'error\|ERR\|abort\|GGML_ASSERT' $P/logs/server.log | tail -n 5 | cut -c1-200; exit 1; }; done
curl -s -m 3 http://127.0.0.1:8093/health | grep -q '"ok"' || { echo "[chain] TIMEOUT waiting"; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
grep -a "n_slots" $P/logs/server.log | tail -n 1 | cut -c1-120
bash $T/post_v8.sh
echo "[chain] DEPLOY DONE $(date +%T)"
# defrag shard 3 now that the old inodes are released
H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/UD-Q4_K_XL; f=$H/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf; df -h /mnt/slowdisk | tail -n 1
t0=$(date +%s); nice -n 10 cp $f $f.contig && [ "$(stat -c %s $f)" = "$(stat -c %s $f.contig)" ] && mv -f $f.contig $f && echo "[defrag] shard 3 re-copied contiguously in $(( $(date +%s) - t0 )) s $(date +%T)" || { echo "[defrag] shard 3 COPY FAILED"; rm -f $f.contig; }
sync; echo "[v81] ALL DONE $(date +%T)"
