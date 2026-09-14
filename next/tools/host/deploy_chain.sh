#!/bin/bash
# opt-v6 rollout: deploy8.sh (lib-new20 + launch.json.v4) -> wait for the server -> post10.sh (sweeps, TTFT, moea A/B, trace)
P=/home/douya/qwen-3.8-next-opt; T=/home/douya/tests
echo "[chain] deploy start $(date +%T)"
bash $T/deploy8.sh 2>&1
echo "[chain] waiting for the server (HDD load ~29 min) $(date +%T)"
for i in $(seq 1 300); do
  sleep 10
  if curl -s -m 3 http://127.0.0.1:8093/health 2>/dev/null | grep -q '"ok"'; then echo "[chain] READY after $((i*10))s $(date +%T)"; break; fi
  if ! systemctl is-active --quiet qwen38-flashnext-opt-262k.service; then echo "[chain] SERVICE DIED $(date +%T)"; grep -a "error\|ERR\|abort\|GGML_ASSERT" $P/logs/server.log | tail -n 5 | cut -c1-200; exit 1; fi
done
curl -s -m 3 http://127.0.0.1:8093/health | grep -q '"ok"' || { echo "[chain] TIMEOUT waiting for the server"; exit 1; }
grep -a "q8a:\|moea\|plans read" $P/logs/server.log | tail -n 3 | cut -c1-160
nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv,noheader | tr '\n' ' '; echo
bash ${POST:-$T/post10.sh} 2>&1
echo "[chain] DEPLOY DONE $(date +%T)"
