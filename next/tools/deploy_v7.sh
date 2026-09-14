#!/bin/bash
# after the huihui mini smoke test passes: switch production to the abliterated target (launch.json.v7, same lib-new28)
T=/home/douya/tests; P=/home/douya/qwen-3.8-next-opt
until grep -q "AFTER_DL DONE\|not verified\|structure mismatch" $T/after_dl.log 2>/dev/null; do sleep 15; done
if ! grep -q "AFTER_DL DONE" $T/after_dl.log || ! grep -a "hh1_2048 .* got [1-9]" $T/after_dl.log >/dev/null || grep -a "CUDA error\|GGML_ASSERT\|abort" $T/after_dl.log | grep -v "^errors: $" | grep -q .; then
  echo "[v7] smoke test not clean - NOT deploying"; grep -a "hh1_\|error\|abort\|DIED" $T/after_dl.log | cut -c1-140; exit 1
fi
echo "[v7] smoke test ok, deploying $(date +%T)"
cat > $T/post_v7.sh <<'POST'
#!/bin/bash
python3 /home/douya/tests/prod_sweep.py short 2>&1 | grep -E "short_greedy|EQUAL" | cut -c1-140
POST
chmod +x $T/post_v7.sh
L=$T/mini/lib-new28 LJ=launch.json.v7 POST=$T/post_v7.sh bash $T/host/deploy_chain.sh
