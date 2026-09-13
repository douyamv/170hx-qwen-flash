P=/home/douya/qwen-3.8-next-opt
N0=$(grep -ac "listening on" $P/logs/server.log)
for i in $(seq 1 120); do sleep 30
  if [ "$(grep -ac 'listening on' $P/logs/server.log)" -gt "$N0" ]; then echo "READY after $((i*30))s $(date +%T)"; break; fi
  if ! systemctl is-active --quiet qwen38-flashnext-opt-262k; then echo "SERVICE DIED $(date +%T)"; tail -30 $P/logs/server.log | cut -c1-200; exit 1; fi
done
tail -c 400000 $P/logs/server.log | grep -a "error\|ERR\|abort\|failed" | grep -v "CORS\|security\|more info" | tail -5 | cut -c1-200
nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | tr '\n' ' '; echo; free -m | sed -n 2p
pkill -f "llama-server.*--port 8094" 2>/dev/null
sleep 5; python3 /home/douya/tests/prod_sweep.py short 2>&1
python3 /home/douya/tests/ttft_probe.py 2000 2>&1
echo "MONITOR DONE $(date +%T)"
