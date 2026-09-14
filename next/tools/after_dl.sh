#!/bin/bash
# after the huihui download verified: MTP Q4 head from the new output.weight, 4-layer mini + mini MTP, smoke test,
# launch.json.v7. Does NOT restart production.
T=/home/douya/tests; H=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF; B=/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF
until grep -q "SHA256 ALL OK\|SHA256 MISMATCH" $T/dl_huihui.log 2>/dev/null; do sleep 30; done
grep -q "SHA256 ALL OK" $T/dl_huihui.log || { echo "download not verified, stop"; exit 1; }
echo "AFTER_DL START $(date +%T)"; mkdir -p $H/MTP $H/mini
python3 - <<'PY'
import re
T = '/home/douya/tests'; H = '/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/'; B = '/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/'
s = open(T + '/make_mtp_q4head.py').read()
s = s.replace("SRC_MTP = BASE + 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'", "SRC_MTP = '%sMTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'" % B)
s = s.replace("DST_MTP = BASE + 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf'", "DST_MTP = BASE + 'MTP/mtp-Huihui-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf'")
s = s.replace("BASE = '%s'" % B, "BASE = '%s'" % H)
assert H in s and 'mtp-Huihui' in s; open(T + '/make_mtp_q4head_huihui.py', 'w').write(s)
s = open(T + '/make_mini.py').read().replace("BASE = '%s'" % B, "BASE = '%s'" % H).replace("mini/Qwen3.8-Flash-Next-mini{NL}-noPLE.gguf", "mini/Huihui-Qwen3.8-Flash-Next-mini{NL}-noPLE.gguf")
assert H in s; open(T + '/make_mini_huihui.py', 'w').write(s)
s = open(T + '/make_mini_mtp.py').read().replace("BASE = '%s'" % B, "BASE = '%s'" % H).replace("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf", "MTP/mtp-Huihui-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf").replace("mini/mtp-mini{NL}-shared-Q8_0-q4head.gguf", "mini/mtp-huihui-mini{NL}-shared-Q8_0-q4head.gguf")
assert H in s; open(T + '/make_mini_mtp_huihui.py', 'w').write(s)
import json
j = json.load(open('/home/douya/qwen-3.8-next-opt/launch.json.v6')); a = j['command']
a[a.index('--model') + 1] = H + 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
a[a.index('--spec-draft-model') + 1] = H + 'MTP/mtp-Huihui-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf'
a[a.index('--alias') + 1] = 'Huihui-Qwen3.8-Flash-Next-abliterated-UD-Q4_K_XL'
j['stage'] = 'opt-v7: opt-v6.2 libs + huihui-ai abliterated UD-Q4_K_XL target, base MTP layer with a Q4 head from the abliterated output.weight'
json.dump(j, open('/home/douya/qwen-3.8-next-opt/launch.json.v7', 'w'), indent=2); print('scripts + launch.json.v7 written')
PY
echo "-- MTP Q4 head $(date +%T)"; /usr/bin/python3 $T/make_mtp_q4head_huihui.py 2>&1 | tail -n 4
echo "-- mini target $(date +%T)"; /usr/bin/python3 $T/make_mini_huihui.py 4 2>&1 | tail -n 3
echo "-- mini MTP $(date +%T)"; /usr/bin/python3 $T/make_mini_mtp_huihui.py 4 2>&1 | tail -n 3
ls -la $H/MTP $H/mini | awk '{print $5, $9}'
echo "-- mini smoke test (GPU3, q8a capped) $(date +%T)"; cd $T/mini
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new28/libnext-trace.so LIBDIR=lib-new28 NEXT_DEVICE_MASK=2
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
bash mini_run4.sh hh1 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' --spec-draft-backend-sampling --model $H/mini/Huihui-Qwen3.8-Flash-Next-mini4-noPLE.gguf --spec-draft-model $H/mini/mtp-huihui-mini4-shared-Q8_0-q4head.gguf || exit 1
grep -a "loading model\|draft model" logs/hh1.log | cut -c1-160
python3 stream_md5.py hh1_2048 2048 48 2>/dev/null | tail -1 | cut -c1-100
python3 stream_md5.py hh1_16384 16384 24 2>/dev/null | tail -1 | cut -c1-100
grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/hh1.log | grep -av CORS | head -n 3 | cut -c1-160
pkill -f "llama-server.*--port 8094"; sleep 3
echo "AFTER_DL DONE $(date +%T)"
