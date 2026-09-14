#!/bin/bash
# DM4: checkpoint / rewind test with the GPU mask (exercises the dirty -> full re-upload path) vs host mask
T=/home/douya/tests/mini; cd $T
while ! grep -q "VAL_DM3 DONE" /home/douya/tests/dmask/val_dm3.out 2>/dev/null; do sleep 10; done
python3 - <<'PY'
import re
p = '/home/douya/tests/mini/mini_ckpt_test.py'; s = open(p).read()
n = 0
for pat, rep in ((r'text=True(?!, errors)', "text=True, errors='replace'"), (r'universal_newlines=True(?!, errors)', "universal_newlines=True, errors='replace'"), (r"\.decode\('utf-8'\)", ".decode('utf-8', errors='replace')"), (r"\.decode\(\)", ".decode(errors='replace')")):
    s, k = re.subn(pat, rep, s); n += k
open(p, 'w').write(s); print('mini_ckpt_test.py: %d decode sites made tolerant' % n)
PY
grep -n "subprocess\|decode\|urlopen\|json.loads" mini_ckpt_test.py | head -n 6 | cut -c1-120
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0"
for dm in 1 0; do
  echo "########## DM4 NEXT_DEVICE_MASK=$dm: checkpoint rewind test ##########"
  NEXT_DEVICE_MASK=$dm TRACER=$T/lib-new25/libnext-trace.so NEXT_TOPK_NOSORT=1 MINI_DEVS=$G3 LIBDIR=lib-new25 bash mini_run4.sh dm4_$dm $A1 --spec-draft-backend-sampling || exit 1
  python3 mini_ckpt_test.py dm4_$dm 2>&1 | tail -n 8 | cut -c1-160
  grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/dm4_$dm.log | grep -v CORS | head -n 2 | cut -c1-140
  pkill -f "llama-server.*--port 8094"; sleep 3
done
echo "VAL_DM4 DONE $(date +%T)"
