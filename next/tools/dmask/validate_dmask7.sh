#!/bin/bash
# kernel-sequence diff: host mask vs device mask (lib-new26, no speculation, 1 GPU): trace a 16K prefill + 10 decode
# steps with CUPTI and compare the kernel name sequences -> any structural difference besides kq_mask_dev itself
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new26/libnext-trace.so LIBDIR=lib-new26
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-type none"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
for dm in 0 2; do
  echo "########## K$dm: dm=$dm, no speculation, CUPTI trace of a 2K prefill + 12 decode steps ##########"
  rm -f profile.flag profile.flag.cpu; : > profile.csv
  NEXT_DEVICE_MASK=$dm MINI_DEVS=$G3 bash mini_run4.sh k$dm $A1 || exit 1
  python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:2048]
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=12, cache_prompt=False, ignore_eos=True, temperature=0.0, reasoning_format='none')); time.sleep(0.8); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
print('traced: prompt_n', (r.get('timings') or {}).get('prompt_n'), 'predicted', (r.get('timings') or {}).get('predicted_n'), r.get('err', ''))
PY
  grep -a "^KERNEL" profile.csv | awk -F'"' '{print $2}' | sed 's/^_Z[0-9]*//' > kseq_dm$dm.txt
  wc -l < kseq_dm$dm.txt
  echo "errors: $(errs k$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
echo "-- kernel multiset difference (count name), dm0 vs dm2:"
sort kseq_dm0.txt | uniq -c | sed 's/^ *//' > kcnt0.txt; sort kseq_dm2.txt | uniq -c | sed 's/^ *//' > kcnt2.txt
diff kcnt0.txt kcnt2.txt | cut -c1-120 | head -n 40
echo "-- first sequence divergence (ignoring kq_mask_dev lines):"
grep -av kq_mask_dev kseq_dm2.txt > kseq_dm2_nomask.txt
cmp kseq_dm0.txt kseq_dm2_nomask.txt | head -n 2
python3 - <<'PY'
a = open('/home/douya/tests/mini/kseq_dm0.txt').read().split('\n'); b = open('/home/douya/tests/mini/kseq_dm2_nomask.txt').read().split('\n')
i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print('sequence lengths', len(a), len(b), 'first differing index', i)
for k in range(max(0, i - 3), min(i + 6, len(a), len(b))): print('%6d  %-60.60s | %-60.60s' % (k, a[k], b[k]))
PY
echo "VALIDATE7 DONE $(date +%T)"
