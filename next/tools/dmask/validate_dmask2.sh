#!/bin/bash
# device-mask validation, round 2: same q8a cap for every run (the 644 MiB head repack only fits when the mask path
# frees memory, and then the pools OOM on the shared GPU3), baseline + test with identical settings, CPU-sampling probes.
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new25/libnext-trace.so LIBDIR=lib-new25
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-backend-sampling"
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b), 'IDENTICAL' if a==b and len(a)>0 else 'DIFF')"; }
cmpprobs() { python3 -c "
import json; a=json.load(open('probs-$1.json')); b=json.load(open('probs-$2.json'))
print('probs $1 vs $2:', 'IDENTICAL' if a['probs']==b['probs'] and len(a['probs'])>0 else 'DIFF', len(a['probs']), len(b['probs']), 'tokens', 'same' if a['tokens']==b['tokens'] else 'differ')"; }
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -v CORS | head -3 | cut -c1-160; }
mem() { sleep 1; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -v "^235637" | tr '\n' ' '; }
touch opt/spec_adaptive_off
for dm in 0 2; do
  echo "########## R1 dm=$dm: 1 GPU (GPU3) streams 2048/16384 + probes ##########"
  NEXT_DEVICE_MASK=$dm NEXT_DEVICE_MASK_VERBOSE=1 MINI_DEVS=$G3 bash mini_run4.sh r1_$dm $A1 || exit 1
  for n in 2048 16384; do python3 stream_md5.py r1_${dm}_$n $n 96 2>/dev/null | tail -1 | cut -c1-100; done
  python3 $T/probe_probs.py r1_$dm 14 2>&1 | cut -c1-120
  echo "mem after 16K: $(mem)"; grep -a "device mask:\|q8a_get_weight" logs/r1_$dm.log | head -n 2 | cut -c20-140
  echo "errors: $(errs r1_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
for n in 2048 16384; do pref r1_0_$n r1_2_$n; done; cmpprobs r1_0 r1_2
echo "########## R2: kernel proof + decode timeline (1 GPU, dm=2) ##########"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh r2 $A1 || exit 1
python3 - <<'PY'
import json, urllib.request, glob, time, os
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:16384]
r = post(dict(prompt=toks, n_predict=1, cache_prompt=True)); print('prefill:', (r.get('timings') or {}).get('prompt_n'), r.get('err', ''))
open('/home/douya/tests/mini/profile.flag.cpu', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=200, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none')); t = r.get('timings') or {}
time.sleep(0.5); os.remove('/home/douya/tests/mini/profile.flag.cpu'); time.sleep(0.5)
print('decode with GPU mask: %s tok/s (%s tokens) %s' % (t.get('predicted_per_second'), t.get('predicted_n'), r.get('err', '')))
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=40, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none')); time.sleep(0.8); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
print('traced decode:', (r.get('timings') or {}).get('predicted_n'), r.get('err', ''))
PY
echo "-- host timeline:"; python3 $T/host/an_cpu.py profile.csv 0 2>&1 | grep -E "steps|SET_INPUTS|QSA_CPU|DECODE_TARGET|SYNC" | head -n 8
echo "-- mask kernels:"; grep -a "^KERNEL" profile.csv | awk -F'"' '{print $2}' | grep -iE "kq_mask|mask" | sed 's/^_Z[0-9]*//' | cut -c1-40 | sort | uniq -c
echo "-- memcpy HtoD count/bytes:"; grep -a "^MEMCPY" profile.csv | head -n 3 | cut -c1-120
echo "errors: $(errs r2)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## R3: 2 GPUs (CUDA0 + GPU3, split 2,2) host vs device mask ##########"
for dm in 0 2; do
  NEXT_DEVICE_MASK=$dm NEXT_DEVICE_MASK_VERBOSE=1 MINI_DEVS=$C0,$G3 bash mini_run4.sh r3_$dm $A2 || exit 1
  python3 stream_md5.py r3_${dm}_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100
  python3 $T/probe_probs.py r3_$dm 14 2>&1 | cut -c1-120
  grep -a "device mask:" logs/r3_$dm.log | head -n 1 | cut -c20-140
  echo "errors: $(errs r3_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
pref r3_0_2048 r3_2_2048; cmpprobs r3_0 r3_2
echo "########## R4: rewind equality (1 GPU, dm=2 vs dm=0) ##########"
for dm in 2 0; do
  NEXT_DEVICE_MASK=$dm MINI_DEVS=$G3 bash mini_run4.sh r4_$dm $A1 || exit 1
  python3 - $dm <<'PY'
import json, urllib.request, glob, sys
dm = sys.argv[1]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
toks = post({'content': ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))}, '/tokenize')['tokens']
A = toks[:16384]; out = {}
def run(k, prompt, n):
    r = post(dict(prompt=prompt, n_predict=n, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none', n_probs=3, backend_sampling=False))
    out[k] = r.get('tokens', r.get('err')); out[k + '_prompt_n'] = (r.get('timings') or {}).get('prompt_n')
    out[k + '_probs'] = [[(p.get('id'), p.get('prob', p.get('logprob'))) for p in (c.get('top_logprobs') or c.get('top_probs') or c.get('probs') or [])] for c in r.get('completion_probabilities', [])]
run('A', A, 24); run('A2', A[:16084] + toks[30000:30300], 48); run('A3', (A[:16084] + toks[30000:30300])[:12000] + toks[40000:40200], 48)
json.dump(out, open(f'/home/douya/tests/mini/rewind-r4_{dm}.json', 'w')); print('dm=%s prompt_n A=%s A2=%s A3=%s' % (dm, out['A_prompt_n'], out['A2_prompt_n'], out['A3_prompt_n']))
PY
  echo "errors: $(errs r4_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/rewind-r4_2.json')); b = json.load(open('/home/douya/tests/mini/rewind-r4_0.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a[k], b[k]
    if isinstance(x, list) and isinstance(y, list):
        p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        print('%s: tokens gpu-mask vs host-mask common prefix %d / %d %d; probs %s' % (k, p, len(x), len(y), 'IDENTICAL' if a[k+'_probs'] == b[k+'_probs'] and a[k+'_probs'] else 'DIFF'))
    else: print(k, 'error:', x if not isinstance(x, list) else '', y if not isinstance(y, list) else '')
PY
echo "VALIDATE2 DONE $(date +%T)"
