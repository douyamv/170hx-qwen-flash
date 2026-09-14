#!/bin/bash
# lib-new26, round 4: (S4) placement proof on 2 GPUs via CUPTI per-device kernel counts (GGML_SCHED_DEBUG output never
# reaches the server log), (S5/S6) reproducibility of the rewind scenario for dm=2 and dm=0, (S7/S8) the same scenario
# WITHOUT the MTP draft (--spec-draft-n-max 0) so every position carries CPU-sampling probabilities.
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new26/libnext-trace.so LIBDIR=lib-new26
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-backend-sampling"
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
rewind() { python3 - "$1" "$2" <<'PY'
import json, urllib.request, glob, sys
tag, nprobs = sys.argv[1], int(sys.argv[2])
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=600).read())
    except Exception as e: return {'err': str(e)[:80]}
toks = post({'content': ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))}, '/tokenize')['tokens']
A = toks[:16384]; out = {}
def run(k, prompt, n):
    r = post(dict(prompt=prompt, n_predict=n, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, reasoning_format='none', n_probs=nprobs, backend_sampling=False))
    out[k] = r.get('tokens', r.get('err')); out[k + '_prompt_n'] = (r.get('timings') or {}).get('prompt_n')
    out[k + '_probs'] = [[(p.get('id'), p.get('prob', p.get('logprob'))) for p in (c.get('top_logprobs') or c.get('top_probs') or c.get('probs') or [])] for c in r.get('completion_probabilities', [])]
run('A', A, 24); run('A2', A[:16084] + toks[30000:30300], 48); run('A3', (A[:16084] + toks[30000:30300])[:12000] + toks[40000:40200], 48)
json.dump(out, open(f'/home/douya/tests/mini/rewind-{tag}.json', 'w')); print('%s prompt_n A=%s A2=%s A3=%s' % (tag, out['A_prompt_n'], out['A2_prompt_n'], out['A3_prompt_n']))
PY
}
cmprw() { python3 - "$1" "$2" <<'PY'
import json, sys
ta, tb = sys.argv[1], sys.argv[2]
a = json.load(open(f'/home/douya/tests/mini/rewind-{ta}.json')); b = json.load(open(f'/home/douya/tests/mini/rewind-{tb}.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a[k], b[k]
    if isinstance(x, list) and isinstance(y, list):
        p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        pa, pb = a[k+'_probs'], b[k+'_probs']; npos = sum(1 for r in pa if r)
        q = next((i for i, (u, v) in enumerate(zip(pa, pb)) if u != v), min(len(pa), len(pb)))
        print('%s vs %s %s: tokens common prefix %d / %d %d; probs first diff at %d of %d (%d positions carry probs)' % (ta, tb, k, p, len(x), len(y), q, min(len(pa), len(pb)), npos))
    else: print('%s vs %s %s: error %s %s' % (ta, tb, k, x if not isinstance(x, list) else '', y if not isinstance(y, list) else ''))
PY
}
echo "########## S4: 2 GPUs, dm=2, lib-new26: CUPTI per-device kq_mask_dev counts (placement proof) ##########"
rm -f profile.flag profile.flag.cpu; : > profile.csv
NEXT_DEVICE_MASK=2 MINI_DEVS=$C0,$G3 bash mini_run4.sh s4 $A2 || exit 1
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
open('/home/douya/tests/mini/profile.flag', 'w').close(); time.sleep(0.3)
r = post(dict(prompt=toks, n_predict=40, cache_prompt=True, ignore_eos=True, temperature=0.0, reasoning_format='none')); time.sleep(0.8); os.remove('/home/douya/tests/mini/profile.flag'); time.sleep(1.5)
print('traced decode:', (r.get('timings') or {}).get('predicted_n'), r.get('err', ''))
PY
echo "-- kernels per device column (all kernels):"; grep -a "^KERNEL" profile.csv | cut -d, -f2 | sort | uniq -c | tr '\n' ' '; echo
echo "-- kq_mask_dev per device column:"; grep -a "^KERNEL" profile.csv | grep -a kq_mask_dev | cut -d, -f2 | sort | uniq -c | tr '\n' ' '; echo
echo "-- kq_mask_dev per (device, context) columns:"; grep -a "^KERNEL" profile.csv | grep -a kq_mask_dev | cut -d, -f2,5 | sort | uniq -c | tr '\n' ' '; echo
echo "-- flash_attn per device (for reference):"; grep -a "^KERNEL" profile.csv | grep -a -i "flash_attn" | cut -d, -f2 | sort | uniq -c | tr '\n' ' '; echo
echo "errors: $(errs s4)"; pkill -f "llama-server.*--port 8094"; sleep 3
echo "########## S5: 1 GPU, dm=2, lib-new26: rewind repeat (reproducibility vs s2) ##########"
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh s5 $A1 || exit 1
rewind s5 3; echo "errors: $(errs s5)"; pkill -f "llama-server.*--port 8094"; sleep 3
cmprw s2 s5
echo "########## S6: 1 GPU, dm=0, lib-new26: rewind repeat (reproducibility vs r4_0) ##########"
NEXT_DEVICE_MASK=0 MINI_DEVS=$G3 bash mini_run4.sh s6 $A1 || exit 1
rewind s6 3; echo "errors: $(errs s6)"; pkill -f "llama-server.*--port 8094"; sleep 3
cmprw r4_0 s6; cmprw s5 s6
echo "########## S7/S8: 1 GPU, NO draft (--spec-draft-n-max 0), dm=2 vs dm=0, probs at every position ##########"
for dm in 2 0; do
  NEXT_DEVICE_MASK=$dm MINI_DEVS=$G3 bash mini_run4.sh s7_$dm $A1 --spec-draft-n-max 0 || exit 1
  grep -a "speculative\|draft" logs/s7_$dm.log | grep -av "spec-draft" | head -n 2 | cut -c20-140
  rewind s7_$dm 5; echo "errors: $(errs s7_$dm)"; pkill -f "llama-server.*--port 8094"; sleep 3
done
cmprw s7_2 s7_0
echo "VALIDATE4 DONE $(date +%T)"
