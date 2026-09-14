#!/bin/bash
# lib-new26 (= lib-new25 + scheduler placement rule for the mask op): placement proof on 2 GPUs, equality vs the
# host-mask baselines of round 2 (r1_0 / r3_0 / r4_0), rewinds.
T=/home/douya/tests; cd $T/mini || exit 1
export NEXT_Q8A_MAX_MB=300 NEXT_TOPK_NOSORT=1 TRACER=$T/mini/lib-new26/libnext-trace.so LIBDIR=lib-new26
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3; C0=GPU-524355f6-cdea-1376-4724-94c1d248c7d8
A1="--ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor ^token_embd\.weight$=CUDA0 --spec-draft-backend-sampling"
A2="--tensor-split 2,2 --spec-draft-device CUDA1 --override-tensor ^token_embd\.weight$=CUDA1 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-backend-sampling"
pref() { python3 -c "
import json; a=json.load(open('stream-$1.json')); b=json.load(open('stream-$2.json'))
p=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v), min(len(a),len(b))); print('$1 vs $2: common prefix', p, '/', len(a), len(b), 'IDENTICAL' if a==b and len(a)>0 else 'DIFF')"; }
cmpprobs() { python3 -c "
import json; a=json.load(open('probs-$1.json')); b=json.load(open('probs-$2.json'))
print('probs $1 vs $2:', 'IDENTICAL' if a['probs']==b['probs'] and len(a['probs'])>0 else 'DIFF', len(a['probs']), len(b['probs']), 'tokens', 'same' if a['tokens']==b['tokens'] else 'differ')"; }
errs() { grep -a "CUDA error\|ERR\|abort\|GGML_ASSERT" logs/$1.log | grep -av CORS | head -3 | cut -c1-160; }
touch opt/spec_adaptive_off
echo "########## S1: 2 GPUs (CUDA0 + GPU3), dm=2, lib-new26: mask placement + equality vs r3_0 ##########"
GGML_SCHED_DEBUG=1 NEXT_DEVICE_MASK=2 NEXT_DEVICE_MASK_VERBOSE=1 MINI_DEVS=$C0,$G3 bash mini_run4.sh s1 $A2 || exit 1
python3 stream_md5.py s1_2048 2048 96 2>/dev/null | tail -1 | cut -c1-100
python3 $T/probe_probs.py s1 14 2>&1 | cut -c1-120
echo "-- mask op assignments (node lines from GGML_SCHED_DEBUG, unique):"
grep -a "attn_kq_mask_dev" logs/s1.log | grep -a "node #" | sed 's/^.*node #[ 0-9]*//' | cut -c1-150 | sort | uniq -c | head -n 8
echo "-- cell_pos copies (inputs listed in split headers mentioning cellpos):"; grep -a "cache_cellpos" logs/s1.log | grep -a "SPLIT\|input" | head -n 3 | cut -c1-150
echo "-- splits per graph (last graph):"; grep -a "## SPLIT #" logs/s1.log | tail -n 1 | cut -c1-80
grep -a "device mask:" logs/s1.log | head -n 1 | cut -c20-140
echo "errors: $(errs s1)"; pkill -f "llama-server.*--port 8094"; sleep 3
pref r3_0_2048 s1_2048; cmpprobs r3_0 s1
echo "########## S2: 1 GPU (GPU3), dm=2, lib-new26: equality vs r1_0 + rewinds vs r4_0 ##########"
NEXT_DEVICE_MASK=2 MINI_DEVS=$G3 bash mini_run4.sh s2 $A1 || exit 1
for n in 2048 16384; do python3 stream_md5.py s2_$n $n 96 2>/dev/null | tail -1 | cut -c1-100; done
python3 $T/probe_probs.py s2 48 2>&1 | cut -c1-120
python3 - s2 <<'PY'
import json, urllib.request, glob, sys
tag = sys.argv[1]
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
json.dump(out, open(f'/home/douya/tests/mini/rewind-{tag}.json', 'w')); print('%s prompt_n A=%s A2=%s A3=%s' % (tag, out['A_prompt_n'], out['A2_prompt_n'], out['A3_prompt_n']))
PY
echo "errors: $(errs s2)"; pkill -f "llama-server.*--port 8094"; sleep 3
for n in 2048 16384; do pref r1_0_$n s2_$n; done
echo "########## S3: 1 GPU (GPU3), dm=0, lib-new26: 48-position CPU-sampling probe baseline ##########"
NEXT_DEVICE_MASK=0 MINI_DEVS=$G3 bash mini_run4.sh s3 $A1 || exit 1
python3 $T/probe_probs.py s3 48 2>&1 | cut -c1-120
echo "errors: $(errs s3)"; pkill -f "llama-server.*--port 8094"; sleep 3
cmpprobs s3 s2
python3 - <<'PY'
import json
a = json.load(open('/home/douya/tests/mini/rewind-s2.json')); b = json.load(open('/home/douya/tests/mini/rewind-r4_0.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a[k], b[k]
    if isinstance(x, list) and isinstance(y, list):
        p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        print('%s: tokens s2 vs r4_0 common prefix %d / %d %d; probs %s' % (k, p, len(x), len(y), 'IDENTICAL' if a[k+'_probs'] == b[k+'_probs'] and a[k+'_probs'] else 'DIFF'))
    else: print(k, 'error:', x if not isinstance(x, list) else '', y if not isinstance(y, list) else '')
PY
echo "VALIDATE3 DONE $(date +%T)"
