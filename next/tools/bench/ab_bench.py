#!/usr/bin/env python3
"""A/B benchmark for the optimized llama-server test build (runs against 127.0.0.1:8093)."""
import glob, json, os, re, sys, time, urllib.request
URL = 'http://127.0.0.1:8093'
OPT = os.environ.get('NEXT_OPT_DIR', '/home/douya/qwen-3.8-next-opt/opt')
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/douya/tests/ab-results.jsonl'
SRC = '/home/douya/src/llama.cpp-flashnext-20260913'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path, obj, timeout=7200):
    req = urllib.request.Request(URL + path, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())
def record(tag, **kv):
    kv['tag'] = tag; kv['time'] = time.strftime('%H:%M:%S')
    line = json.dumps(kv, ensure_ascii=False)
    print(line, flush=True); open(OUT, 'a').write(line + '\n')
def toggle(name, on, content=None):
    p = f'{OPT}/{name}'
    if on:
        with open(p, 'w') as f: f.write(content or '1')
    elif os.path.exists(p): os.remove(p)
    time.sleep(1.5)   # switches are re-read at most once per second
def run(tag, tokens, n_predict, **extra):
    t0 = time.time()
    r = post('/completion', dict(prompt=tokens, n_predict=n_predict, cache_prompt=True, **extra))
    t = r.get('timings', {})
    record(tag, wall=round(time.time() - t0, 2), prompt_n=t.get('prompt_n'), prompt_tps=round(t.get('prompt_per_second') or 0, 1),
           gen_n=t.get('predicted_n'), gen_tps=round(t.get('predicted_per_second') or 0, 2),
           draft_n=t.get('draft_n'), draft_acc=t.get('draft_n_accepted'))
    return r
srcs = sorted(glob.glob(SRC + '/src/*.cpp')) + sorted(glob.glob(SRC + '/common/*.cpp')) + sorted(glob.glob(SRC + '/ggml/src/*.c'))
text = ''.join(open(p, errors='ignore').read() for p in srcs)
toks = post('/tokenize', {'content': text})['tokens']
record('corpus', tokens=len(toks))
def piece(i, n):   # disjoint slices so every prompt is a fresh prefill
    return toks[i:i + n]
cur = 0
def take(n):
    global cur
    p = piece(cur, n); cur += n; return p

run('warmup', take(512), 16)
# 1) shallow prefill A/B (MoE grid), 8K fresh prompts, with routing stats on the optimized run
toggle('mmq_grid_off', True)
run('prefill8k_grid_off', take(8192), 1)
toggle('mmq_grid_off', False); toggle('mmq_stats', True)
run('prefill8k_grid_on', take(8192), 1)
toggle('mmq_stats', False)
toggle('mmq_grid_off', True)
run('prefill8k_grid_off_repeat', take(8192), 1)
toggle('mmq_grid_off', False)
run('prefill8k_grid_on_repeat', take(8192), 1)
# 2) deep prefill (70K fresh) with the grid on, then decode A/B at ~70K
deep = take(70000)
run('prefill70k_grid_on', deep, 1)
greedy = dict(temperature=0.0, top_k=1, n_probs=1, ignore_eos=True)
toggle('qsa_min_kv', True, '131072')
a = run('decode70k_qsa131072', deep + take(64), 384, **greedy)
toggle('qsa_min_kv', True, '4096')
b = run('decode70k_qsa4096', deep + piece(cur - 64, 64), 384, **greedy)
ta = a.get('content', ''); tb = b.get('content', '')
same = sum(1 for x, y in zip(ta, tb) if x == y)
record('decode70k_output_compare', chars_a=len(ta), chars_b=len(tb), prefix_equal=len(os.path.commonprefix([ta, tb])), same_pos=same)
# 3) short context decode sanity with the new defaults
run('decode_short_qsa4096', take(256), 384, **greedy)
toggle('qsa_min_kv', False)
run('decode_short_default', take(256), 384, **greedy)
record('done')
