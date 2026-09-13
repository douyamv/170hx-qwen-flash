#!/usr/bin/env python3
"""Post-deploy validation on production (:8093). usage: prod_sweep.py short|long|bs"""
import json, sys, time, urllib.request, glob, hashlib, os
URL = 'http://127.0.0.1:8093'; OPT = '/home/douya/qwen-3.8-next-opt/opt'; OUT = '/home/douya/tests/prod-sweep.jsonl'
mode = sys.argv[1] if len(sys.argv) > 1 else 'short'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request(URL + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(op.open(r, timeout=3600).read())
def setopt(n_max, q8head):
    open(OPT + '/spec_n_max', 'w').write(str(n_max)); f = OPT + '/draft_head_target'
    if q8head: open(f, 'w').close()
    elif os.path.exists(f): os.remove(f)
    time.sleep(1.3)
def rec(tag, cfg, r):
    t = r['timings']
    d = dict(tag=tag, cfg=cfg, gen_tps=round(t['predicted_per_second'], 2), gen_n=t['predicted_n'], prompt_n=t.get('prompt_n'), prompt_ms=round(t.get('prompt_ms', 0)),
             draft_n=t.get('draft_n'), draft_acc=t.get('draft_n_accepted'), md5=hashlib.md5(r['content'].encode()).hexdigest()[:10], head=r['content'][:50].replace('\n', ' '), time=time.strftime('%H:%M:%S'))
    print(json.dumps(d, ensure_ascii=False), flush=True); open(OUT, 'a').write(json.dumps(d, ensure_ascii=False) + '\n'); return d
S = '/home/douya/src/llama.cpp-flashnext-20260913'
if mode in ('short', 'bs'):
    text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob(S + '/tools/server/*.cpp')))
    toks = post({'content': text}, '/tokenize')['tokens'][:2000]
    instr = post({'content': '\n\n// 请用中文逐步解释上面这段代码的整体结构和关键函数的作用。\n'}, '/tokenize')['tokens']
else:
    text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob(S + '/src/*.cpp')))
    toks = post({'content': text}, '/tokenize')['tokens'][:70000]
    instr = post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数，并说明它们之间的调用关系。\n'}, '/tokenize')['tokens']
prompt = toks + instr
t0 = time.time(); r = post(dict(prompt=prompt, n_predict=1, cache_prompt=True)); print(mode, 'prefill', round(time.time() - t0, 1), 's, prompt_n', r['timings'].get('prompt_n'), flush=True)
if mode == 'bs':
    setopt(3, False)
    for bs in (False, True, False, True):
        r = post(dict(prompt=prompt, n_predict=300, cache_prompt=True, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=7, backend_sampling=bs)); rec('bs_' + str(bs), '3/q4', r)
    sys.exit()
md5s = {}
for (n, q8) in [(3, False), (5, False), (5, True), (4, False)]:
    setopt(n, q8); r = post(dict(prompt=prompt, n_predict=300, cache_prompt=True, ignore_eos=True, temperature=0)); d = rec(mode + '_greedy', f'{n}/{"q8" if q8 else "q4"}', r); md5s[d['cfg']] = d['md5']
print('GREEDY MD5 EQUAL:', len(set(md5s.values())) == 1, md5s, flush=True)
for (n, q8) in [(3, False), (4, False), (5, False), (3, True), (5, True), (3, False)]:
    setopt(n, q8); r = post(dict(prompt=prompt, n_predict=400, cache_prompt=True, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=7)); rec(mode + '_sample', f'{n}/{"q8" if q8 else "q4"}', r)
setopt(3, False)
