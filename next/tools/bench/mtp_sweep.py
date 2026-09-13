#!/usr/bin/env python3
import glob, json, sys, time, urllib.request
URL = 'http://127.0.0.1:8093'; OUT = sys.argv[1]; CTX = int(sys.argv[2])
SRC = '/home/douya/src/llama.cpp-flashnext-20260913'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(obj, path='/completion'):
    req = urllib.request.Request(URL + path, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=7200) as r: return json.loads(r.read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob(SRC + '/tools/server/*.cpp')) + sorted(glob.glob(SRC + '/examples/*/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens']
base = toks[:CTX]
# 让模型真正“写代码/解释代码”,贴近实际使用
instr = post({'content': '\n\n// 请用中文逐步解释上面这段代码的整体结构和关键函数的作用。\n'}, '/tokenize')['tokens']
prompt = base + instr
post(dict(prompt=prompt, n_predict=1, cache_prompt=True))   # 预填充一次,后面都命中缓存
configs = []
for samp in ('greedy', 'default'):
    for n_max in (1, 2, 3, 4, 5):
        configs.append((samp, n_max, 0.0))
    for p_min in (0.5, 0.8):
        configs.append((samp, 4, p_min))
for samp, n_max, p_min in configs:
    req = dict(prompt=prompt, n_predict=256, cache_prompt=True, ignore_eos=True, seed=1234)
    req['speculative.n_max'] = n_max; req['speculative.n_min'] = 0; req['speculative.p_min'] = p_min
    if samp == 'greedy': req.update(temperature=0.0, top_k=1)
    r = post(req); t = r.get('timings', {})
    acc = (t.get('draft_n_accepted') or 0) / max(1, t.get('draft_n') or 1)
    line = dict(ctx=CTX, samp=samp, n_max=n_max, p_min=p_min, gen_tps=round(t.get('predicted_per_second', 0), 2),
                draft_n=t.get('draft_n'), draft_acc=t.get('draft_n_accepted'), acc=round(acc, 3))
    print(json.dumps(line), flush=True); open(OUT, 'a').write(json.dumps(line) + '\n')
