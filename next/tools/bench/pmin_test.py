import json, time, urllib.request, glob, hashlib, os, sys
URL='http://127.0.0.1:8093'; OPT='/home/douya/qwen-3.8-next-opt/opt'
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r=urllib.request.Request(URL+path, data=json.dumps(o).encode(), headers={'Content-Type':'application/json'})
    return json.loads(op.open(r, timeout=3600).read())
S='/home/douya/src/llama.cpp-flashnext-20260913'
text=''.join(open(p, errors='ignore').read() for p in sorted(glob.glob(S+'/src/*.cpp')))
toks=post({'content':text},'/tokenize')['tokens'][:70000]
instr=post({'content':'\n\n// 请用中文总结上面代码里最重要的五个函数，并说明它们之间的调用关系。\n'},'/tokenize')['tokens']
prompt=toks+instr
def setopt(n, pmin):
    open(OPT+'/spec_n_max','w').write(str(n))
    if pmin is None:
        if os.path.exists(OPT+'/spec_p_min'): os.remove(OPT+'/spec_p_min')
    else: open(OPT+'/spec_p_min','w').write(str(pmin))
    time.sleep(1.3)
for (n, pmin, seed) in [(5,0.6,7),(5,0.8,7),(5,0.9,7),(4,0.8,7),(3,None,7),(4,None,11),(3,None,11),(5,0.8,11)]:
    setopt(n, pmin)
    r=post(dict(prompt=prompt, n_predict=400, cache_prompt=True, ignore_eos=True, temperature=1.0, top_k=20, top_p=0.95, seed=seed))
    t=r['timings']; steps=t['predicted_n']-t['draft_n_accepted']
    print(json.dumps(dict(n=n, pmin=pmin, seed=seed, tps=round(t['predicted_per_second'],2), draft=t['draft_n'], acc=t['draft_n_accepted'], steps=steps, drafts_per_step=round(t['draft_n']/steps,2), acc_rate=round(t['draft_n_accepted']/t['draft_n'],2), md5=hashlib.md5(r['content'].encode()).hexdigest()[:8], head=r['content'][:40].replace('\n',' '))), flush=True)
setopt(4, None)
