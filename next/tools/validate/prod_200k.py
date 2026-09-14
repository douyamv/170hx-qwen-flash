# 200K-context probe on production (:8093): prefill time, greedy + sampled decode, short-turn TTFT
import json, time, urllib.request, glob, sys
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion', timeout=3600):
    r = urllib.request.Request('http://127.0.0.1:8093' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=timeout).read())
files = sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/ggml/src/ggml-cuda/*.cu') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/tools/server/*.cpp') + glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/common/*.cpp'))
text = ''.join(open(p, errors='ignore').read() for p in files)
toks = post({'content': text}, '/tokenize')['tokens']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
print('corpus tokens', len(toks)); toks = toks[:N] + post({'content': '\n\n// 请用中文总结上面代码里最重要的五个函数。\n'}, '/tokenize')['tokens']
t0 = time.time(); r = post(dict(prompt=toks, n_predict=1, cache_prompt=True)); print('prefill %d tokens: %.1f s' % (len(toks), time.time() - t0))
for cfg, sp in (('greedy 4/q4', dict(temperature=0.0)), ('sampled 4/q4', dict(temperature=1.0, top_k=20, top_p=0.95, seed=7))):
    open('/home/douya/qwen-3.8-next-opt/opt/spec_n_max', 'w').write('4\n')
    r = post(dict(prompt=toks, n_predict=300, cache_prompt=True, ignore_eos=True, **sp)); t = r['timings']
    print('%-14s %6.1f tok/s  prompt_ms %4.0f  draft %d acc %d  %s' % (cfg, t['predicted_per_second'], t['prompt_ms'], t['draft_n'], t['draft_n_accepted'], r['content'][:50].replace('\n', ' ')))
# short-turn TTFT at 200K: 17 new tokens, 8 generated
turn = toks + post({'content': ' 再补充一点：这些函数之间的调用关系是什么？'}, '/tokenize')['tokens']
for i in range(2):
    t0 = time.time(); r = post(dict(prompt=turn, n_predict=8, cache_prompt=True, temperature=0.0)); t = r['timings']
    print('turn %d: wall %.2f s  prompt_n %d prompt_ms %.0f  cache_n %d' % (i + 1, time.time() - t0, t['prompt_n'], t['prompt_ms'], r.get('tokens_cached', -1)))
    turn = turn + post({'content': r['content'] + ' 继续。'}, '/tokenize')['tokens']
