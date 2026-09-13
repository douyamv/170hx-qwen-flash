#!/usr/bin/env python3
"""checkpoint/rewind test on the running mini server (MTP on): sizes logged + a rewind that must restore a checkpoint. usage: mini_ckpt_test.py <tag>"""
import json, sys, time, glob, urllib.request, subprocess
tag = sys.argv[1]; T = '/home/douya/tests/mini'; LOG = f'{T}/logs/{tag}.log'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=3600).read())
    except urllib.error.HTTPError as e: return {'err': e.code}
def tok(s): return post({'content': s, 'add_special': False}, '/tokenize')['tokens']
def logtail(pat, n=4): return subprocess.run(['bash', '-c', f'grep -a "{pat}" {LOG} | tail -{n} | cut -c30-190'], capture_output=True, text=True).stdout.strip()
BAN = [[t[0], False] for t in (tok(s) for s in ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>']) if len(t) == 1]
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
A = tok(text)[:16384]
t0 = time.time(); r = post(dict(prompt=A, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0, logit_bias=BAN)); print(f'{tag} A: {time.time()-t0:.1f}s', flush=True)
print(' checkpoints:', logtail('context checkpoint'))
A2 = A[:-200] + tok('以下是完全不同的结尾内容，用于测试回退。' * 40)[:200]
t0 = time.time(); r = post(dict(prompt=A2, n_predict=48, cache_prompt=True, ignore_eos=True, temperature=0, logit_bias=BAN)); dt = time.time() - t0
print(f'{tag} A2 (rewind 200): {dt:.1f}s prompt_n', (r.get('timings') or {}).get('prompt_n'), 'cache_n', (r.get('timings') or {}).get('cache_n'), flush=True)
print(' restore:', logtail('restored context checkpoint\\|forcing full\\|n_past = '))
print(' prompt eval:', logtail('prompt eval time', 2))
print(' errors:', logtail('ERR\\|abort\\|mismatch', 3) or 'none')
