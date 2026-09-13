#!/usr/bin/env python3
"""greedy generation via SSE streaming so the text survives the server's final chat-parse failure. usage: stream_md5.py <tag> <n_ctx_tokens> <n_predict>"""
import json, sys, glob, hashlib, urllib.request, time
tag, n, npred = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
URL = 'http://127.0.0.1:8094'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request(URL + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(op.open(r, timeout=3600).read())
def tok(s): return post({'content': s, 'add_special': False}, '/tokenize')['tokens']
BAN = [[t[0], False] for t in (tok(s) for s in ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>']) if len(t) == 1]
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = tok(text)[:n]
req = urllib.request.Request(URL + '/completion', data=json.dumps(dict(prompt=toks, n_predict=npred, cache_prompt=True, ignore_eos=True, temperature=0, logit_bias=BAN, stream=True, return_tokens=True)).encode(), headers={'Content-Type': 'application/json'})
ids = []; t0 = time.time()
try:
    with op.open(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode(errors='ignore').strip()
            if not line.startswith('data: '): continue
            try: d = json.loads(line[6:])
            except Exception: continue
            ids += d.get('tokens', [])
except Exception as e:
    print(tag, 'stream ended with', type(e).__name__, str(e)[:80])
print(tag, 'ctx', n, 'got', len(ids), 'tokens in %.1fs' % (time.time() - t0), 'md5', hashlib.md5(json.dumps(ids).encode()).hexdigest()[:10], 'head', ids[:12], flush=True)
open('/home/douya/tests/mini/stream-%s.json' % tag, 'w').write(json.dumps(ids))
