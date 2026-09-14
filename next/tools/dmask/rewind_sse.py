#!/usr/bin/env python3
"""rewind scenario (A: 16384-token prompt, A2: keep 16084 + 300 new, A3: keep 12000 of A2 + 200 new) via SSE streaming
(the text survives the server's final chat-parse failure). CPU sampling, greedy, n_probs per position.
usage: rewind_sse.py <tag> <n_probs>  -> /home/douya/tests/mini/rewind-<tag>.json"""
import json, sys, glob, urllib.request
tag, nprobs = sys.argv[1], int(sys.argv[2])
URL = 'http://127.0.0.1:8094'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request(URL + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(op.open(r, timeout=3600).read())
def gen(prompt, n):
    req = urllib.request.Request(URL + '/completion', data=json.dumps(dict(prompt=prompt, n_predict=n, cache_prompt=True, ignore_eos=True, temperature=0.0, return_tokens=True, n_probs=nprobs, backend_sampling=False, stream=True)).encode(), headers={'Content-Type': 'application/json'})
    ids, probs, timings, err = [], [], None, ''
    try:
        with op.open(req, timeout=3600) as resp:
            for line in resp:
                line = line.decode(errors='ignore').strip()
                if not line.startswith('data: '): continue
                try: d = json.loads(line[6:])
                except Exception: continue
                ids += d.get('tokens', [])
                for c in d.get('completion_probabilities', []) or []:
                    cands = c.get('top_logprobs') or c.get('top_probs') or c.get('probs') or []
                    probs.append([(p.get('id', p.get('token')), p.get('prob', p.get('logprob'))) for p in cands])
                if d.get('timings'): timings = d['timings']
    except Exception as e:
        err = type(e).__name__ + ' ' + str(e)[:60]
    return ids, probs, (timings or {}).get('prompt_n'), err
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens']
A = toks[:16384]; A2 = A[:16084] + toks[30000:30300]; A3 = A2[:12000] + toks[40000:40200]
out = {}
for k, prompt, n in (('A', A, 24), ('A2', A2, 48), ('A3', A3, 48)):
    ids, probs, pn, err = gen(prompt, n)
    out[k] = ids; out[k + '_probs'] = probs; out[k + '_prompt_n'] = pn; out[k + '_err'] = err
    print('%s %s: prompt_n=%s tokens=%d positions_with_probs=%d %s' % (tag, k, pn, len(ids), sum(1 for r in probs if r), err), flush=True)
json.dump(out, open('/home/douya/tests/mini/rewind-%s.json' % tag, 'w'))
