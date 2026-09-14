# fetch top-5 token probabilities for the first n positions of the 2048-token mini prompt (greedy); saves raw + parsed json
import json, sys, urllib.request, glob, math
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 14
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'}); return json.loads(op.open(r, timeout=600).read())
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
toks = post({'content': text}, '/tokenize')['tokens'][:2048]
r = post(dict(prompt=toks, n_predict=n, cache_prompt=False, temperature=0.0, ignore_eos=True, n_probs=5, backend_sampling=False))
json.dump(r, open(f'/home/douya/tests/mini/probs-raw-{tag}.json', 'w'))
out = []
for c in r.get('completion_probabilities', []):
    cands = c.get('top_logprobs') or c.get('top_probs') or c.get('probs') or []
    row = []
    for p in cands:
        tid = p.get('id', p.get('token', p.get('tok_str')))
        prob = p.get('prob') if 'prob' in p else (math.exp(p['logprob']) if 'logprob' in p else None)
        row.append((tid, prob))
    out.append(row)
json.dump({'tokens': post({'content': r['content']}, '/tokenize')['tokens'], 'probs': out}, open(f'/home/douya/tests/mini/probs-{tag}.json', 'w'))
print(tag, 'positions', len(out), 'first', out[0][:3] if out else None, 'keys', list(r.get('completion_probabilities', [{}])[0].keys()) if r.get('completion_probabilities') else None)
