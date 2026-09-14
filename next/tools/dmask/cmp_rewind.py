#!/usr/bin/env python3
"""compare two rewind-<tag>.json files: token common prefix and first differing probability position. usage: cmp_rewind.py a b"""
import json, sys
ta, tb = sys.argv[1], sys.argv[2]
a = json.load(open(f'/home/douya/tests/mini/rewind-{ta}.json')); b = json.load(open(f'/home/douya/tests/mini/rewind-{tb}.json'))
for k in ('A', 'A2', 'A3'):
    x, y = a.get(k) or [], b.get(k) or []
    pa, pb = a.get(k + '_probs') or [], b.get(k + '_probs') or []
    p = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    q = next((i for i, (u, v) in enumerate(zip(pa, pb)) if u != v), min(len(pa), len(pb)))
    verdict = 'IDENTICAL' if (x == y and pa == pb and len(x) > 0) else 'DIFF'
    print('%s vs %s %s: %s tokens prefix %d / %d %d; probs first diff at %d of %d (%d/%d positions carry probs); prompt_n %s/%s' % (ta, tb, k, verdict, p, len(x), len(y), q, min(len(pa), len(pb)), sum(1 for r in pa if r), sum(1 for r in pb if r), a.get(k + '_prompt_n'), b.get(k + '_prompt_n')))
    if verdict == 'DIFF' and q < min(len(pa), len(pb)) and pa[q] and pb[q]:
        print('   first prob diff: %s | %s' % ([f'{t[0]}:{t[1]:.6g}' if isinstance(t[1], float) else str(t) for t in pa[q][:3]], [f'{t[0]}:{t[1]:.6g}' if isinstance(t[1], float) else str(t) for t in pb[q][:3]]))
