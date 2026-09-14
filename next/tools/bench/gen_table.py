# best (R, s) per (K, N, B) from q8a_sweep.csv -> C++ table lines + a summary vs the v5 plan (R=2, s=1 / K>=4096 rule)
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
best = {}; allp = collections.defaultdict(dict)
for r in rows:
    key = (int(r['K']), int(r['N']), int(r['B'])); us = float(r['us']); plan = (int(r['R']), int(r['s']))
    allp[key][plan] = us
    if key not in best or us < best[key][1]: best[key] = (plan, us)
def v5plan(K, N):
    nchunk = K // 16; warps_n = (N + 1) // 2; s = 1
    if K >= 4096:
        while warps_n * s < 2048 and s < 8 and nchunk % (s * 2) == 0 and nchunk // (s * 2) >= 64: s *= 2
    return (2, s)
print('%-6s %-6s %2s  %-8s %8s  %-8s %8s  %6s' % ('K', 'N', 'B', 'best', 'us', 'v5plan', 'us', 'gain'))
lines = []
for key in sorted(best):
    K, N, B = key; (plan, us) = best[key]; vp = v5plan(K, N); vus = allp[key].get(vp, float('nan'))
    gain = (vus - us) / vus * 100 if vus == vus else float('nan')
    print('%-6d %-6d %2d  R%d s%d    %8.2f  R%d s%d    %8.2f  %5.1f%%' % (K, N, B, plan[0], plan[1], us, vp[0], vp[1], vus, gain))
    if gain > 3: lines.append('    {%d, %d, %d, %d, %d},' % (N, K, B, plan[0], plan[1]))
print('\n// measured on CMP 170HX (70 SMs): {N, K, B, R, splitk}, only where > 3% faster than the R=2 rule')
print('\n'.join(lines))
