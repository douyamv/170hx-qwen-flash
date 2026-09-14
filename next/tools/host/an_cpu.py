# host timeline from a markers-only trace: per DECODE_TARGET step, phases (ms) and the gaps; usage: an_cpu.py trace.csv [n_steps_to_print]
import csv, sys, bisect, collections
fn = sys.argv[1]; nprint = int(sys.argv[2]) if len(sys.argv) > 2 else 2
cpu = []
for row in csv.reader(open(fn, newline='')):
    if len(row) < 9 or row[0] != 'CPU': continue
    try: cpu.append((int(row[2]), int(row[3]), row[8]))
    except: pass
cpu.sort()
tg = [c for c in cpu if c[2] == 'DECODE_TARGET']
if len(tg) < 5: print('only', len(tg), 'DECODE_TARGET markers'); sys.exit()
steady = tg[2:-1]; N = len(steady)
cs = [c[0] for c in cpu]
tot = collections.Counter(); cnt = collections.Counter(); period = 0
for (s, e, _) in steady:
    nxt = tg[tg.index((s, e, _)) + 1][0]; period += nxt - s
    a = bisect.bisect_left(cs, s); b = bisect.bisect_left(cs, nxt)
    for x in cpu[a:b]: tot[x[2]] += x[1] - x[0]; cnt[x[2]] += 1
print('steps %d, period %.2f ms (untraced-equivalent: markers only)' % (N, period / N / 1e6))
print('%-20s %8s %6s' % ('marker', 'ms/step', 'n/step'))
for k, v in sorted(tot.items(), key=lambda x: -x[1]): print('%-20s %8.2f %6.1f' % (k, v / N / 1e6, cnt[k] / N))
for (s, e, _) in steady[N // 2:N // 2 + nprint]:
    nxt = tg[tg.index((s, e, _)) + 1][0]
    a = bisect.bisect_left(cs, s); b = bisect.bisect_left(cs, nxt)
    print('--- step timeline (offset ms, dur ms, marker) ---')
    for x in cpu[a:b]: print('  %7.2f %7.2f %s' % ((x[0] - s) / 1e6, (x[1] - x[0]) / 1e6, x[2]))
    # uncovered time: period minus the union of top-level markers (DECODE_TARGET, SRV_*, SPEC_*)
    top = [x for x in cpu[a:b] if x[2] in ('SRV_DECODE', 'SRV_DRAFT', 'SRV_ACCEPT', 'SRV_SEND', 'SRV_SAMPLE', 'SRV_CKPT', 'SPEC_PROCESS')]
    covered = 0; last = s
    for x in sorted(top):
        st = max(x[0], last); covered += max(0, x[1] - st); last = max(last, x[1])
    print('  covered by server phases %.2f ms of %.2f -> uncovered %.2f ms' % (covered / 1e6, (nxt - s) / 1e6, (nxt - s - covered) / 1e6))
