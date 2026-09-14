import array, sys
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
a = load(sys.argv[1]); b = load(sys.argv[2])
n = min(len(a), len(b)); md = 0.0; mb = 0.0; sd = 0.0; sb = 0.0
for i in range(n):
    d = abs(a[i] - b[i]); md = max(md, d); mb = max(mb, abs(b[i])); sd += d; sb += abs(b[i])
print('   moea vs mmvq: max|diff| %.4g, max|mmvq| %.4g, mean rel diff %.3e, n=%d' % (md, mb, sd / sb if sb else 0, n))
