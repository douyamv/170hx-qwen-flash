# fill the Q8A_TABLE block in q8a.cu from gen_table.py output lines ("    {N, K, B, R, s},")
import sys, re
src, lines_file = sys.argv[1], sys.argv[2]
s = open(src).read()
lines = [l for l in open(lines_file).read().splitlines() if l.strip().startswith('{')]
a = s.index("//Q8A_TABLE_BEGIN\n") + len("//Q8A_TABLE_BEGIN\n"); b = s.index("//Q8A_TABLE_END")
s = s[:a] + "\n".join(lines) + ("\n" if lines else "") + s[b:]
open(src, 'w').write(s); print("table entries:", len(lines))
