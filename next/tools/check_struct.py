#!/usr/bin/env python3
"""compare the huihui UD-Q4_K_XL shards with the base ones: exact file sizes + GGUF header structure (tensor names,
types, dims, split metadata). Pure python (numpy is broken on the box)."""
import struct, os, sys
B = '/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000%d-of-00004.gguf'
H = '/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000%d-of-00004.gguf'
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def parse(path):
    data = open(path, 'rb').read(64 << 20); pos = 0
    def u32():
        nonlocal pos; v = struct.unpack_from('<I', data, pos)[0]; pos += 4; return v
    def u64():
        nonlocal pos; v = struct.unpack_from('<Q', data, pos)[0]; pos += 8; return v
    def string():
        nonlocal pos; n = u64(); s = data[pos:pos + n]; pos += n; return s.decode(errors='replace')
    def value(t):
        nonlocal pos
        if t == 8: return string()
        if t == 9:
            et = u32(); n = u64(); return [value(et) for _ in range(n)]
        v = data[pos:pos + SIZES[t]]; pos += SIZES[t]; return v
    assert data[:4] == b'GGUF', 'bad magic'
    pos = 4; ver = u32(); nt = u64(); nkv = u64()
    kv = {}
    for _ in range(nkv):
        k = string(); t = u32(); kv[k] = value(t)
    tensors = [(string(), [u64() for _ in range(u32())], u32(), u64()) for _ in range(nt)]
    tensors = [(n, d, t) for (n, d, t, off) in tensors]
    return ver, kv, tensors
ok = True
for i in (1, 2, 3, 4):
    sb, sh = os.path.getsize(B % i), os.path.getsize(H % i)
    vb, kb, tb = parse(B % i); vh, kh, th = parse(H % i)
    same_t = tb == th
    kdiff = sorted(k for k in set(kb) | set(kh) if kb.get(k) != kh.get(k))
    print('shard %d: size base %d huihui %d %s | tensors %d/%d %s | kv keys differing: %s' % (i, sb, sh, 'SAME' if sb == sh else 'DIFF', len(tb), len(th), 'IDENTICAL' if same_t else 'DIFF', kdiff[:6]))
    if not same_t:
        for a, b in zip(tb, th):
            if a != b: print('   first tensor diff:', a, '|', b); break
    ok &= (sb == sh) and same_t and all(k.startswith('general.') for k in kdiff)
import hashlib
# cheap content check: the weights must actually differ from the base (not a re-upload): sample 64 KB at 3 offsets of shard 2
diff = 0
with open(B % 2, 'rb') as fb, open(H % 2, 'rb') as fh:
    for off in (1 << 30, 10 << 30, 40 << 30):
        fb.seek(off); fh.seek(off); diff += sum(1 for x, y in zip(fb.read(65536), fh.read(65536)) if x != y)
print('sampled weight bytes differing from the base (shard 2, 3x64KB):', diff)
print('STRUCTURE OK' if ok and diff > 0 else 'STRUCTURE MISMATCH')
