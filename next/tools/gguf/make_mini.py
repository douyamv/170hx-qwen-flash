#!/usr/bin/env python3
"""Build a 4-layer, PLE-free mini GGUF of Qwen3.8-Flash-Next for single-GPU timing experiments."""
import sys, time, re
sys.path.insert(0, '/home/douya/src/llama.cpp-flashnext-20260913/gguf-py')
import numpy as np, gguf
from gguf import GGUFReader, GGUFWriter
BASE = '/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/'
SRC1 = BASE + 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
SRC2 = BASE + 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf'
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DST = BASE + f'mini/Qwen3.8-Flash-Next-mini{NL}-noPLE.gguf'
t0 = time.time()
r1 = GGUFReader(SRC1); r2 = GGUFReader(SRC2)
arch = r1.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = GGUFWriter(DST, arch=arch, endianess=r1.endianess)
for f in r1.fields.values():
    n = f.name
    if n == gguf.Keys.General.ARCHITECTURE or n.startswith('GGUF.') or n.startswith('split.') or '.ple.' in n:
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents()
    if n == f'{arch}.block_count': val = NL
    if n == f'{arch}.attention.compress_ratios': val = val[:NL]
    w.add_key_value(n, val, vt, sub_type=st)
keep = []
for t in r2.tensors:
    m = re.match(r'blk\.(\d+)\.', t.name)
    if m:
        if int(m.group(1)) >= NL or '.ple_' in t.name: continue
    elif t.name == 'per_layer_token_embd.weight':
        continue
    keep.append(t)
tot = sum(t.data.nbytes for t in keep)
print(f'{len(keep)} tensors, {tot/2**30:.2f} GiB', flush=True)
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
done = 0
for t in keep:
    w.write_tensor_data(t.data, tensor_endianess=r2.endianess)
    done += t.data.nbytes
    print(f'{done/2**30:6.2f} GiB  {time.time()-t0:5.0f}s  {t.name}', flush=True)
w.close()
chk = GGUFReader(DST)
print('verify: tensors', len(chk.tensors), 'kv', len(chk.fields), 'block_count', chk.fields[f'{arch}.block_count'].contents(), f'{time.time()-t0:.0f}s', flush=True)
