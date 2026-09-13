#!/usr/bin/env python3
"""Rename the MTP draft GGUF's blk.48 -> blk.NL so it pairs with the NL-layer mini target."""
import sys, time
sys.path.insert(0, '/home/douya/src/llama.cpp-flashnext-20260913/gguf-py')
import gguf
from gguf import GGUFReader, GGUFWriter
BASE = '/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/'
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SRC = BASE + 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf'
DST = BASE + f'mini/mtp-mini{NL}-shared-Q8_0-q4head.gguf'
t0 = time.time()
r = GGUFReader(SRC)
arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = GGUFWriter(DST, arch=arch, endianess=r.endianess)
for f in r.fields.values():
    n = f.name
    if n == gguf.Keys.General.ARCHITECTURE or n.startswith('GGUF.'):
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents()
    if n == f'{arch}.block_count': val = NL + 1
    if n == f'{arch}.attention.compress_ratios': val = val[:NL] + [val[48]]
    w.add_key_value(n, val, vt, sub_type=st)
for t in r.tensors:
    assert t.name.startswith('blk.48.'), t.name
    w.add_tensor_info('blk.%d.' % NL + t.name[len('blk.48.'):], t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
chk = GGUFReader(DST)
print('verify:', len(chk.tensors), 'tensors', [t.name for t in chk.tensors][:3], 'block_count', chk.fields[f'{arch}.block_count'].contents(), f'{time.time()-t0:.0f}s', flush=True)
