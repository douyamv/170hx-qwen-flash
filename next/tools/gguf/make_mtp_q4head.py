#!/usr/bin/env python3
"""Copy the MTP GGUF and add a draft-only Q4_0 LM head quantized from the target output.weight."""
import sys, time
sys.path.insert(0, '/home/douya/src/llama.cpp-flashnext-20260913/gguf-py')
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize

BASE = '/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/'
SRC_MTP = BASE + 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
DST_MTP = BASE + 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q4head.gguf'
HEAD_NAME = 'blk.48.nextn.shared_head_head.weight'

t0 = time.time()
out_t = None
for i in (2, 3, 4):
    r = GGUFReader(BASE + f'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf')
    for t in r.tensors:
        if t.name == 'output.weight':
            out_t = t; break
    if out_t is not None:
        print('found output.weight in shard', i, out_t.tensor_type.name, list(out_t.shape), out_t.data.shape, flush=True)
        break
assert out_t is not None and out_t.tensor_type == Q.Q8_0
n_vocab, n_embd = out_t.data.shape[0], int(out_t.shape[0])
q4 = np.empty((n_vocab, n_embd // 32 * 18), dtype=np.uint8)
chunk = 8192
for s in range(0, n_vocab, chunk):
    e = min(n_vocab, s + chunk)
    f = dequantize(np.asarray(out_t.data[s:e]), Q.Q8_0)
    q4[s:e] = quantize(f.reshape(e - s, n_embd).astype(np.float32), Q.Q4_0)
    if s % (chunk * 8) == 0:
        print(f'  quantized rows {e}/{n_vocab}  {time.time()-t0:.0f}s', flush=True)
# spot-check fidelity: cosine similarity of a few rows
idx = np.random.default_rng(0).choice(n_vocab, 64, replace=False)
ref = dequantize(np.asarray(out_t.data[idx]), Q.Q8_0).reshape(64, n_embd)
new = dequantize(q4[idx], Q.Q4_0).reshape(64, n_embd)
cos = (ref * new).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(new, axis=1) + 1e-12)
print(f'Q4_0 vs Q8_0 cosine: min {cos.min():.5f} mean {cos.mean():.5f}', flush=True)

reader = GGUFReader(SRC_MTP)
assert all(t.name != HEAD_NAME for t in reader.tensors), 'head already present'
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = GGUFWriter(DST_MTP, arch=arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
for t in reader.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.add_tensor_info(HEAD_NAME, q4.shape, q4.dtype, q4.nbytes, Q.Q4_0)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.write_tensor_data(q4, tensor_endianess=reader.endianess)
writer.close()
print(f'wrote {DST_MTP} in {time.time()-t0:.0f}s', flush=True)

chk = GGUFReader(DST_MTP)
names = {t.name: t for t in chk.tensors}
h = names[HEAD_NAME]
print('verify:', len(chk.tensors), 'tensors (src', len(reader.tensors), ')', HEAD_NAME, h.tensor_type.name, list(h.shape),
      'kv', len(chk.fields), '(src', len(reader.fields), ')', flush=True)
