#!/bin/bash
# download huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF (UD-Q4_K_XL, 4 shards + mmproj) via hf-mirror, resumable,
# then verify sha256 against the LFS metadata
D=/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
U=https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/resolve/main
mkdir -p $D/UD-Q4_K_XL; cd $D || exit 1
echo "DL START $(date +%T)"; df -h /mnt/slowdisk | tail -n 1
for f in README.md LICENSE .gitattributes; do curl -s -L -m 120 -o $f "$U/$f"; done
for f in mmproj-model-bf16.gguf UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf; do
  dir=$D/$(dirname $f); base=$(basename $f)
  for try in $(seq 1 60); do
    aria2c -q -x 8 -s 8 -k 4M --file-allocation=none -c --max-tries=5 --retry-wait=10 --summary-interval=0 -d $dir -o $base "$U/$f" && break
    echo "retry $try for $base $(date +%T)"; sleep 15
  done
  echo "got $base $(stat -c %s $dir/$base 2>/dev/null) bytes $(date +%T)"
done
echo "DL FILES DONE $(date +%T)"; df -h /mnt/slowdisk | tail -n 1
echo "verifying sha256 ..."
python3 - <<'PY'
import json, urllib.request, hashlib, os
D = '/mnt/slowdisk/AI-archive/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF'
r = urllib.request.Request('https://hf-mirror.com/api/models/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF?blobs=true', headers={'User-Agent': 'curl/8'})
m = json.loads(urllib.request.urlopen(r, timeout=60).read())
ok = True
for s in m['siblings']:
    lfs = s.get('lfs'); name = s['rfilename']; p = os.path.join(D, name)
    if not lfs or not os.path.exists(p): continue
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''): h.update(chunk)
    good = h.hexdigest() == lfs['sha256'] and os.path.getsize(p) == lfs['size']
    ok &= good; print(('OK  ' if good else 'BAD ') + name, os.path.getsize(p), flush=True)
print('SHA256 ALL OK' if ok else 'SHA256 MISMATCH')
PY
echo "DL DONE $(date +%T)"
