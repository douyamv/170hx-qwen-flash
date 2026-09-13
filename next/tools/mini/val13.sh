S=/home/douya/src/llama.cpp-flashnext-20260913
T=/home/douya/tests/mini; cd $T
systemd-run --scope -p MemoryMax=2500M -p MemorySwapMax=0 --user bash -c "cd $S && taskset -c 20-39 cmake --build build-sm80 --target llama-common -j1 2>&1 | grep -E 'error' -A3 | head; echo build done" 2>&1 | grep -v "^Running"
cp -a $S/build-sm80/bin/libllama-common.so* $T/lib-new13/ && ls -la --time-style=+%H:%M $T/lib-new13/libllama-common.so.0.3.0 | cut -c30-
G3=GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3
echo 4096 > opt/qsa_min_kv; echo 4 > opt/spec_n_max; rm -f opt/hc_fuse_off opt/spec_adaptive_off
bench() { # tag mode
python3 - "$1" "$2" << 'PY'
import json, time, urllib.request, glob, subprocess, sys
tag, mode = sys.argv[1], sys.argv[2]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(o, path='/completion'):
    r = urllib.request.Request('http://127.0.0.1:8094' + path, data=json.dumps(o).encode(), headers={'Content-Type': 'application/json'})
    try: return json.loads(op.open(r, timeout=3600).read())
    except urllib.error.HTTPError as e: return {'err': e.code}
def tok(s): return post({'content': s, 'add_special': False}, '/tokenize')['tokens']
BAN = [[t[0], False] for t in (tok(s) for s in ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>']) if len(t) == 1]
text = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob('/home/douya/src/llama.cpp-flashnext-20260913/src/*.cpp')))
alltoks = tok(text)
def last(): return subprocess.run(['bash', '-c', f'grep -a "eval time =" /home/douya/tests/mini/logs/{tag}.log | grep -v "prompt eval" | tail -1 | cut -c60-150'], capture_output=True, text=True).stdout.strip()
for n in (2048, 16384):
    toks = alltoks[:n]; post(dict(prompt=toks, n_predict=1, cache_prompt=True))
    for rep in range(3):
        post(dict(prompt=toks, n_predict=128, cache_prompt=True, ignore_eos=True, temperature=0, logit_bias=BAN))
        print(f'{tag} {mode} ctx={n} | {last()}', flush=True)
PY
}
echo "########## A. lib-new13 (q8a + fusion + conv + adaptive) ##########"
MINI_DEVS=$G3 LIBDIR=lib-new13 bash mini_run3.sh v13 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' || { grep -a " E \|abort\|GGML_ASSERT" logs/v13.log | head -5 | cut -c1-200; exit 1; }
nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader | grep 81:00
for mode in fused unfused fused; do
  if [ $mode = unfused ]; then touch opt/hc_fuse_off; else rm -f opt/hc_fuse_off; fi; sleep 1.3
  python3 stream_md5.py v13_${mode}_2k 2048 64 | tail -1 | cut -c1-110
  python3 stream_md5.py v13_${mode}_16k 16384 64 | tail -1 | cut -c1-110
  bench v13 $mode
done
rm -f opt/hc_fuse_off
python3 - << 'PY'
import json
for n in ('2k', '16k'):
    a = json.load(open(f'/home/douya/tests/mini/stream-v13_fused_{n}.json')); b = json.load(open(f'/home/douya/tests/mini/stream-v13_unfused_{n}.json'))
    print(n, 'fused vs unfused identical prefix:', next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))), 'of', min(len(a), len(b)))
PY
grep -a "ERR\|abort\|GGML_ASSERT\|could not allocate" logs/v13.log | head -3
echo "########## B. lib-new13 with NEXT_Q8A=0 (q8a off) ##########"
NEXT_Q8A=0 MINI_DEVS=$G3 LIBDIR=lib-new13 bash mini_run3.sh v13q0 --ctx-size 20480 --batch-size 512 --ubatch-size 512 --spec-draft-device CUDA0 --override-tensor '^token_embd\.weight$=CUDA0' || { grep -a " E \|abort" logs/v13q0.log | head -3; }
python3 stream_md5.py v13q0_2k 2048 64 | tail -1 | cut -c1-110
bench v13q0 q8a_off
python3 - << 'PY'
import json
a = json.load(open('/home/douya/tests/mini/stream-v13_fused_2k.json')); b = json.load(open('/home/douya/tests/mini/stream-v13q0_2k.json'))
print('2k q8a on vs off identical prefix:', next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))), 'of', min(len(a), len(b)), '(numerics may differ slightly)')
PY
pkill -f "llama-server.*--port 8094"; echo "VAL13 DONE $(date +%T)"
