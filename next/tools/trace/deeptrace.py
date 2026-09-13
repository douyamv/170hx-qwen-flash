#!/usr/bin/env python3
import json, os, time, subprocess, urllib.request, re, glob, threading
URL = 'http://127.0.0.1:8093'
BASE = '/home/douya/qwen-3.8-next'
FLAG = BASE + '/profile.flag'
PROFILE = BASE + '/logs/profile.csv'
SLOG = BASE + '/logs/server.log'
OUT = '/home/douya/tests/deeptrace-20260913'
SRC = '/home/douya/src/llama.cpp-flashnext-20260913'
os.makedirs(OUT, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
events = open(OUT + '/events.log', 'a')

def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True); events.write(line + '\n'); events.flush()

def post(path, obj, timeout=7200):
    req = urllib.request.Request(URL + path, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())

def psize():
    return os.path.getsize(PROFILE) if os.path.exists(PROFILE) else 0

def trace_on(tag):
    if os.path.exists(FLAG):
        os.remove(FLAG); time.sleep(0.3)
    log(f'TRACE_ON {tag} profile_offset={psize()}')
    open(FLAG, 'w').close()

def trace_off(tag):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    time.sleep(0.6)
    log(f'TRACE_OFF {tag} profile_offset={psize()}')

def perf(tag, secs):
    return subprocess.Popen(['sudo', '-n', 'perf', 'record', '-F', '499', '-g', '-p', '8077',
                             '-o', f'{OUT}/perf-{tag}.data', '--', 'sleep', str(secs)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def brief(r):
    t = dict(r.get('timings') or {})
    return json.dumps(t)

try:
    docs = sorted(glob.glob(SRC + '/docs/*.md'))
    textA = ''.join(open(p, errors='ignore').read() for p in docs[:8])
    tokA = post('/tokenize', {'content': textA})['tokens'][:2000]
    srcs = sorted(glob.glob(SRC + '/src/*.cpp'))
    textB = ''.join(open(p, errors='ignore').read() for p in srcs)
    tokB_all = post('/tokenize', {'content': textB})['tokens']
    tokB = tokB_all[:70000]
    textS = ''.join(open(p, errors='ignore').read() for p in docs[8:12])
    tokS = post('/tokenize', {'content': textS})['tokens'][:128]
    log(f'tokens A={len(tokA)} B={len(tokB)} (avail {len(tokB_all)}) S={len(tokS)}')

    # W0: short prompt prefill + decode
    trace_on('W0_short')
    p = perf('W0_short', 12)
    t0 = time.time()
    r = post('/completion', {'prompt': tokA, 'n_predict': 400, 'cache_prompt': True})
    log(f'A_done wall={time.time()-t0:.1f}s timings={brief(r)}')
    p.wait(); time.sleep(max(0, 15.5 - (time.time() - t0))); trace_off('W0_short')
    time.sleep(2)

    # B: 70K prefill, W1 at start, W2 at >=62000 processed tokens
    slog_off = os.path.getsize(SLOG)
    res = {}
    def reqB():
        t = time.time()
        res['r'] = post('/completion', {'prompt': tokB, 'n_predict': 1, 'cache_prompt': True})
        res['t'] = time.time() - t
    trace_on('W1_shallow_prefill')
    p = perf('W1_shallow_prefill', 12)
    th = threading.Thread(target=reqB); th.start()
    time.sleep(16); p.wait(); trace_off('W1_shallow_prefill')
    w2 = False
    while th.is_alive():
        with open(SLOG, 'rb') as f:
            f.seek(slog_off); chunk = f.read().decode(errors='ignore')
        ns = [int(x) for x in re.findall(r'prompt processing, n_tokens =\s*(\d+)', chunk)]
        cur = ns[-1] if ns else 0
        if not w2 and cur >= 62000:
            log(f'deep window trigger at n_tokens={cur}')
            trace_on('W2_deep_prefill'); p = perf('W2_deep_prefill', 12)
            time.sleep(16); p.wait(); trace_off('W2_deep_prefill'); w2 = True
        time.sleep(1)
    th.join()
    log(f'B_done wall={res["t"]:.1f}s timings={brief(res["r"])}')
    time.sleep(2)

    # C: B + suffix, decode 512 tokens at ~70K context
    trace_on('W3_decode70k')
    p = perf('W3_decode70k', 12)
    t0 = time.time()
    r = post('/completion', {'prompt': tokB + tokS, 'n_predict': 512, 'cache_prompt': True})
    log(f'C_done wall={time.time()-t0:.1f}s timings={brief(r)}')
    p.wait(); time.sleep(max(0, 15.5 - (time.time() - t0))); trace_off('W3_decode70k')
    log('ALL_DONE')
except Exception as e:
    log(f'ERROR {type(e).__name__}: {e}')
    if os.path.exists(FLAG): os.remove(FLAG)
