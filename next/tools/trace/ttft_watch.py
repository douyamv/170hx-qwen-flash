import os, time, subprocess, re
P = '/home/douya/qwen-3.8-next-opt'; LOG = P + '/logs/server.log'; FLAG = P + '/profile.flag'; PROF = P + '/logs/profile.csv'
def tail_size(): return os.path.getsize(LOG)
pos = tail_size(); print('watching from', pos, flush=True)
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    time.sleep(0.2)
    sz = tail_size()
    if sz <= pos: continue
    with open(LOG, 'rb') as f:
        f.seek(pos); chunk = f.read(sz - pos).decode(errors='ignore'); pos = sz
    if re.search(r'new prompt|launch_slot_|processing task', chunk):
        off = os.path.getsize(PROF) if os.path.exists(PROF) else 0
        open(FLAG, 'w').close(); t0 = time.time(); print('TRIGGER at', time.strftime('%H:%M:%S'), chunk.strip().splitlines()[-1][:120], flush=True)
        time.sleep(16); 
        if os.path.exists(FLAG): os.remove(FLAG)
        time.sleep(1.5)
        out = '/home/douya/tests/trace-ttft-%s.csv' % time.strftime('%H%M%S')
        with open(PROF, 'rb') as f, open(out, 'wb') as g: f.seek(off); g.write(f.read())
        print('saved', out, os.path.getsize(out), flush=True); break
print('watcher done', flush=True)
