import os,sys,time,json,urllib.request,pathlib
pid=int(sys.argv[1]);proc=pathlib.Path(f'/proc/{pid}')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(2400):
 if not proc.exists():break
 try:
  with opener.open('http://127.0.0.1:8093/health',timeout=2) as response:ok=json.load(response).get('status')=='ok'
  if ok and b'/qwen-3.8-next-opt/bin/llama-server' in (proc/'cmdline').read_bytes():
   os.sched_setaffinity(pid,{0});print(f'Main thread {pid} pinned to CPU 0',flush=True);break
 except Exception:pass
 time.sleep(2)
