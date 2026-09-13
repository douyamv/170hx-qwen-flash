import pathlib,json,os,subprocess,sys
p=pathlib.Path(__file__).resolve().parent
c=json.loads((p/'launch.json').read_text())
subprocess.Popen([sys.executable,str(p/'pin-main.py'),str(os.getpid())])
os.environ.update(c['environment'])
os.execvpe(c['command'][0],c['command'],os.environ)
