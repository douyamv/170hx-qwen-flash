#!/usr/bin/env bash
# 前台启动测试版 llama-server(需先停掉线上服务以释放显存)。
cd /home/douya
exec python3 - << 'PY'
import json, os
c = json.load(open('/home/douya/qwen-3.8-next-opt/launch.json'))
os.environ.update(c['environment'])
os.execvpe(c['command'][0], c['command'], os.environ)
PY
