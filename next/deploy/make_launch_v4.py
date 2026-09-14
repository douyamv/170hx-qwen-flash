# launch.json.v4 = v3 + env for opt-v6 (async input uploads, top-k without the final sort)
import json, shutil
P = '/home/douya/qwen-3.8-next-opt'
j = json.load(open(f'{P}/launch.json.v3'))
env = j.setdefault('environment', {})
env['NEXT_SCHED_ASYNC_INPUTS'] = '1'
env['NEXT_TOPK_NOSORT'] = '1'
json.dump(j, open(f'{P}/launch.json.v4', 'w'), indent=2, ensure_ascii=False)
print('launch.json.v4 written; env keys:', sorted(env))
