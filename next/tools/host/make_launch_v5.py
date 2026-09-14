# launch.json.v5 = v4b + --spec-draft-backend-sampling + tracer loaded from bin/ (lib-new22 ships libnext-trace.so)
import json
P = '/home/douya/qwen-3.8-next-opt'
j = json.load(open(f'{P}/launch.json.v4b'))
cmd = j['command']
if '--spec-draft-backend-sampling' not in cmd:
    i = cmd.index('--spec-draft-n-max'); cmd.insert(i, '--spec-draft-backend-sampling')
j['environment']['LD_PRELOAD'] = f'{P}/bin/libnext-trace.so'
json.dump(j, open(f'{P}/launch.json.v5', 'w'), indent=2, ensure_ascii=False)
print('launch.json.v5:', 'draft backend sampling', '--spec-draft-backend-sampling' in cmd, '| LD_PRELOAD', j['environment']['LD_PRELOAD'], '| nosort', j['environment'].get('NEXT_TOPK_NOSORT'), '| async', j['environment'].get('NEXT_SCHED_ASYNC_INPUTS'))
