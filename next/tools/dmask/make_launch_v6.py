# launch.json.v6 = v5 + NEXT_DEVICE_MASK=1 (GPU-generated KQ masks)
import json
P = '/home/douya/qwen-3.8-next-opt'
j = json.load(open(f'{P}/launch.json.v5'))
j['environment']['NEXT_DEVICE_MASK'] = '1'
json.dump(j, open(f'{P}/launch.json.v6', 'w'), indent=2, ensure_ascii=False)
print('launch.json.v6:', {k: v for k, v in j['environment'].items() if k.startswith('NEXT_')}, '| draft bs', '--spec-draft-backend-sampling' in j['command'])
