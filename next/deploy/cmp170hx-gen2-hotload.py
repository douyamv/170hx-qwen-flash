#!/usr/bin/env python3
"""Reload the installed CMP driver without rebooting or writing GPU registers."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

UNITS = ['vast_metrics.service', 'vastai.service']
PARAMS = ['NVreg_EnableGpuFirmware=18', 'NVreg_EnableGpuFirmwareLogs=2',
          'NVreg_RegistryDwords=RmForceEnableGen2=1;RMPcieLinkSpeed=0x1']

def run(args, timeout=45, check=True):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f'{args[0]}: {p.stdout.strip()} {p.stderr.strip()}')
    return p

def gpus():
    r = run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,memory.total',
             '--format=csv,noheader,nounits'], timeout=45)
    out = []
    for line in r.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 5 or not fields[0].startswith('GPU-'):
            raise RuntimeError('Unexpected NVIDIA inventory: '+line)
        out.append(dict(uuid=fields[0], pci=fields[1], gen=int(fields[2]), width=int(fields[3]), memory_mib=int(fields[4])))
    if not out:
        raise RuntimeError('No GPUs returned by NVIDIA')
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only report live link status')
    parser.add_argument('--reload', action='store_true', help='Reload even if all cards already use Gen2')
    args = parser.parse_args()
    before = gpus()
    if args.check:
        print(json.dumps(before, ensure_ascii=False, indent=2))
        return
    if all(x['gen'] >= 2 for x in before) and not args.reload:
        print('GEN2_ALREADY_READY '+json.dumps(before), flush=True)
        return
    if os.geteuid() != 0:
        raise RuntimeError('Run this helper using sudo')
    lock = open('/run/lock/cmp170hx-gen2-hotload.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = pathlib.Path('/var/log/cmp170hx-hotload')
    root.mkdir(exist_ok=True)
    path = root/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'.json')
    boot = pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    module = pathlib.Path(run(['modinfo','-n','nvidia']).stdout.strip())
    if '/updates/cmpunlocker/' not in str(module):
        raise RuntimeError('The selected module is not the cmpunlocker installation')
    version = run(['modinfo','-F','version','nvidia']).stdout.strip()
    if version != '610.57.04':
        raise RuntimeError('This helper was validated with driver 610.57.04; inspect a changed driver before reloading')
    for row in before:
        bdf = '0000:'+row['pci'].split(':',1)[1]
        if (pathlib.Path('/sys/bus/pci/devices')/bdf/'device').read_text().strip() != '0x2082':
            raise RuntimeError('Unexpected GPU SKU; nothing unloaded')
    data = dict(boot_id=boot, before=before, module=str(module), version=version,
                module_sha256=hashlib.file_digest(module.open('rb'),'sha256').hexdigest(), stopped_units=[])
    error = None
    try:
        jobs = run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout
        if any(x.strip().isdigit() for x in jobs.splitlines()):
            raise RuntimeError('Compute processes are active; stop inference before hot loading')
        for unit in UNITS:
            if run(['systemctl','is-active','--quiet',unit],check=False).returncode == 0:
                data['stopped_units'].append(unit)
                run(['systemctl','stop',unit],timeout=60)
        devs = [str(x) for x in pathlib.Path('/dev').glob('nvidia*') if x.is_char_device()]
        if devs and run(['fuser',*devs],check=False).returncode == 0:
            raise RuntimeError('GPU file handles remain open; driver has not been unloaded')
        data['kernel_since'] = datetime.datetime.now().isoformat(timespec='seconds')
        for name in ['nvidia_drm','nvidia_modeset','nvidia_peermem','nvidia_uvm','nvidia']:
            if (pathlib.Path('/sys/module')/name).exists():
                run(['modprobe','-r',name],timeout=25)
        print('CMP_DRIVER_UNLOADED; system remains running',flush=True)
        run(['modprobe','nvidia',*PARAMS],timeout=45)
        run(['modprobe','nvidia_uvm'],timeout=30)
        data['persistence'] = run(['nvidia-smi','-pm','1'],timeout=45).stdout
        for attempt in range(13):
            after = gpus()
            data['after'] = after
            print('GEN2_CHECK '+json.dumps(after),flush=True)
            if {x['uuid'] for x in after} != {x['uuid'] for x in before}:
                raise RuntimeError('GPU inventory changed during reload')
            if any(x['memory_mib'] != 40960 for x in after):
                raise RuntimeError('GPU capacity changed during reload')
            if all(x['gen'] >= 2 for x in after):
                break
            if attempt < 12:
                time.sleep(5)
        else:
            raise RuntimeError('Gen2 not reached on all cards; no reset or reboot was attempted')
        if pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
            raise RuntimeError('Boot ID changed unexpectedly')
        kernel = run(['journalctl','-k','-b','--since',data['kernel_since'],'--no-pager']).stdout
        data['errors'] = [x for x in kernel.splitlines() if 'NVRM: Xid' in x or 'GPU has fallen off' in x]
        if data['errors']:
            raise RuntimeError('New NVIDIA errors detected; inspect the saved evidence')
        data['success'] = True
        print('GEN2_READY; BOOT_ID_UNCHANGED',flush=True)
    except Exception as exc:
        error = exc
        data['success'] = False
        data['error'] = str(exc)
        # Restore module availability after a partial unload. Never force-remove or reboot.
        for args in [['modprobe','nvidia',*PARAMS],['modprobe','nvidia_uvm']]:
            try:
                run(args,timeout=45,check=False)
            except Exception:
                pass
    finally:
        data['monitor_restore_errors'] = []
        for unit in reversed(data['stopped_units']):
            try:
                run(['systemctl','start',unit],timeout=45)
            except Exception as exc:
                data['monitor_restore_errors'].append(str(exc))
        path.write_text(json.dumps(data,indent=2))
        print('EVIDENCE '+str(path),flush=True)
    if error:
        raise error

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('HOTLOAD_FAILED: '+str(exc),file=sys.stderr)
        sys.exit(1)
