"""Restart just the head camera; bind replacement lifetime to its original launch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': stat[19],
                'argv_sha256': hashlib.sha256((path / 'cmdline').read_bytes()).hexdigest()}
    except (OSError, IndexError):
        return None


def alive(row):
    return bool(row and identity(row['pid']) == row)


def arguments(pid):
    return [part.decode() for part in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if part]


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.chmod(0o600)
    temp.replace(path)


def stop(row):
    for sig, timeout in ((signal.SIGINT, 4), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if not alive(row):
            return
        os.kill(row['pid'], sig)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not alive(row):
                return
            time.sleep(.1)
    if alive(row):
        raise RuntimeError('Head camera process did not stop: ' + str(row['pid']))


def find_head():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = arguments(int(path.name))
            if args and Path(args[0]).name == 'realsense2_camera_node' and '__ns:=/head' in args and '__node:=camera' in args:
                found.append(int(path.name))
        except (OSError, UnicodeError):
            continue
    if len(found) > 1:
        raise RuntimeError('Multiple local head camera processes; nothing will be stopped')
    return found[0] if found else None


def snapshot(runtime, values):
    pid = find_head()
    if pid is None:
        raise RuntimeError('Cannot identify a local /head/camera process for controlled restart')
    row = identity(pid)
    proc = Path('/proc') / str(pid)
    args = arguments(pid)
    ppid = int((proc / 'stat').read_text().split(') ', 1)[1].split()[1])
    guardian_state = Path(runtime) / 'managed-camera.json'
    guardian = None
    tracking_state = None
    owner = identity(ppid)
    if guardian_state.exists():
        record = json.loads(guardian_state.read_text())
        if record.get('camera') == row and alive(record.get('guardian')):
            guardian = record['guardian']
            owner = record['owner']
            tracking_state = record.get('tracking_state')
    if guardian is None:
        parent_args = arguments(ppid)
        if 'cognav_head_guardian_v2' in parent_args:
            marker = parent_args.index('cognav_head_guardian_v2')
            owner = json.loads(parent_args[marker + 1])
            # The previous lifecycle supervisor uses this same process identity.
            owner = {key: owner[key] for key in ('pid', 'start_ticks', 'argv_sha256')}
            guardian = identity(ppid)
            tracking_state = parent_args[marker + 3]
        elif not any('ros2' in arg for arg in parent_args) or 'launch' not in parent_args:
            raise RuntimeError('Head camera parent is not a recognized ROS launch/supervisor; nothing stopped')
    if not alive(owner):
        raise RuntimeError('Original robot launch is no longer running')
    # Copy launch YAMLs so restoration does not depend on transient launch files.
    for index, arg in enumerate(args[:-1]):
        if arg == '--params-file':
            copy = Path(runtime) / f'camera-params-{index}.yaml'
            copy.write_bytes(Path(args[index + 1]).read_bytes())
            copy.chmod(0o600)
            args[index + 1] = str(copy)
    env = dict(part.decode().split('=', 1) for part in (proc / 'environ').read_bytes().split(b'\0') if b'=' in part)
    return {'camera': row, 'guardian': guardian, 'owner': owner,
            'command': args, 'cwd': str(proc.joinpath('cwd').resolve()),
            'environment': env, 'before': values, 'tracking_state': tracking_state}


def stop_record(record):
    if record.get('guardian') and alive(record['guardian']):
        stop(record['guardian'])
    stop(record.get('camera'))


def start_guardian(runtime, original, overrides):
    if not alive(original['owner']):
        raise RuntimeError('Original robot launch exited; camera will not be restarted')
    runtime = Path(runtime)
    # Let USB/sensor teardown settle before opening a new pipeline. Immediate
    # repeated opens can produce depth-start/MIPI errors on this D455.
    time.sleep(2)
    command = []
    source = original['command']
    index = 0
    while index < len(source):
        if source[index] == '-p' and index + 1 < len(source) and source[index + 1].split(':=', 1)[0] in overrides:
            index += 2
            continue
        command.append(source[index])
        index += 1
    # Append parameter overrides after the original launch files/options.
    for key, value in overrides.items():
        encoded = json.dumps(value) if isinstance(value, str) else str(value).lower()
        command.extend(['-p', f'{key}:={encoded}'])
    spec = runtime / 'guardian-spec.json'
    write_json(spec, {'owner': original['owner'], 'command': command, 'cwd': original['cwd'],
                     'environment': original['environment'], 'tracking_state': original.get('tracking_state')})
    log = (runtime / 'camera-driver.log').open('ab')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--guardian', str(runtime)],
                             start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    log.close()
    state = runtime / 'managed-camera.json'
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if state.exists():
            record = json.loads(state.read_text())
            if record.get('guardian') == identity(child.pid) and alive(record.get('camera')):
                if original.get('tracking_state'):
                    # Keep the previous low-resolution tool's ownership record
                    # valid after handing its camera back to a new supervisor.
                    write_json(original['tracking_state'], {
                        'schema': 'managed_head_camera_v2', 'status': 'RUNNING',
                        'camera': record['camera'], 'guardian': record['guardian'],
                        'launch_owner': record['owner'], 'serial': str(overrides['serial_no']).lstrip('_'),
                        'namespace': '/head', 'profile': [int(v) for v in overrides['rgb_camera.color_profile'].replace(' ', '').split(',')],
                        'managed': True})
                return record
        if child.poll() is not None:
            detail = (runtime / 'camera-driver.log').read_text(errors='replace')[-1500:]
            raise RuntimeError('Camera supervisor exited; see ' + str(runtime / 'camera-driver.log') + '\n' + detail)
        time.sleep(.1)
    stop(identity(child.pid))
    raise RuntimeError('Camera supervisor startup timed out')


def guardian(runtime):
    import ctypes
    runtime = Path(runtime)
    spec = json.loads((runtime / 'guardian-spec.json').read_text())
    owner = spec['owner']
    if not alive(owner):
        return
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    parent = os.getpid()
    def parent_death_signal():
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(1)
    child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=spec['environment'],
                             stdin=subprocess.DEVNULL, preexec_fn=parent_death_signal)
    camera = identity(child.pid)
    if camera is None:
        raise RuntimeError('Camera exited immediately')
    write_json(runtime / 'managed-camera.json', {'owner': owner, 'guardian': identity(os.getpid()),
               'camera': camera, 'tracking_state': spec.get('tracking_state')})
    try:
        while child.poll() is None and not stopping and alive(owner):
            time.sleep(.1)
    finally:
        stop(camera)
        child.wait(timeout=3)


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--guardian':
    guardian(sys.argv[2])
