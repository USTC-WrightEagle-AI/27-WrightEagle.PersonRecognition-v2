#!/usr/bin/env python3
"""Temporarily borrow a head ROS camera profile and restore it on exit."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NODE = '/head/camera'
PARAMETERS = ('serial_no', 'rgb_camera.color_profile', 'depth_module.depth_profile',
              'enable_color', 'enable_depth', 'align_depth.enable')


class CameraParametersNotReady(RuntimeError):
    pass


def log(message):
    print('[CAMERA] ' + message, file=sys.stderr, flush=True)


def profile(value):
    parts = str(value).replace(' ', '').replace('x', ',').split(',')
    if len(parts) != 3 or any(not p.isdigit() or int(p) <= 0 for p in parts):
        raise ValueError('Invalid camera profile: ' + repr(value))
    return tuple(map(int, parts))


class ProfileLease:
    """Keep a durable snapshot before the first potentially partial write."""
    def __init__(self, io, journal):
        self.io = io
        self.journal = Path(journal)
        self.before = None

    def save(self, values):
        self.before = values
        temporary = self.journal.with_suffix('.tmp')
        temporary.write_text(json.dumps({'node': NODE, 'before': values}, indent=2))
        os.chmod(temporary, 0o600)
        temporary.replace(self.journal)

    def prepare(self, serial, target, with_depth):
        before = self.io.get(PARAMETERS)
        if str(before['serial_no']).lstrip('_') != serial:
            raise RuntimeError(f"Head camera serial differs: expected {serial}, got {before['serial_no']!r}")
        for name in ('enable_color', 'enable_depth', 'align_depth.enable'):
            if type(before[name]) is not bool:
                raise RuntimeError('Missing boolean camera parameter: ' + name)
        for name in ('rgb_camera.color_profile', 'depth_module.depth_profile'):
            profile(before[name])
        desired = {'rgb_camera.color_profile': target, 'enable_color': True}
        if with_depth:
            desired.update({'depth_module.depth_profile': target,
                            'enable_depth': True, 'align_depth.enable': True})
        if all(before[key] == value for key, value in desired.items()):
            log('相机已使用所需配置，无需切换。')
            return
        self.save(before)
        log(f"临时切换到 {target}；退出后恢复 RGB={before['rgb_camera.color_profile']}、"
            f"Depth={before['depth_module.depth_profile']}。")
        self.apply(desired)

    def apply(self, values):
        current = self.io.get(PARAMETERS)
        changed_profiles = [key for key in ('rgb_camera.color_profile', 'depth_module.depth_profile')
                            if key in values and current[key] != values[key]]
        # The installed RealSense driver applies new profiles on re-enable.
        # Set profile values only while that stream is disabled.
        for key in changed_profiles:
            enable = 'enable_color' if key.startswith('rgb_') else 'enable_depth'
            if current[enable]:
                self.io.set({enable: False})
        for key in changed_profiles:
            self.io.set({key: values[key]})
        if 'align_depth.enable' in values and current['align_depth.enable'] != values['align_depth.enable']:
            self.io.set({'align_depth.enable': values['align_depth.enable']})
        for key in ('enable_depth', 'enable_color'):
            if key in values:
                self.io.set({key: values[key]})
        actual = self.io.get(tuple(values))
        if actual != values:
            raise RuntimeError(f'Camera parameter readback mismatch: {actual!r} != {values!r}')

    def restore(self):
        if self.before is None:
            return
        current = self.io.get(('serial_no',))
        if current['serial_no'] != self.before['serial_no']:
            raise RuntimeError('Camera identity changed; refusing to restore another device')
        self.apply({key: value for key, value in self.before.items() if key != 'serial_no'})
        self.io.wait_restored(self.before)
        self.journal.unlink(missing_ok=True)
        self.before = None
        log('已恢复进入脚本前的相机配置。')


class RestartLease(ProfileLease):
    """Use process restart for drivers whose dynamic profile update can hang."""
    def __init__(self, io, journal):
        super().__init__(io, journal)
        self.original = None

    def prepare(self, serial, target, with_depth):
        from camera_process import snapshot, stop_record, start_guardian, write_json
        before = self.io.get(PARAMETERS)
        if str(before['serial_no']).lstrip('_') != serial:
            raise RuntimeError('Head camera serial differs; nothing stopped')
        desired = {'rgb_camera.color_profile': target, 'enable_color': True}
        if with_depth:
            desired.update({'depth_module.depth_profile': target, 'enable_depth': True,
                            'align_depth.enable': True})
        if all(before[key] == value for key, value in desired.items()):
            log('相机已使用所需配置，无需切换。')
            return
        # Snapshot the launch command, environment and live parameter values
        # BEFORE stopping the original head process. No other camera is touched.
        self.original = snapshot(self.journal.parent, before)
        self.before = before
        write_json(self.journal, {'node': NODE, 'before': before, 'restart': self.original})
        log(f"受控重启头部相机至 {target}；结束后恢复原配置。")
        stop_record(self.original)
        self.io.wait_node_gone()
        start_guardian(self.journal.parent, self.original, {**before, **desired})
        if not self.io.discover():
            raise RuntimeError('Restarted head camera parameter service did not appear')
        self.io.wait_parameters()

    def restore(self):
        if self.original is None:
            return
        from camera_process import alive, stop_record, start_guardian
        state = self.journal.parent / 'managed-camera.json'
        if state.exists():
            current = json.loads(state.read_text())
            stop_record(current)
        # If startup was interrupted before a managed state was written, the
        # original process may still be running. Stop only its exact identity.
        stop_record(self.original)
        self.io.wait_node_gone()
        if alive(self.original['owner']):
            start_guardian(self.journal.parent, self.original, self.before)
            if not self.io.discover():
                raise RuntimeError('Restored head camera parameter service did not appear')
            if self.io.wait_parameters() != self.before:
                raise RuntimeError('Restored camera parameter readback differs')
            self.io.wait_restored(self.before)
            log('已恢复进入脚本前的相机配置。')
        else:
            log('原机器人启动程序已退出，已关闭临时相机。')
        self.journal.unlink(missing_ok=True)
        self.before = self.original = None


class RosCamera:
    def __init__(self):
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from rcl_interfaces.srv import GetParameters, SetParameters
        self.rclpy = rclpy
        os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node('person_recognition_camera_session')
        self.get_type, self.set_type = GetParameters, SetParameters
        self.get_client = self.node.create_client(GetParameters, NODE + '/get_parameters')
        self.set_client = self.node.create_client(SetParameters, NODE + '/set_parameters')

    def discover(self):
        start = time.monotonic()
        end = start + 10
        found = False
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            self.assert_unique()
            found = found or any(name == 'camera' and ns == '/head'
                                 for name, ns in self.node.get_node_names_and_namespaces())
            if time.monotonic() - start >= 2 and found and self.get_client.service_is_ready():
                return True
        if found and not self.get_client.wait_for_service(timeout_sec=5):
            raise RuntimeError('已有 /head/camera 节点，但参数服务不可用；不会再启动重复节点。')
        ready = self.get_client.service_is_ready()
        if not ready:
            from camera_process import find_head
            if find_head() is not None:
                raise RuntimeError('头部相机进程已存在，但当前 ROS 环境不可见；请检查 ROS_LOCALHOST_ONLY、ROS_DOMAIN_ID 和 RMW_IMPLEMENTATION，不会重复启动相机。')
        return ready

    def assert_unique(self):
        count = sum(name == 'camera' and ns == '/head'
                    for name, ns in self.node.get_node_names_and_namespaces())
        if count > 1:
            raise RuntimeError('发现多个 /head/camera 节点，请先关闭重复的相机启动程序。')

    def wait_node_gone(self):
        end = time.monotonic() + 20
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=.1)
            found = any(name == 'camera' and ns == '/head'
                        for name, ns in self.node.get_node_names_and_namespaces())
            if not found and not self.get_client.service_is_ready():
                return
        raise RuntimeError('Old head camera still visible in the ROS graph')

    def call(self, client, request):
        self.assert_unique()
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Camera service unavailable: ' + client.srv_name)
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=12)
        if not future.done():
            raise RuntimeError('Camera parameter request timed out: ' + client.srv_name)
        return future.result()

    def get(self, names):
        request = self.get_type.Request()
        request.names = list(names)
        reply = self.call(self.get_client, request).values
        if len(reply) != len(names):
            raise CameraParametersNotReady('Incomplete camera parameter reply')
        values = {}
        for name, value in zip(names, reply):
            if value.type == 1:
                values[name] = value.bool_value
            elif value.type == 4:
                values[name] = value.string_value
            elif value.type == 0:
                raise CameraParametersNotReady('Camera parameter not declared yet: ' + name)
            else:
                raise RuntimeError('Missing or unsupported camera parameter: ' + name)
        return values

    def wait_parameters(self, process=None, timeout=30):
        # RealSense exposes get_parameters before the sensors have finished
        # declaring profile parameters. A ready service is not a ready camera.
        end = time.monotonic() + timeout
        reason = 'parameter service not ready'
        while time.monotonic() < end:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'Head camera driver exited during initialization: {process.returncode}')
            if self.get_client.wait_for_service(timeout_sec=.5):
                try:
                    return self.get(PARAMETERS)
                except CameraParametersNotReady as error:
                    reason = str(error)
            self.rclpy.spin_once(self.node, timeout_sec=.2)
            # spin_once may return immediately when another discovery callback
            # is ready. Still throttle initialization probes to avoid hammering
            # the driver while it is declaring parameters.
            time.sleep(.2)
        raise RuntimeError('等待相机完整参数初始化超时: ' + reason)

    def set(self, values):
        from rclpy.parameter import Parameter
        request = self.set_type.Request()
        request.parameters = [Parameter(key, value=value).to_parameter_msg()
                              for key, value in values.items()]
        results = self.call(self.set_client, request).results
        if len(results) != len(values) or any(not row.successful for row in results):
            raise RuntimeError('Camera rejected parameters: ' + repr([row.reason for row in results]))

    def wait_live(self, serial, target, with_depth):
        size = profile(target)[:2]
        streams = {'rgb': ('/head/camera/color/image_raw', size),
                   'info': ('/head/camera/color/camera_info', size)}
        if with_depth:
            streams['depth'] = ('/head/camera/aligned_depth_to_color/image_raw', size)
        self.wait_streams(streams, require_pair=with_depth)
        values = self.get(('serial_no', 'rgb_camera.color_profile', 'depth_module.depth_profile'))
        if str(values['serial_no']).lstrip('_') != serial:
            raise RuntimeError('Camera identity changed during startup')
        if profile(values['rgb_camera.color_profile']) != profile(target) or (
                with_depth and profile(values['depth_module.depth_profile']) != profile(target)):
            raise RuntimeError('Camera profile changed during startup')
        log('已收到所需分辨率的新图像，开始运行。')

    def wait_restored(self, values):
        streams = {}
        color_size = profile(values['rgb_camera.color_profile'])[:2]
        if values['enable_color']:
            streams['rgb'] = ('/head/camera/color/image_raw', color_size)
        if values['enable_depth']:
            aligned = values['align_depth.enable'] and values['enable_color']
            streams['depth'] = (
                '/head/camera/aligned_depth_to_color/image_raw' if aligned else '/head/camera/depth/image_rect_raw',
                color_size if aligned else profile(values['depth_module.depth_profile'])[:2])
        self.wait_streams(streams)

    def wait_streams(self, streams, require_pair=False):
        from sensor_msgs.msg import Image, CameraInfo
        from rclpy.qos import qos_profile_sensor_data
        if not streams:
            return
        latest = {}
        history = {key: deque(maxlen=16) for key in streams}
        barrier = self.node.get_clock().now().nanoseconds
        def receive(key, msg):
            stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            if (msg.width, msg.height) == streams[key][1] and stamp >= barrier:
                latest[key] = (stamp, time.monotonic())
                history[key].append(latest[key])
        subs = [self.node.create_subscription(CameraInfo if key == 'info' else Image, topic,
                lambda msg, key=key: receive(key, msg), qos_profile_sensor_data)
                for key, (topic, _) in streams.items()]
        try:
            end = time.monotonic() + 20
            while time.monotonic() < end:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                if set(latest) != set(streams):
                    continue
                now = time.monotonic()
                if any(now - arrived > 0.5 for _, arrived in latest.values()):
                    continue
                if require_pair:
                    paired = any(abs(rgb_stamp - depth_stamp) <= 50000000
                                 and now - rgb_arrival <= 0.5 and now - depth_arrival <= 0.5
                                 for rgb_stamp, rgb_arrival in history['rgb']
                                 for depth_stamp, depth_arrival in history['depth'])
                    if not paired:
                        continue
                return
            raise RuntimeError('相机配置后的新图像验证超时，收到的流: ' + ', '.join(latest))
        finally:
            for sub in subs:
                self.node.destroy_subscription(sub)

    def close(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait(timeout=timeout)
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def run_command(command, *, viewer=None):
    # The supervisor catches terminal signals; the child gets its own group.
    child = subprocess.Popen(command, start_new_session=True)
    window = None
    try:
        if viewer:
            try:
                window = subprocess.Popen(viewer, start_new_session=True)
                log('已启动实时检测画面窗口。')
            except OSError as error:
                log('无法打开检测画面窗口: ' + str(error))
        warned = False
        while child.poll() is None:
            if window is not None and window.poll() is not None and not warned:
                if window.returncode:
                    log(f'画面查看器异常退出，状态码 {window.returncode}；请检查上面的 GUI 错误。检测继续运行。')
                else:
                    log('画面窗口已关闭，检测继续运行；终端按 Ctrl+C 结束检测。')
                warned = True
            time.sleep(.1)
        code = child.returncode
        return code if code >= 0 else 128 - code
    finally:
        with ignore_signals():
            try:
                stop_process(window)
            finally:
                stop_process(child)


def detection_viewer(mode, command):
    if mode != 'detection' or '--no-image' in command[2:]:
        return None
    # Only the real detection entry needs a viewer, not diagnostic commands.
    if len(command) < 2 or Path(command[1]).name != 'start_3d.sh':
        return None
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        log('未发现桌面显示环境；检测结果图像仍发布到 /pose_classifier_3d/image。')
        return None
    return ['bash', str(ROOT / 'posture_recognition/scripts/view_image.sh')]


@contextmanager
def camera_lock():
    runtime = Path('/tmp') / f'person-recognition-camera-{os.getuid()}-domain-{os.environ.get("ROS_DOMAIN_ID", "0")}'
    runtime.mkdir(mode=0o700, exist_ok=True)
    if runtime.is_symlink() or runtime.stat().st_uid != os.getuid():
        raise RuntimeError('Camera runtime directory has an unexpected owner')
    os.chmod(runtime, 0o700)
    with (runtime / 'session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个录入/检测任务正在使用头部相机，请先结束该任务。') from None
        yield runtime / 'restore.json'


def start_camera(serial, target, with_depth):
    from ament_index_python.packages import get_package_prefix
    executable = Path(get_package_prefix('realsense2_camera')) / 'lib/realsense2_camera/realsense2_camera_node'
    log('未发现头部相机节点，临时启动头部 D455；结束后关闭。')
    log('相机驱动: ' + str(executable))
    command = [str(executable), '--ros-args', '-r', '__node:=camera', '-r', '__ns:=/head']
    parameters = {'serial_no': '_' + serial, 'camera_name': 'head_camera', 'publish_tf': False,
                  'enable_color': True, 'enable_depth': with_depth,
                  'enable_infra1': False, 'enable_infra2': False,
                  'enable_accel': False, 'enable_gyro': False,
                  'pointcloud.enable': False, 'align_depth.enable': with_depth,
                  'rgb_camera.color_profile': target, 'depth_module.depth_profile': target}
    for key, value in parameters.items():
        command.extend(['-p', f'{key}:={str(value).lower() if isinstance(value, bool) else value}'])
    parent = os.getpid()
    def parent_death_signal():
        # This temporary camera belongs to this session, not the robot bringup.
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(1)
    return subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                            preexec_fn=parent_death_signal)


def supervise(io, journal, serial, target, with_depth, command, *, restore_only=False, lease_factory=ProfileLease, viewer=None):
    lease = lease_factory(io, journal)
    camera = None
    try:
        if restore_only:
            if not journal.exists():
                log('没有待恢复的相机配置。')
                return 0
            record = json.loads(journal.read_text())
            if record['node'] != NODE or set(record['before']) != set(PARAMETERS):
                raise RuntimeError('Invalid camera restore record')
            if 'restart' in record:
                lease = RestartLease(io, journal)
                lease.original = record['restart']
            lease.before = record['before']
            lease.restore()
            return 0
        if journal.exists():
            raise RuntimeError('发现上次未完成的恢复记录，请先运行 scripts/restore_camera.sh。')
        if not io.discover():
            camera = start_camera(serial, target, with_depth)
        io.wait_parameters(process=camera)
        lease.prepare(serial, target, with_depth)
        io.wait_live(serial, target, with_depth)
        return run_command(command, viewer=viewer) if viewer else run_command(command)
    finally:
        # A second Ctrl+C must not interrupt the restoration half way through.
        with ignore_signals():
            try:
                if camera is None and not restore_only:
                    try:
                        lease.restore()
                    except Exception:
                        log('相机恢复未完成，保留了参数快照。原相机在线后请运行 scripts/restore_camera.sh。')
                        raise
            finally:
                if camera is not None:
                    stop_process(camera)
                    # This node did not exist before the session; leave no node.
                    if lease.before is not None:
                        journal.unlink(missing_ok=True)


@contextmanager
def ignore_signals():
    saved = {sig: signal.signal(sig, signal.SIG_IGN)
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def interrupted(sig, _frame):
    raise KeyboardInterrupt


def settings(mode, arguments):
    if mode == 'enroll':
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--camera', default='head')
        parser.add_argument('--head-backend', default='ros')
        parser.add_argument('--calibration-dir', type=Path, default=ROOT / 'calibration/head_d455_hd')
        parser.add_argument('--depth-filter', action=argparse.BooleanOptionalAction, default=True)
        args, _ = parser.parse_known_args(arguments)
        if args.camera != 'head' or args.head_backend != 'ros':
            return None
        session = json.loads((args.calibration_dir / 'session.json').read_text())
        return str(session['serial']), ','.join(str(session[k]) for k in ('width', 'height', 'fps')), args.depth_filter
    import yaml
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, default=Path(os.environ.get('POSE_3D_CONFIG', ROOT / 'posture_recognition/config/pose_3d_config.yaml')))
    args, _ = parser.parse_known_args(arguments)
    camera = yaml.safe_load(args.config.read_text())['camera']
    if camera.get('namespace', 'head/camera').strip('/') != 'head/camera':
        raise ValueError('Automatic camera switching supports only /head/camera')
    return str(camera['serial_no']), ','.join(str(camera[k]) for k in ('width', 'height', 'fps')), True


def absolute_config_arguments(command):
    # start_3d.sh changes cwd; preserve the caller's --config path for both
    # camera preparation and the actual detector.
    result = list(command)
    for index in range(2, len(result)):
        if result[index] == '--config' and index + 1 < len(result):
            result[index + 1] = str(Path(result[index + 1]).resolve())
        elif result[index].startswith('--config='):
            result[index] = '--config=' + str(Path(result[index].split('=', 1)[1]).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('enroll', 'detection'))
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--switch-method', choices=('restart', 'parameters'), default='restart',
                        help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.restore and not command:
        parser.error('a command after -- is required')
    if args.mode == 'detection':
        command = absolute_config_arguments(command)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    options = None if args.restore else settings(args.mode, command[2:])
    viewer = detection_viewer(args.mode, command) if not args.restore else None
    if not args.restore and (options is None or os.environ.get('PERSON_CAMERA_AUTO') == '0'):
        return run_command(command, viewer=viewer) if viewer else run_command(command)
    with camera_lock() as journal:
        io = RosCamera()
        try:
            factory = RestartLease if args.switch_method == 'restart' else ProfileLease
            return supervise(io, journal, *(options or ('', '', False)), command,
                             restore_only=args.restore, lease_factory=factory, viewer=viewer)
        finally:
            io.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('任务已中断。')
        raise SystemExit(130)
    except Exception as error:
        log('ERROR: ' + str(error))
        raise SystemExit(1)
