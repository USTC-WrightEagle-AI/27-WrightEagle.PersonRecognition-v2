"""Run with system Python after sourcing ROS. Uses an isolated ROS domain."""
import os
os.environ['ROS_DOMAIN_ID'] = '174'
os.environ['ROS_LOCALHOST_ONLY'] = '1'

from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = {'serial_no': '260722304986', 'rgb_camera.color_profile': '640,480,15',
           'depth_module.depth_profile': '640,480,15', 'enable_color': True,
           'enable_depth': True, 'align_depth.enable': False}


def main():
    rclpy.init(args=[])
    node = rclpy.create_node('camera', namespace='/head')
    for key, value in DEFAULT.items():
        node.declare_parameter(key, value)
    publishers = {
        'rgb': node.create_publisher(Image, '/head/camera/color/image_raw', qos_profile_sensor_data),
        'info': node.create_publisher(CameraInfo, '/head/camera/color/camera_info', qos_profile_sensor_data),
        'depth': node.create_publisher(Image, '/head/camera/aligned_depth_to_color/image_raw', qos_profile_sensor_data),
        'raw': node.create_publisher(Image, '/head/camera/depth/image_rect_raw', qos_profile_sensor_data),
    }
    observed = set()
    def publish():
        values = {key: node.get_parameter(key).value for key in DEFAULT}
        color = tuple(map(int, values['rgb_camera.color_profile'].split(',')))
        depth = tuple(map(int, values['depth_module.depth_profile'].split(',')))
        observed.add((color, depth, values['align_depth.enable']))
        stamp = node.get_clock().now().to_msg()
        for key, publisher in publishers.items():
            if key in ('rgb', 'info') and not values['enable_color']:
                continue
            if key in ('depth', 'raw') and not values['enable_depth']:
                continue
            if key == 'depth' and not values['align_depth.enable']:
                continue
            msg = CameraInfo() if key == 'info' else Image()
            msg.width, msg.height = (depth if key == 'raw' else color)[:2]
            msg.header.stamp = stamp
            publisher.publish(msg)
    timer = node.create_timer(.07, publish)
    try:
        with tempfile.TemporaryDirectory(prefix='camera-session-ros-') as tmp:
            signals = {'interrupt': signal.SIGINT, 'terminate': signal.SIGTERM, 'hangup': signal.SIGHUP}
            for outcome in ('success', 'failure', *signals):
                marker = Path(tmp) / outcome
                app = (f'from pathlib import Path; import time; Path({str(marker)!r}).touch(); '
                       + ('time.sleep(120)' if outcome in signals else f'raise SystemExit({7 if outcome == "failure" else 0})'))
                process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/camera_session.py'),
                                            '--mode', 'detection', '--switch-method', 'parameters',
                                            '--', sys.executable, '-c', app],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                sent_signal = False
                try:
                    deadline = time.monotonic() + 40
                    while process.poll() is None and time.monotonic() < deadline:
                        rclpy.spin_once(node, timeout_sec=.05)
                        if outcome in signals and marker.exists() and not sent_signal:
                            process.send_signal(signals[outcome])
                            sent_signal = True
                    stdout, stderr = process.communicate(timeout=2)
                    expected = {'success': 0, 'failure': 7}.get(outcome, 130)
                    assert process.returncode == expected, (process.returncode, stderr.decode())
                    assert marker.exists(), stderr.decode()
                    actual = {key: node.get_parameter(key).value for key in DEFAULT}
                    assert actual == DEFAULT, actual
                    assert ((1280, 720, 30), (1280, 720, 30), True) in observed
                    assert '已恢复进入脚本前的相机配置' in stderr.decode(), stderr.decode()
                    print(f'PASS: ROS switch + new frames + application {outcome} + restore original frames', flush=True)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
    finally:
        node.destroy_timer(timer)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
