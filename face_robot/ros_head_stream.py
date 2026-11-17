#!/usr/bin/env python3
"""System-Python ROS reader. Emit length-prefixed NumPy packets to stdout."""
from __future__ import annotations
import argparse
import io
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_robot.head_camera import load_head_configuration, validate_camera_parameters
from face_robot.frame_sync import FrameSynchronizer


def image_array(msg):
    enc = msg.encoding.lower()
    if enc in ('bgr8','rgb8'):
        a = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        return (a[:,:,::-1] if enc == 'rgb8' else a).copy()
    if enc not in ('16uc1','32fc1'):
        raise ValueError('Unsupported head image encoding: '+msg.encoding)
    dtype = np.dtype(('>' if msg.is_bigendian else '<')+('u2' if enc == '16uc1' else 'f4'))
    a = np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.step,dtype.itemsize)).astype(np.float32)
    return a*.001 if enc == '16uc1' else a


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration-dir',required=True)
    p.add_argument('--with-depth',action='store_true')
    args=p.parse_args()
    session, intr=load_head_configuration(args.calibration_dir)
    rclpy.init(args=[])
    node=rclpy.create_node('face_head_camera_reader')
    frames=FrameSynchronizer(args.with_depth)
    info_ok=False
    serial_ok=False
    info_count=0
    error=None
    last_output=time.monotonic()

    def info(msg):
        nonlocal info_ok, info_count, error
        info_count += 1
        try:
            validate_camera_parameters(msg.width,msg.height,[msg.k[0],msg.k[4],msg.k[2],msg.k[5]],msg.d,intr)
            info_ok=True
        except ValueError as exc:error=exc

    def publish(color, depth=None):
        nonlocal last_output
        frame=image_array(color)
        if frame.shape != (session['height'],session['width'],3):
            raise ValueError('ROS head color resolution differs from calibration')
        payload={'color':frame}
        if depth is not None:
            array=image_array(depth)
            if array.shape != frame.shape[:2]:raise ValueError('Aligned depth/color size mismatch')
            payload['depth']=array
        data=io.BytesIO()
        np.savez(data,**payload)
        b=data.getvalue()
        sys.stdout.buffer.write(struct.pack('!I',len(b))+b)
        sys.stdout.buffer.flush()
        last_output=time.monotonic()

    def try_pair():
        if not info_ok or not serial_ok:return
        pair=frames.pop()
        if pair is not None:publish(*pair)

    def on_color(msg):
        frames.add('rgb',msg)

    def on_depth(msg):
        frames.add('depth',msg)

    subs=[node.create_subscription(CameraInfo,'/head/camera/color/camera_info',info,qos_profile_sensor_data),
          node.create_subscription(Image,'/head/camera/color/image_raw',on_color,qos_profile_sensor_data)]
    if args.with_depth:
        subs.append(node.create_subscription(Image,'/head/camera/aligned_depth_to_color/image_raw',on_depth,qos_profile_sensor_data))
    client=node.create_client(GetParameters,'/head/camera/get_parameters')
    future=None
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.05)
            if error:raise error
            if future is None and client.service_is_ready():
                cameras=[(name,ns) for name,ns in node.get_node_names_and_namespaces()
                         if name=='camera' and ns=='/head']
                if len(cameras)>1:
                    raise ValueError('Multiple /head/camera nodes detected; stop the duplicate camera launcher before enrollment')
                req=GetParameters.Request();req.names=['serial_no','rgb_camera.color_profile']
                if args.with_depth:req.names.append('depth_module.depth_profile')
                future=client.call_async(req)
            if future is not None and future.done() and not serial_ok:
                values=future.result().values
                if len(values)!=len(req.names):
                    raise ValueError(f'Incomplete camera parameter reply: expected {req.names}')
                if values[0].string_value.lstrip('_') != session['serial']:
                    raise ValueError(f"ROS head camera serial_no differs from calibration: expected {session['serial']!r}, got {values[0].string_value!r}; check for duplicate /head/camera nodes")
                expected_profile=f"{session['width']},{session['height']},{session['fps']}"
                for name,value in zip(req.names[1:],values[1:]):
                    if value.string_value.replace(' ','') != expected_profile:
                        raise ValueError(f'ROS head stream profile differs from calibration: {name} expected {expected_profile!r}, got {value.string_value!r}')
                serial_ok=True
            try_pair()
            if time.monotonic()-last_output > 8:
                raise RuntimeError(
                    f'No usable head ROS frames for 8s; {frames.diagnostics()}; '
                    f'camera_info: received={info_count}, valid={info_ok}; '
                    f'camera_parameters_verified={serial_ok}. '
                    'Check camera topics and stream timestamps. For single-person RGB enrollment, '
                    'use --no-depth-filter; use --head-backend sdk only after stopping the ROS head camera.')
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        print(str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
