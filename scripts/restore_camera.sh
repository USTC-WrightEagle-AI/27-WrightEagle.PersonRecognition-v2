#!/usr/bin/env bash
# 异常断电/kill -9 后，恢复上次自动切换前保存的参数。
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${FACE_ROS_SETUP:-${POSE_ROS_SETUP:-/opt/ros/humble/setup.bash}}"
source "${SCRIPT_DIR}/ros_environment.sh"
exec /usr/bin/python3 "${SCRIPT_DIR}/camera_session.py" --restore
