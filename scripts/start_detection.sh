#!/usr/bin/env bash
# 一键综合检测：自动切换相机并打开实时画面，退出后关闭画面并恢复相机。
# 用法: ./scripts/start_detection.sh [start_3d.sh 的额外参数...]
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="${POSE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${POSE_PYTHON:-/home/rpp/miniconda3/envs/foundationpose/bin/python}"
for ARG in "$@"; do
    if [[ "${ARG}" == "--help" || "${ARG}" == "-h" ]]; then
        source "${SCRIPT_DIR}/ros_environment.sh"
        export PYTHONPATH="${ROOT_DIR}/posture_recognition/src${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${PYTHON}" "${ROOT_DIR}/posture_recognition/src/pose_classifier_3d_ros.py" --help
    fi
done
if [[ ! -f "${ROS_SETUP}" || ! -x "${PYTHON}" ]]; then
    echo "[ERROR] ROS 环境或姿态 Python 不存在: ${ROS_SETUP}, ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/data/faces.npz" ]]; then
    echo "[WARN] 默认人脸库不存在；使用默认配置时姓名将输出 Unknown。" >&2
fi
source "${SCRIPT_DIR}/ros_environment.sh"
echo "[INFO] 结果话题: /pose_classifier_3d/results 和 /person_identity_locations"
exec /usr/bin/python3 "${SCRIPT_DIR}/camera_session.py" --mode detection -- \
    bash "${ROOT_DIR}/posture_recognition/scripts/start_3d.sh" "$@"
