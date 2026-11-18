#!/usr/bin/env bash
# 一键综合检测：人脸身份 + 姿态分类 + 三维位置（pose_classifier_3d_ros）。
# 用法: ./scripts/start_detection.sh [start_3d.sh 的额外参数...]
# 例如: ./scripts/start_detection.sh --no-image
#       ./scripts/start_detection.sh --frame-skip 1
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${ROOT_DIR}/data/faces.npz" ]]; then
    echo "[WARN] 人脸库不存在: ${ROOT_DIR}/data/faces.npz" >&2
    echo "       检测仍可运行，但所有人员姓名将输出 Unknown。" >&2
    echo "       可先运行 ./scripts/enroll_face.sh 录入人脸。" >&2
fi

echo "[INFO] 启动综合检测（环境、依赖与头部相机话题由 start_3d.sh 检查）。"
echo "[INFO] 结果话题: /pose_classifier_3d/results 和 /person_identity_locations"
exec bash "${ROOT_DIR}/posture_recognition/scripts/start_3d.sh" "$@"