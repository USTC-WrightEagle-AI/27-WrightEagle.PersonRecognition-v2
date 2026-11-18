#!/usr/bin/env bash
# 一键人脸录入：提示输入姓名，然后调用 remember_person.py（同名自动覆盖）。
# 用法: ./scripts/enroll_face.sh [remember_person.py 的额外参数...]
# 例如: ./scripts/enroll_face.sh --no-depth-filter
#       ./scripts/enroll_face.sh --head-backend sdk   (需先停止占用头部 D455 的 ROS 相机)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="${FACE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${FACE_PYTHON:-/home/rpp/miniconda3/envs/robot-face/bin/python}"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS 2 环境不存在: ${ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] 人脸 Python 不存在: ${PYTHON}" >&2
    exit 1
fi

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

NAME=""
while [[ -z "${NAME}" ]]; do
    read -r -p "请输入要录入的姓名: " INPUT
    NAME="$(trim "${INPUT}")"
    if [[ -z "${NAME}" ]]; then
        echo "[WARN] 姓名不能为空，请重新输入。" >&2
    fi
done
echo "[INFO] 即将录入 '${NAME}'（同名数据将被覆盖），采集时请正对头部相机并缓慢转头，按 q 取消。"

# 判断头部相机后端：默认 ros；显式传入 --head-backend sdk 时跳过话题检查。
BACKEND="ros"
PREV=""
for ARG in "$@"; do
    if [[ "${PREV}" == "--head-backend" ]]; then
        BACKEND="${ARG}"
    fi
    case "${ARG}" in
        --head-backend=*) BACKEND="${ARG#*=}" ;;
    esac
    PREV="${ARG}"
done

# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

if [[ "${BACKEND}" == "ros" ]]; then
    for TOPIC in \
        /head/camera/color/image_raw \
        /head/camera/aligned_depth_to_color/image_raw \
        /head/camera/color/camera_info; do
        if ! ros2 topic info "${TOPIC}" 2>/dev/null | \
                grep 'Publisher count: [1-9]' >/dev/null; then
            echo "[ERROR] 头部相机话题没有发布者: ${TOPIC}" >&2
            echo "        请先启动机器人 camera bringup；" >&2
            echo "        或停止 ROS 相机后改用 SDK 采集: $0 --head-backend sdk" >&2
            exit 1
        fi
    done
else
    echo "[INFO] 使用 SDK 直连采集，请确认没有 ROS 节点占用头部 D455。"
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" "${ROOT_DIR}/remember_person.py" "${NAME}" --replace "$@"