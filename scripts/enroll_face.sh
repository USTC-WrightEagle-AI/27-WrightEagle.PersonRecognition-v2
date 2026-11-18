#!/usr/bin/env bash
# 一键录入：自动临时切换头部相机配置，结束后恢复。
# 用法: ./scripts/enroll_face.sh [remember_person.py 的额外参数...]
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="${FACE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${FACE_PYTHON:-/home/rpp/miniconda3/envs/robot-face/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] 人脸 Python 不存在: ${PYTHON}" >&2
    exit 1
fi
for ARG in "$@"; do
    if [[ "${ARG}" == "--help" || "${ARG}" == "-h" ]]; then
        exec "${PYTHON}" "${ROOT_DIR}/remember_person.py" --help
    fi
done
if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS 2 环境不存在: ${ROS_SETUP}" >&2
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
    if ! read -r -p "请输入要录入的姓名: " INPUT; then
        echo "[ERROR] 未输入姓名，取消录入。" >&2
        exit 1
    fi
    NAME="$(trim "${INPUT}")"
    if [[ -z "${NAME}" ]]; then
        echo "[WARN] 姓名不能为空，请重新输入。" >&2
    fi
done
echo "[INFO] 即将录入 '${NAME}'（同名数据将被覆盖），按 q 取消。"
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/ros_environment.sh"
set -u
cd "${ROOT_DIR}"
exec /usr/bin/python3 "${SCRIPT_DIR}/camera_session.py" --mode enroll -- \
    "${PYTHON}" "${ROOT_DIR}/remember_person.py" --replace "$@" -- "${NAME}"
