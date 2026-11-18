# 27-WrightEagle.PersonRecognition-v2
将姿势识别改为resnet18后的版本

## 一键脚本

机器人上仓库路径为 `/home/rpp/Robocup/PersonRecognition-v2`。两个脚本都要求
头部 D455 已按 `1280,720,30` 启动 RGB 和对齐深度（相机要求见下文），
脚本会在启动前检查话题，不满足时给出明确错误。

一键人脸录入：运行后按提示输入被录入人的姓名（同名数据自动覆盖），
自动检查 ROS 与 `robot-face` 环境后调用 `remember_person.py`：

```bash
./scripts/enroll_face.sh
```

额外参数原样传给 `remember_person.py`，例如：

```bash
./scripts/enroll_face.sh --no-depth-filter
./scripts/enroll_face.sh --head-backend sdk   # 需先停止占用头部 D455 的 ROS 相机
```

一键综合检测（人脸身份 + 姿态分类 + 三维位置）：转交
`posture_recognition/scripts/start_3d.sh`，环境、依赖与相机检查沿用其原有逻辑：

```bash
./scripts/start_detection.sh
./scripts/start_detection.sh --no-image       # 不输出可视化图像
./scripts/start_detection.sh --frame-skip 1   # 每两帧处理一帧，降低 GPU 占用
```

结果话题为 `/pose_classifier_3d/results` 和 `/person_identity_locations`，
字段说明见 `posture_recognition/README.md`。人脸库 `data/faces.npz` 不存在时
检测仍可运行，但所有姓名输出 `Unknown`，请先用录入脚本录入人脸。

## 人脸录入

在仓库根目录运行，先启动头部 D455 的 ROS 相机：

```bash
source /opt/ros/humble/setup.bash
/home/rpp/miniconda3/envs/robot-face/bin/python remember_person.py PERSON_NAME --replace
```

相机必须为序列号 `260722304986`，RGB 和深度均为 `1280,720,30`，
并发布对齐深度；只运行一个 `/head/camera` 节点。将 `PERSON_NAME` 替换为要录入的姓名，
`--replace` 替换同名数据。`data/` 内的人脸数据库和采集数据仅保存在本地，不提交到 Git。

默认按深度选择最近的人。RGB 和深度使用有界队列匹配，时间戳差不超过
50 毫秒，队列内停留超过 0.5 秒的帧会丢弃。无法配对时，错误会分别列出
RGB、深度的收帧数、距上次收到帧的时间、两路最新时间戳差，以及内参和
相机参数的校验状态。缓冲配对能处理短暂的到达顺序差异，不能修复相机
停流或数秒的深度滞后。

只录入一人时，可以显式关闭深度筛选，保持画面中只有该人员：

```bash
/home/rpp/miniconda3/envs/robot-face/bin/python remember_person.py PERSON_NAME --replace --no-depth-filter
```

也可先停止占用头部 D455 的 ROS 相机，再使用 SDK 直接采集 RGB-D：

```bash
/home/rpp/miniconda3/envs/robot-face/bin/python remember_person.py PERSON_NAME --replace --head-backend sdk
```

采集完成后出现 `Saved 'PERSON_NAME' ...`（实际姓名）才表示已保存。配对逻辑回归测试：

```bash
python -m unittest discover -s tests -v
```
