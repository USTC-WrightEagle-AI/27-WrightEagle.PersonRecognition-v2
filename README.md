# 27-WrightEagle.PersonRecognition-v2
将姿势识别改为resnet18后的版本

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
