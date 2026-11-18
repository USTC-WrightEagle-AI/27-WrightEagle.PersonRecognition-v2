# 27-WrightEagle.PersonRecognition-v2
将姿势识别改为resnet18后的版本

## 一键脚本

机器人上仓库路径为 `/home/rpp/Robocup/PersonRecognition-v2`。两个脚本默认会
自动管理头部 D455 的临时分辨率，无需手动杀相机进程或重新启动相机。

- 已有 `/head/camera`：保存原 RGB、深度 profile、启流、深度对齐状态及启动参数，
  只停止头部相机进程，再按项目要求的 `1280,720,30` 启动。
  本机 RealSense 驱动在动态改 profile 时可能阻塞，因此默认使用受控重启。
- 启动应用前验证新图像尺寸和 RGB-D 时间戳配对；配置不匹配或没有新图像时恢复原配置并退出。
- 录入完成、程序报错、`Ctrl+C`、`SIGTERM` 或关闭终端时，先结束应用，再恢复启动前的参数，
  并检查原分辨率的新图像。原配置是 `640,480,15` 时就恢复到该配置，不写死默认分辨率。
- 原来没有头部相机节点时临时启动一个，结束后关闭，恢复到未启动的状态。

切换期间所有订阅头部相机的程序都会短暂断流并看到分辨率变化。
同一 ROS domain 内同时只允许一个自动相机会话；结束录入后再启动检测。
相机节点来自已有 bringup 时，机器人主启动程序和左右相机保持运行；
替换后的头部相机由管理进程监视原启动程序，原启动程序退出时自动关闭，避免遗留占用。

`kill -9` 或断电无法执行退出清理；若留下未完成的参数恢复记录，下次运行会提示先恢复：

```bash
./scripts/restore_camera.sh
```

恢复记录存于用户私有的 `/tmp/person-recognition-camera-UID-domain-DOMAIN/`，
包含相机参数、原启动命令、原启动程序的身份及相机进程环境，不包含图像或人脸信息。
重启方式的恢复需要原机器人启动程序仍在线；系统重启或清理 `/tmp` 后记录可能不存在。
如需手动管理相机，可通过 `PERSON_CAMERA_AUTO=0 ./scripts/enroll_face.sh` 或
`PERSON_CAMERA_AUTO=0 ./scripts/start_detection.sh` 关闭自动切换。

一键人脸录入：运行后按提示输入被录入人的姓名（同名数据自动覆盖），
自动检查 ROS 与 `robot-face` 环境后调用 `remember_person.py`：

```bash
./scripts/enroll_face.sh
```

两个入口会加载 ROS Humble 和机器人工作区 `/home/rpp/rpp_ws/install/setup.bash`，
确保使用机器人工作区内的 RealSense 驱动。可用 `PERSON_ROBOT_SETUP` 指定其他工作区环境文件。
桌面启动器未设置 `ROS_LOCALHOST_ONLY` 时，入口默认使用 `1`，与机器人驱动一致；
调用者显式设置的值会保留。相机进程存在而当前 ROS 网络不可见时直接报错，不重复启动相机。
新启动相机会等待完整参数初始化（最长 30 秒）再核对 profile；参数服务刚出现不代表初始化完成。

额外参数原样传给 `remember_person.py`，例如：

```bash
./scripts/enroll_face.sh --no-depth-filter
./scripts/enroll_face.sh --head-backend sdk   # 需先停止占用头部 D455 的 ROS 相机
```

`--no-depth-filter` 只要求 RGB 和内参，不再强制检查或修改深度流。
`--camera` 指定其他相机、或选择 `--head-backend sdk` 时不自动管理 ROS 相机；
SDK 模式仍需要先释放占用。`--help` 可直接查看参数，无需输入姓名或启动相机。

一键综合检测（人脸身份 + 姿态分类 + 三维位置）：转交
`posture_recognition/scripts/start_3d.sh`，自动准备相机后再检查环境和依赖。
目标配置读取 `POSE_3D_CONFIG` 或 `--config` 指定的配置文件：

```bash
./scripts/start_detection.sh
./scripts/start_detection.sh --no-image       # 不发布可视化图像，也不打开画面窗口
./scripts/start_detection.sh --frame-skip 1   # 每两帧处理一帧，降低 GPU 占用
```

桌面环境下默认自动打开实时检测画面窗口（`/pose_classifier_3d/image`），
终端按 `Ctrl+C` 后检测与窗口一起结束，再恢复相机配置。
单独关闭画面窗口时检测继续运行；无桌面显示环境时只发布图像话题。
starterButton 的“4-综合检测”按钮调用同一入口，自动显示画面。

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

隔离 ROS domain 内的相机会话模拟验证（不使用真实相机、不录入人脸）：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 tests/ros_camera_session_smoke.py
```

2026-11-18 实机验证包含从原低分辨率脚本启动的 `640x480@15` 自动切换到
`1280x720@30`，录入通道连续读取 8 组 RGB-D，正常结束和实际 SIGINT 后恢复低分辨率。
无头部相机节点时的冷启动也已验证：收到 3 帧高清检测图像、5 条时间戳推进的结果，
查看器已订阅图像，退出后临时相机关闭（当时未检测到人）。测试没有录入或保存人脸。
动态参数切换曾在实机驱动中阻塞，因此默认采用重启方式，并在重新打开设备前留出
2 秒等待 USB/传感器释放。
