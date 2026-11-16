# 头部相机人体静态姿态识别

本目录迁移自
[`USTC-WrightEagle-AI/27-WrightEagle.Human-static-posture-recognition-by-pzy`](https://github.com/USTC-WrightEagle-AI/27-WrightEagle.Human-static-posture-recognition-by-pzy)。
原项目使用 ROS 1 Noetic 和另一台 D455；当前版本已适配本机 ROS 2 Humble
及头部 D455。

系统使用 YOLOv8-Pose 提取人体框和 COCO17 骨架，通过头部 D455 对齐深度
反投影为三维骨架和人体位置。姿势分类由 POLAR 数据训练的 ResNet18 完成：
按每个人体框裁剪 RGB 图像，识别站立、坐下、蹲、躺和弯腰。当前配置处理
画面中的所有人员，YOLO-Pose 检测和 ByteTrack 跟踪只运行一条流程。

2026-11-10 起，默认 ROS 节点使用 ResNet18 替换 TCN。骨架、人脸全局关联、
三维定位、ROS 话题及对外字段继续使用原有流程。分类器只依赖当前人体裁图；
深度无效时仍可输出姿势，坐标则按原规则省略，绝不以分类结果补造三维点。

当前版本还使用 ByteTrack 为每个人维持 `person_id`。人脸子进程每次只对
整帧运行一次 InsightFace，再根据 YOLOv8-Pose 的耳朵、眼睛、鼻子和肩膀
关键点为每个身体生成虚拟脸框。真实脸框与虚拟脸框使用 IoU、修正 IoU、
中心距离和尺度一致性评分，通过 Hungarian 全局一对一分配后绑定到
ByteTrack ID。姓名仍需要在同一 ID 上多帧确认；深度位置、姓名和姿态
共用同一个 ByteTrack ID。

人脸与身体关联门限位于 `config/pose_3d_config.yaml` 的 `face_recognition`
段：`keypoint_confidence` 是生成虚拟脸框的关键点置信度门限，
`min_assignment_iou` 和 `min_assignment_score` 分别是候选对的最低 IoU 和综合
分数。降低门限会提高远距离召回率，但会增加多人串名风险。

## 本机相机配置

- 相机：头部 Intel RealSense D455
- 序列号：`260722304986`
- 流配置：`1280x720@30`
- RGB：`/head/camera/color/image_raw`
- 对齐深度：`/head/camera/aligned_depth_to_color/image_raw`
- 内参：`/head/camera/color/camera_info`
- 标定快照：`../calibration/head_d455_hd/`

联合识别的人脸检测器输入为 `1280`，与独立录入和识别一致；人体姿态模型输入仍为 `640`。

节点启动时会核对相机参数服务中的序列号和 RGB/Depth profile，并将实时
`CameraInfo` 与高清流对应的工厂内参比较。任何设备、分辨率、帧率或内参
不匹配都会拒绝启动。

`camera.down_axis` 是 2026-10-16 在机器人静止时连续采集 5 秒 D455 IMU
数据，再通过设备内部 IMU 到彩色光学系外参转换得到：
`[-0.004230, 0.997762, 0.066737]`。它相对理想图像向下方向
`[0, 1, 0]` 偏离约 `3.83 deg`，主要是相机俯仰。该重力轴不使用
`handeye_candidate_v3.json` 中未验收的左臂外参；头部相机安装姿态变化后
必须重新测量。

## 运行

姿态进程需要相互匹配的 PyTorch/TorchVision，建议保留机器人现有 CUDA 版本。
`posture_recognition/requirements.txt` 列出了运行依赖，NumPy 限制在 1.x 以兼容
Humble 的 `cv_bridge`。启动脚本会检查 TorchVision；无需重新训练模型。

先启动机器人 bringup，确认头部 RGB 和对齐深度话题存在，然后运行：

```bash
cd /home/rpp/Robocup/pzy-cs
./posture_recognition/scripts/start_3d.sh
```

默认使用本机已有的 `foundationpose` Conda 环境。无可视化图像输出时：

```bash
./posture_recognition/scripts/start_3d.sh --no-image
```

跳过部分输入帧可降低 GPU 占用，例如每两帧处理一帧：

```bash
./posture_recognition/scripts/start_3d.sh --frame-skip 1
```

## ROS 2 输出

- `/pose_classifier_3d/results`：完整姿态、ID、姓名和坐标 JSON
- `/person_identity_locations`：精简的 ID、姓名和坐标 JSON
- `/pose_classifier_3d/image`：`sensor_msgs/msg/Image`，骨架与姿态可视化

完整结果中的 `person_id`、`bbox`、`name`、`name_known`、`face_score`、`pose`、`pose_cn`、
`conf`、`distance_m`、`camera_xyz_m`、`left_base_xyz_m`、`left_base_transform_validated` 沿用原有命名、
类型和坐标约定，时间戳与可选坐标字段规则也沿用原接口。`pose` 类别字符串不变，
`sitting` 的中文仍为“坐下”。图像分类结果直接绑定检测得到的 ByteTrack ID。

`conf` 现在是按 ID 平滑后的 ResNet18 最大 softmax 分数，不是经过校准的正确率。
`features` 保留原有诊断字段：`track_id、history_size、warmup、stale_feature、stale_count`；
其中历史长度现在统计有效 RGB 裁图，`stale_feature/stale_count` 表示无效裁图及连续次数，
不再表示深度特征失效。开启 `--debug` 且三维骨架有效时仍输出膝角等几何诊断，
这些诊断不参与姿势分类。TCN 专有的深蹲强制改类逻辑不再执行。

查看结果：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /pose_classifier_3d/results
ros2 topic echo /person_identity_locations
./posture_recognition/scripts/view_image.sh
```

精简输出示例：

```json
{
  "camera_coordinate_frame": "head_camera_color_optical_frame",
  "camera_coordinate_convention": "+X right, +Y down, +Z forward",
  "left_base_transform_validated": false,
  "persons": [
    {
      "person_id": 1,
      "name": "张三",
      "name_known": true,
      "face_score": 0.71,
      "camera_xyz_m": [0.10, 0.42, 2.27],
      "left_base_xyz_m": [2.39, 0.21, -0.20],
      "left_base_transform_validated": false
    }
  ]
}
```

`camera_xyz_m` 是可信主输出，坐标约定为相机光学系 `+X` 向右、`+Y` 向下、
`+Z` 向前。`left_base_xyz_m` 来自昨天的候选手眼外参；由于该外参未通过旋转
验收，其 `left_base_transform_validated` 固定为 `false`，不能直接用于机械臂
抓取或安全相关动作。

姓名来自当前项目的 `data/faces.npz`。只有人脸清晰可见并连续两次通过阈值后，
才会把姓名绑定到 ByteTrack ID；否则输出 `Unknown`。录入或更新人员仍使用：

```bash
cd /home/rpp/Robocup/pzy-cs
/home/rpp/miniconda3/envs/robot-face/bin/python remember_person.py 张三 --replace
```

不要在当前 Conda `base` 环境中直接执行 `ros2 run rqt_image_view ...`：该环境的
`python3` 是 3.14，而 ROS 2 Humble 的 `rclpy` 扩展是为系统 Python 3.10 编译的。
查看脚本会显式使用 `/usr/bin/python3`，不受当前 Conda 环境影响。等价命令为：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view \
  /pose_classifier_3d/image
```

## RGB 分类参数

`config/pose_3d_config.yaml` 的 `classifier` 段控制 ResNet18：

- `crop_margin: 0.05`：人体框每边外扩 5%，截断到图像范围；BGR 转 RGB 后按比例
  缩放，以训练权重中的 RGB 补边值填充至 224×224，不做中心裁剪或人体拉伸。
- `batch_size: 16`：多人分批分类，保持与原人体检测顺序及 ID 对齐。
- `smoothing_alpha: 0.6`：每个 ByteTrack ID 独立对五类概率做 EMA；设为 `1.0`
  可关闭历史影响。无效裁图立即输出 `unknown`，清除该 ID 的旧分类概率。
- `min_history: 1`：有效裁图首帧即可分类，不再需要 TCN 的时序输入；可提高此值
  延迟首次有效输出。`max_track_age: 15` 按处理帧数清理离场轨迹的分类状态。
- `confidence_threshold: 0.0`：默认不因低分拒识；未知类别仍为 `unknown`。
  如需拒识，应使用 D455 现场验证集选择阈值，不能直接沿用原 TCN 的 `0.4`。

下游可继续解析原 JSON，但依赖 `conf` 数值、分类预热时长或诊断字段含义的逻辑
需要按上述语义验证。身份识别、三维几何和姿势分类共用同一个 `person_id`。

## 模型

- `models/yolov8n-pose.pt`：本机 CUDA PyTorch/Ultralytics 使用的姿态检测模型
- `models/resnet18_polar5.pth`：当前分类器；来源于 POLAR 训练运行
  `resnet18_polar5_20261108_01/best_model.pth`，由验证集选择第 25 轮权重。
- `models/pose_tcn_ntu_fukinect.pth`：保留的上游 TCN 权重，用于历史回归测试，默认节点不加载
- `models/yolov8n-pose.engine`：保留的上游 Jetson TensorRT 文件，本机不使用

校验模型：

```bash
cd /home/rpp/Robocup/pzy-cs/posture_recognition/models
sha256sum -c SHA256SUMS
```

ResNet18 的来源、预处理和校验信息见 `models/resnet18_polar5.json`。保守清洗的
POLAR 留出集（466 张人体裁图）准确率 96.57%、宏平均 F1 96.41%；这不是 D455
实机或完整检测跟踪流程的准确率，远距、遮挡和俯视仍需独立现场验证。

## 测试

```bash
cd /home/rpp/Robocup/pzy-cs
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/rpp/Robocup/pzy-cs/posture_recognition/src:$PYTHONPATH
/home/rpp/miniconda3/envs/foundationpose/bin/python \
  -m unittest discover -s posture_recognition/tests -v
```

测试覆盖 POLAR 实权重加载、训练预处理一致性、多人 ID 顺序变化、无效裁图、
状态过期、深度缺失和原 ROS 输出字段。ROS 管线测试使用传输替身，并调用实际
节点处理方法；不能替代机器人上的 D455、DDS、人脸子进程联调。

三人耗时对比脚本为 `scripts/benchmark_pose_classifiers.py`，读取原 TCN 版本的
Git 节点代码和配置，与当前 ResNet18 节点在相同 CUDA 环境中交替测试：

```bash
python posture_recognition/scripts/benchmark_pose_classifiers.py \
  --video rgb_posture_recognition/examples/input_multi.avi \
  --frame-indices 2,3,4,7,8,9 \
  --output runs/three_person_classifier_benchmark
```

上述素材为本地合成视频；选取的帧中三人都有有效躯干关键点，以保证 TCN 每次
也实际分类三人。每轮先预热 60 帧，主处理链每种方案测量 4×300 帧；分类器
分段计时和含绘图回调分别另测 2×150 帧。FPS 按总帧数除以总耗时计算。
脚本检查三人 ID、有效 TCN 历史和真实模型 batch size，并保存逐帧 CSV、环境
信息及权重校验值。深度为固定 2 m 合成数据，人脸网络、全局人脸关联、相机采集、
ROS 通信及 `cv_bridge` 拷贝均不计时；这些数值不代表机器人实机帧率或准确率。
