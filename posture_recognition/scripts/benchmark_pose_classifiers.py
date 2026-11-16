#!/usr/bin/env python3
"""Compare the current RGB classifier with the original TCN ROS processing code.

This is an offline latency benchmark. ROS imports/transports are stubbed; the
actual YOLO-Pose, ByteTrack, depth backprojection and classifiers execute.
No face network, camera capture, DDS or video decoding is included in timings.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageFont
import torch
import torchvision
import ultralytics
from ultralytics import YOLO
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE / "src"))
from depth_backproject import DepthBackProjector
from person_identity import NameBinder
from rgb_classifier import ResNetPoseClassifier
from tcn_classifier import TCNPoseClassifier


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT).decode("utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_node(source, name):
    names = ["cv_bridge", "rclpy", "rclpy.node", "rclpy.qos", "rcl_interfaces",
             "rcl_interfaces.srv", "sensor_msgs", "sensor_msgs.msg", "std_msgs",
             "std_msgs.msg", "message_filters"]
    stubs = {key: types.ModuleType(key) for key in names}
    stubs["cv_bridge"].CvBridge = object
    stubs["cv_bridge"].CvBridgeError = RuntimeError
    stubs["rclpy.node"].Node = object
    stubs["rclpy.qos"].qos_profile_sensor_data = object()
    stubs["rcl_interfaces.srv"].GetParameters = object
    stubs["sensor_msgs.msg"].CameraInfo = object
    stubs["sensor_msgs.msg"].Image = object
    stubs["std_msgs.msg"].String = lambda data: types.SimpleNamespace(data=data)
    module = types.ModuleType(name)
    module.__file__ = str(PACKAGE / "src/pose_classifier_3d_ros.py")
    with patch.dict(sys.modules, stubs):
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.PoseClassifier3DNode


class Publisher:
    def publish(self, message):
        self.last_message = message


def make_node(node_class, config, classifier, width, height, font):
    node = object.__new__(node_class)
    node.classifier = classifier
    node.model = YOLO(str(PACKAGE / "models/yolov8n-pose.pt"))
    model_config = config["model"]
    node.conf_threshold = model_config["conf_threshold"]
    node.input_size = model_config["input_size"]
    node.device = model_config["device"]
    node.tracker_config = str(PACKAGE / "config/bytetrack.yaml")
    node.down_axis = np.asarray(config["camera"]["down_axis"], dtype=float)
    node.down_axis /= np.linalg.norm(node.down_axis)
    intrinsics = json.loads((ROOT / "calibration/head_d455_hd/intrinsics.json").read_text(encoding="utf-8"))
    matrix = np.asarray(intrinsics["camera_matrix"])
    depth = config["camera"]["depth"]
    node.backprojector = DepthBackProjector(
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
        min_depth=depth["min_valid"], max_depth=depth["max_valid"],
        hole_radius=depth["hole_search_radius"])
    node.nearest_only = False
    node.face_client = None
    node.face_interval = config["face_recognition"]["every_n_frames"]
    transform_path = (PACKAGE / "config" / config["localization"]["candidate_base_transform"]).resolve()
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    node.base_transform = np.asarray(transform["transforms"]["T_base_camera"]["matrix_4x4"])
    node.base_transform_validated = bool(transform.get("validation", {}).get("passed", False))
    node.image_width, node.image_height = width, height
    node.publish_image, node.frame_skip, node.debug = False, 0, False
    node.results_pub, node.identity_locations_pub, node.image_pub = Publisher(), Publisher(), Publisher()
    node._log_throttled = lambda *args: None
    node.bridge = types.SimpleNamespace(
        imgmsg_to_cv2=lambda message, encoding: message.frame,
        cv2_to_imgmsg=lambda frame, encoding: types.SimpleNamespace(frame=frame))
    if font:
        node.label_font = ImageFont.truetype(str(font), 22)
        node.stats_font = ImageFont.truetype(str(font), 24)
    return node


def reset_node(node, config):
    if isinstance(node.classifier, ResNetPoseClassifier):
        node.classifier.reset()
    else:
        node.classifier.tracker.tracks.clear()
        node.classifier.tracker.frame_index = 0
        node.classifier.tracker.next_track_id = 0
    if node.model.predictor is not None:
        for tracker in node.model.predictor.trackers:
            tracker.reset()
    face = config["face_recognition"]
    node.name_binder = NameBinder(face["confirmations"], face["history_size"], face["max_track_age"])
    node.frame_sequence = node.frame_count = 0
    node.last_callback_time, node.realtime_fps = None, 0.0


def read_frames(path, width, height):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    metadata = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": capture.get(cv2.CAP_PROP_FPS)}
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        padded = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
        padded[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        frames.append(padded)
    capture.release()
    if not frames:
        raise RuntimeError("Video contains no frames")
    metadata["frames"] = len(frames)
    return frames, metadata


def verify(people, name):
    if len(people) != 3 or sorted(p["person_id"] for p in people) != [1, 2, 3]:
        raise AssertionError(f"Expected three stable IDs, got {[(p['person_id'], p['pose']) for p in people]}")
    for person in people:
        feature = person["features"]
        if feature["warmup"] or feature["stale_feature"]:
            raise AssertionError(f"{name} did not classify a current valid input: {feature}")
        if name == "tcn" and feature["history_size"] != 24:
            raise AssertionError(f"TCN history not full: {feature}")
        if "camera_xyz_m" not in person or "left_base_xyz_m" not in person:
            raise AssertionError("3D localization was not executed")


def stats(values):
    array = np.asarray(values, dtype=float)
    return {"frames": len(array), "mean_ms": float(array.mean()),
            "median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
            "std_ms": float(array.std()), "fps_from_mean": float(1000 / array.mean())}


class StageProxy:
    def __init__(self, target, method, row, key):
        self.target, self.method, self.row, self.key = target, method, row, key

    def __getattr__(self, name):
        method = getattr(self.target, name)
        if name != self.method:
            return method

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = method(*args, **kwargs)
            torch.cuda.synchronize()
            self.row[self.key] = (time.perf_counter_ns() - start) / 1e6
            return result
        return timed


def telemetry():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.used,temperature.gpu,power.draw,clocks.current.graphics",
            "--format=csv,noheader"], encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "rgb_posture_recognition/examples/input_multi.avi")
    parser.add_argument("--frame-indices", help="Comma-separated zero-based source frames, recorded in the report")
    parser.add_argument("--baseline-ref", default="dfd3c8699ec73e07d9e3c4627059b19b477f754c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--profile-frames", type=int, default=150)
    parser.add_argument("--draw-frames", type=int, default=150)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=0,
                        help="0 preserves Ultralytics' single-threaded OpenCV runtime setting")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.warmup < 24 or min(args.rounds, args.frames, args.profile_frames, args.cpu_threads) < 1:
        parser.error("Require at least 24 warmup frames and positive measurement counts")
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.opencv_threads)
    torch.manual_seed(0)
    config_path = PACKAGE / "config/pose_3d_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_config = yaml.safe_load(git("show", f"{args.baseline_ref}:posture_recognition/config/pose_3d_config.yaml"))
    for key in config:
        if key != "classifier" and config[key] != baseline_config[key]:
            raise AssertionError(f"Shared config differs: {key}")
    # Verify that the imported historical classifier has not changed since the baseline.
    for filename in ("tcn_classifier.py", "tcn_model.py", "pose_features.py", "depth_backproject.py", "person_identity.py"):
        baseline = git("show", f"{args.baseline_ref}:posture_recognition/src/{filename}")
        current = (PACKAGE / "src" / filename).read_text(encoding="utf-8")
        if baseline != current:
            raise AssertionError(f"Shared/baseline implementation changed: {filename}")
    width, height = config["camera"]["width"], config["camera"]["height"]
    frames, video = read_frames(args.video, width, height)
    frame_indices = list(range(len(frames))) if not args.frame_indices else [int(index) for index in args.frame_indices.split(",")]
    if not frame_indices or any(index < 0 or index >= len(frames) for index in frame_indices):
        parser.error("Frame indices must refer to existing video frames")
    frames = [frames[index] for index in frame_indices]
    depth = np.full((height, width), 2000, dtype=np.uint16)
    current_source = (PACKAGE / "src/pose_classifier_3d_ros.py").read_text(encoding="utf-8")
    baseline_source = git("show", f"{args.baseline_ref}:posture_recognition/src/pose_classifier_3d_ros.py")
    fonts = [args.font] if args.font else [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    font = next((path for path in fonts if path and path.is_file()), None)
    if args.draw_frames and font is None:
        parser.error("A Chinese font is required for the visualization benchmark")
    nodes = {
        "tcn": make_node(load_node(baseline_source, "benchmark_tcn_node"), baseline_config,
                         TCNPoseClassifier.from_config(baseline_config["classifier"], PACKAGE / "models/pose_tcn_ntu_fukinect.pth"),
                         width, height, font),
        "resnet18": make_node(load_node(current_source, "benchmark_rgb_node"), config,
                              ResNetPoseClassifier.from_config(config["classifier"], PACKAGE / "models/resnet18_polar5.pth"),
                              width, height, font),
    }
    # The RGB adapter config sets Torch's thread count; explicitly equalize both runs.
    torch.set_num_threads(args.cpu_threads)
    report = {
        "command": sys.argv,
        "measured_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "Offline sequential GPU latency, real original/current node methods; synthetic RGB montage and constant valid depth",
        "excluded": ["face detector/recognizer and global face association", "camera capture and RGB-depth synchronization",
                     "video decode and resize", "ROS/DDS and cv_bridge copies", "model load and warmup", "actual log I/O"],
        "baseline_commit": args.baseline_ref,
        "environment": {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__, "opencv": cv2.__version__,
                        "ultralytics": ultralytics.__version__, "torch_cpu_threads": torch.get_num_threads(),
                        "opencv_threads": cv2.getNumThreads(), "font": str(font), "gpu_before": telemetry()},
        "input": {"video": str(args.video.resolve()), "sha256": sha256(args.video), "source": video,
                  "selected_source_frame_indices": frame_indices,
                  "processed_size": [width, height], "resize": "aspect preserving linear resize, centered black padding, preloaded",
                  "depth": "synthetic constant 2000 mm; no depth holes", "persons": 3,
                  "detector_imgsz": config["model"]["input_size"], "frame_skip": 0},
        "method": {"warmup_frames_per_block": args.warmup, "rounds": args.rounds, "frames_per_round": args.frames,
                   "profiling_rounds": 2, "profile_frames_per_round": args.profile_frames,
                   "draw_rounds": 2 if args.draw_frames else 0, "draw_frames_per_round": args.draw_frames,
                   "timer": "perf_counter_ns with CUDA synchronization before and after each full call",
                   "fps": "1000 / arithmetic mean milliseconds, no input FPS cap, no averaging instantaneous FPS",
                   "order": "TCN then ResNet18 on even rounds, reversed on odd rounds",
                   "classifier_profile": "Separate instrumented runs; includes preprocessing, transfer, model, postprocessing and state",
                   "precision": "default FP32; no added half precision, compilation or TensorRT"},
        "classifier_configs": {"tcn": baseline_config["classifier"], "resnet18": config["classifier"]},
        "weights": {path.name: sha256(path) for path in [PACKAGE / "models/yolov8n-pose.pt", PACKAGE / "models/pose_tcn_ntu_fukinect.pth", PACKAGE / "models/resnet18_polar5.pth"]},
        "code": {"benchmark_sha256": sha256(__file__), "current_node_sha256": sha256(PACKAGE / "src/pose_classifier_3d_ros.py"),
                 "rgb_classifier_sha256": sha256(PACKAGE / "src/rgb_classifier.py")},
        "blocks": [],
    }
    rows = []

    def warmup(node, name, draw=False):
        reset_node(node, config)
        node.publish_image = draw
        for index in range(args.warmup):
            frame = frames[index % len(frames)]
            if draw:
                header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=index, nanosec=0), frame_id="head_camera_color_optical_frame")
                node.callback(types.SimpleNamespace(frame=frame, header=header),
                              types.SimpleNamespace(frame=depth, encoding="16UC1"))
                people = json.loads(node.results_pub.last_message.data)["persons"]
            else:
                people = node._detect_and_classify(frame, depth)
            if index >= 24:
                verify(people, name)
        verify(people, name)
        torch.cuda.synchronize()

    for scope, rounds, count in [("core", args.rounds, args.frames),
                                 ("profile", 2, args.profile_frames),
                                 ("callback_with_drawing", 2 if args.draw_frames else 0, args.draw_frames)]:
        for round_index in range(rounds):
            order = ("tcn", "resnet18") if round_index % 2 == 0 else ("resnet18", "tcn")
            for name in order:
                node = nodes[name]
                warmup(node, name, draw=scope == "callback_with_drawing")
                stage = {}
                model, classifier = node.model, node.classifier
                batch_shapes = []
                hook = None
                if scope == "profile":
                    node.model = StageProxy(model, "track", stage, "detector_tracking_ms")
                    node.classifier = StageProxy(classifier, "classify_batch", stage, "classifier_ms")
                    hook = classifier.model.register_forward_pre_hook(lambda module, inputs: batch_shapes.append(list(inputs[0].shape)))
                node.publish_image = scope == "callback_with_drawing"
                block_rows = []
                for index in range(count):
                    frame = frames[(args.warmup + index) % len(frames)]
                    header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=index, nanosec=0), frame_id="head_camera_color_optical_frame")
                    rgb_message = types.SimpleNamespace(frame=frame, header=header)
                    depth_message = types.SimpleNamespace(frame=depth, encoding="16UC1")
                    stage.clear()
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    if node.publish_image:
                        node.callback(rgb_message, depth_message)
                    else:
                        people = node._detect_and_classify(frame, depth)
                    torch.cuda.synchronize()
                    duration = (time.perf_counter_ns() - start) / 1e6
                    if node.publish_image:
                        people = json.loads(node.results_pub.last_message.data)["persons"]
                    verify(people, name)
                    row = {"scope": scope, "round": round_index + 1, "classifier": name,
                           "frame": index, "total_ms": duration, "persons": len(people), **stage}
                    if scope == "profile":
                        if batch_shapes[-1][0] != 3 or len(batch_shapes) != index + 1:
                            raise AssertionError("Expected one actual neural inference batch containing all three people")
                        row["geometry_and_output_ms"] = duration - stage["detector_tracking_ms"] - stage["classifier_ms"]
                    block_rows.append(row)
                if hook:
                    hook.remove()
                node.model, node.classifier = model, classifier
                rows.extend(block_rows)
                block = {"scope": scope, "round": round_index + 1, "classifier": name,
                         **stats([row["total_ms"] for row in block_rows]), "gpu": telemetry()}
                if batch_shapes:
                    block["inference_batch_shapes"] = sorted({tuple(shape) for shape in batch_shapes})
                report["blocks"].append(block)
                print(json.dumps(block, ensure_ascii=True), flush=True)

    report["summary"] = {}
    for scope in ("core", "profile", "callback_with_drawing"):
        scope_results = {}
        for name in nodes:
            selected = [row for row in rows if row["scope"] == scope and row["classifier"] == name]
            if not selected:
                continue
            scope_results[name] = {"total": stats([row["total_ms"] for row in selected])}
            for key in ("detector_tracking_ms", "classifier_ms", "geometry_and_output_ms"):
                if key in selected[0]:
                    scope_results[name][key] = stats([row[key] for row in selected])
        report["summary"][scope] = scope_results
    core = report["summary"]["core"]
    report["comparison"] = {
        "resnet_added_mean_ms": core["resnet18"]["total"]["mean_ms"] - core["tcn"]["total"]["mean_ms"],
        "resnet_latency_change_pct": 100 * (core["resnet18"]["total"]["mean_ms"] / core["tcn"]["total"]["mean_ms"] - 1),
        "resnet_fps_change_pct": 100 * (core["resnet18"]["total"]["fps_from_mean"] / core["tcn"]["total"]["fps_from_mean"] - 1),
    }
    report["environment"].update({"gpu_after": telemetry(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                  "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32})
    report["validation"] = {"all_measured_frames_have_three_stable_ids": True,
                            "tcn_has_full_24_frame_valid_history": True,
                            "both_classifiers_execute_batch_size_three": True,
                            "positions_emitted_on_all_measured_frames": True,
                            "accuracy_evaluated": False, "real_ros_or_face_network_tested": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "frame_times.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scope", "round", "classifier", "frame", "total_ms", "persons",
                                                    "detector_tracking_ms", "classifier_ms", "geometry_and_output_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": report["summary"], "comparison": report["comparison"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
