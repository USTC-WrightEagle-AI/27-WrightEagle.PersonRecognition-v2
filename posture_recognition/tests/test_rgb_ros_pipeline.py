"""Exercise the original ROS pipeline with transport stubs, not live DDS."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.engine.results import Results

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from depth_backproject import DepthBackProjector
from person_identity import NameBinder
from rgb_classifier import ResNetPoseClassifier


def load_ros_node():
    names = ["cv_bridge", "rclpy", "rclpy.node", "rclpy.qos", "rcl_interfaces",
             "rcl_interfaces.srv", "sensor_msgs", "sensor_msgs.msg", "std_msgs",
             "std_msgs.msg", "message_filters"]
    stubs = {name: types.ModuleType(name) for name in names}
    stubs["cv_bridge"].CvBridge = object
    stubs["cv_bridge"].CvBridgeError = RuntimeError
    stubs["rclpy.node"].Node = object
    stubs["rclpy.qos"].qos_profile_sensor_data = object()
    stubs["rcl_interfaces.srv"].GetParameters = object
    stubs["sensor_msgs.msg"].CameraInfo = object
    stubs["sensor_msgs.msg"].Image = object
    stubs["std_msgs.msg"].String = lambda data: types.SimpleNamespace(data=data)
    spec = importlib.util.spec_from_file_location("rgb_ros_contract", ROOT / "src" / "pose_classifier_3d_ros.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.PoseClassifier3DNode


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


class RGBDPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.NodeClass = load_ros_node()
        cls.classifier = ResNetPoseClassifier(ROOT / "models" / "resnet18_polar5.pth", device="cpu")

    def setUp(self):
        node = object.__new__(self.NodeClass)
        self.classifier.reset()
        node.classifier = self.classifier
        node.conf_threshold, node.input_size, node.device = .35, 640, "cpu"
        node.tracker_config = str(ROOT / "config" / "bytetrack.yaml")
        node.down_axis = np.array([0., 1., 0.])
        node.backprojector = DepthBackProjector(100, 100, 100, 50)
        node.nearest_only = False
        node.face_interval, node.frame_sequence = 1, 0
        node.name_binder = NameBinder(confirmations=1)
        self.face_requests = []
        node.face_client = types.SimpleNamespace(
            poll=lambda: {"matches": [{"track_id": 41, "name": "Alice", "known": True, "score": .9}]},
            submit=lambda *args: self.face_requests.append(args), pop_error=lambda: None)
        node.base_transform = np.eye(4)
        node.base_transform[:3, 3] = [1, 2, 3]
        node.base_transform_validated = False
        node._log_throttled = lambda *args: None
        self.tracker_calls = []
        self.empty = False
        points = np.zeros((2, 17, 2), dtype=np.float32)
        for i, x in enumerate([50, 150]):
            points[i, :] = [x, 45]
            points[i, 5], points[i, 6] = [x - 10, 30], [x + 10, 30]
            points[i, 11], points[i, 12] = [x - 10, 60], [x + 10, 60]
        self.points = points
        self.output = Results(
            orig_img=np.zeros((120, 200, 3), dtype=np.uint8), path="synthetic", names={0: "person"},
            boxes=torch.tensor([[20, 5, 80, 100, 41, .9, 0], [120, 5, 180, 100, 9, .9, 0]]),
            keypoints=torch.cat([torch.from_numpy(points), torch.ones((2, 17, 1))], dim=2))

        def track(frame, **kwargs):
            self.tracker_calls.append(kwargs)
            return [] if self.empty else [self.output]

        node.model = types.SimpleNamespace(track=track)
        node.publish_image, node.frame_skip, node.frame_count = False, 0, 0
        node.last_callback_time, node.realtime_fps, node.debug = None, 0.0, False
        node.image_width, node.image_height = 200, 120
        node.results_pub, node.identity_locations_pub = Publisher(), Publisher()
        node.bridge = types.SimpleNamespace(imgmsg_to_cv2=lambda msg, encoding: msg.frame)
        self.frame = np.full((120, 200, 3), (20, 50, 220), dtype=np.uint8)
        self.depth = np.full((120, 200), 2000, dtype=np.uint16)
        self.node = node

    def run_callback(self):
        header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=10, nanosec=123000000),
                                       frame_id="head_camera_color_optical_frame")
        rgb = types.SimpleNamespace(frame=self.frame, header=header)
        depth = types.SimpleNamespace(frame=self.depth, encoding="16UC1")
        with patch.object(self.classifier, "_predict", return_value=np.array(
                [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=np.float32)):
            self.node.callback(rgb, depth)
        return self.node.results_pub.messages[-1]

    def test_rgb_labels_preserve_original_ids_face_inputs_xyz_and_json_schema(self):
        result = self.run_callback()
        self.assertEqual({"header", "fps", "realtime_fps", "tracking", "camera_coordinate_frame",
                          "camera_coordinate_convention", "left_base_transform_validated", "num_persons", "persons"},
                         set(result))
        self.assertEqual("ByteTrack", result["tracking"])
        self.assertEqual(10.123, result["header"]["stamp"])
        self.assertEqual([41, 9], [p["person_id"] for p in result["persons"]])
        first, second = result["persons"]
        self.assertEqual(("sitting", "坐下", 1.0, "Alice"),
                         (first["pose"], first["pose_cn"], first["conf"], first["name"]))
        self.assertEqual("lying", second["pose"])
        self.assertEqual({"bbox", "person_id", "name", "name_known", "face_score", "pose", "pose_cn", "conf",
                          "features", "distance_m", "camera_xyz_m", "left_base_xyz_m", "left_base_transform_validated"},
                         set(first))
        np.testing.assert_allclose(first["camera_xyz_m"], [-1., -.1, 2.])
        np.testing.assert_allclose(first["left_base_xyz_m"], [0., 1.9, 5.])
        self.assertFalse(first["left_base_transform_validated"])
        self.assertEqual({"track_id", "history_size", "warmup", "stale_feature", "stale_count"}, set(first["features"]))
        sequence, frame, detections = self.face_requests[0]
        self.assertEqual(1, sequence)
        self.assertIs(self.frame, frame)
        np.testing.assert_array_equal(detections[0]["kpts_2d"][:, :2], self.points[0])
        self.assertEqual((17, 4), detections[0]["kpts_3d"].shape)
        self.assertEqual([41, 9], [d["track_id"] for d in detections])
        self.assertTrue(self.tracker_calls[0]["persist"])
        self.assertEqual(self.node.tracker_config, self.tracker_calls[0]["tracker"])
        identity = self.node.identity_locations_pub.messages[0]
        self.assertEqual({"person_id", "name", "name_known", "face_score", "camera_xyz_m",
                          "left_base_xyz_m", "left_base_transform_validated"}, set(identity["persons"][0]))

    def test_missing_depth_does_not_block_rgb_posture_or_invent_xyz(self):
        self.depth[:] = 0
        result = self.run_callback()
        self.assertEqual(["sitting", "lying"], [p["pose"] for p in result["persons"]])
        for person in result["persons"]:
            self.assertNotIn("camera_xyz_m", person)
            self.assertNotIn("distance_m", person)

    def test_empty_detection_does_not_publish_old_people_or_call_face_worker(self):
        self.run_callback()
        self.empty = True
        result = self.node._detect_and_classify(self.frame, self.depth)
        self.assertEqual([], result)
        self.assertEqual(1, len(self.face_requests))


if __name__ == "__main__":
    unittest.main()
