"""RGB preprocessing, real POLAR weights and per-ID state regression tests."""

import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rgb_classifier import CN_LABELS, POSE_CLASSES, ResNetPoseClassifier, crop_person_rgb


class RGBClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = ROOT / "models" / "resnet18_polar5.pth"
        cls.classifier = ResNetPoseClassifier(cls.weights, device="cpu", batch_size=2)

    def setUp(self):
        self.classifier.reset()
        self.classifier.alpha = 0.6
        self.classifier.min_history = 1
        self.classifier.max_track_age = 15
        self.classifier.confidence_threshold = 0.0
        self.frame = np.full((100, 160, 3), (10, 20, 230), dtype=np.uint8)

    @staticmethod
    def person(track_id, bbox=None):
        return {"track_id": track_id, "bbox": [10, 10, 60, 90] if bbox is None else bbox}

    def classify(self, detections, rows):
        with patch.object(self.classifier, "_predict", return_value=np.asarray(rows).reshape(-1, 5)):
            return self.classifier.classify_batch(detections, frame=self.frame)

    def test_default_config_uses_verified_polar_weights_and_independent_threshold(self):
        config = yaml.safe_load((ROOT / "config" / "pose_3d_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual("../models/resnet18_polar5.pth", config["classifier"]["weights_path"])
        self.assertEqual(0.0, config["classifier"]["confidence_threshold"])
        self.assertNotIn("deep_squat_guard", config["classifier"])
        self.assertEqual("坐下", CN_LABELS["sitting"])
        self.assertEqual("3ef34071e0991e9c5f8a02bb7a6765ce028c49a8a6e6bf634f3b20056207b46f",
                         hashlib.sha256(self.weights.read_bytes()).hexdigest())

    def test_crop_clips_border_expands_five_percent_and_converts_bgr_to_rgb(self):
        crop = crop_person_rgb(self.frame, [-2, 5, 38, 85])
        self.assertEqual((40, 88), crop.size)
        self.assertEqual((230, 20, 10), crop.getpixel((0, 0)))
        self.assertIsNone(crop_person_rgb(self.frame, [200, 10, 220, 90]))
        self.assertIsNone(crop_person_rgb(self.frame, [10, 10, 10, 90]))
        self.assertIsNone(crop_person_rgb(self.frame, [0, 0, np.nan, 90]))

    def test_preprocessing_and_chunked_inference_match_training_reference(self):
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        rng = np.random.default_rng(17)
        crops = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
                 for h, w in [(301, 97), (113, 259), (224, 224)]]
        reference = []
        for crop in crops:
            # Independent PIL operation: LANCZOS contain + centered RGB padding.
            resized = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
            padded = Image.new("RGB", (224, 224), tuple(checkpoint["padding_rgb"]))
            padded.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
            tensor = transforms.ToTensor()(padded)
            tensor = transforms.Normalize(checkpoint["mean"], checkpoint["std"])(tensor)
            torch.testing.assert_close(self.classifier.preprocess(crop), tensor, rtol=0, atol=0)
            reference.append(tensor)
        with torch.inference_mode():
            expected = self.classifier.model(torch.stack(reference)).softmax(1).numpy()
        actual = self.classifier._predict(crops)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual((3, 5), actual.shape)

    def test_real_classifier_works_without_depth_or_tcn_history(self):
        person = self.person(42)
        person["kpts_3d"] = np.zeros((17, 4))
        result = self.classifier.classify_batch([person], self.frame, debug=True)[0]
        self.assertIn(result["label_name"], POSE_CLASSES)
        self.assertEqual(42, result["track_id"])
        self.assertFalse(result["features"]["warmup"])
        self.assertFalse(result["features"]["stale_feature"])

    def test_reordered_people_keep_their_own_probability_history(self):
        self.classify([self.person(11), self.person(22)],
                      [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        results = self.classify([self.person(22), self.person(11)],
                                [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
        self.assertEqual([22, 11], [item["track_id"] for item in results])
        np.testing.assert_allclose(self.classifier.tracks[22]["probabilities"], [0, .4, .6, 0, 0])
        np.testing.assert_allclose(self.classifier.tracks[11]["probabilities"], [.4, .6, 0, 0, 0])

    def test_invalid_crop_keeps_alignment_and_never_reuses_cached_label(self):
        self.classify([self.person(11)], [[1, 0, 0, 0, 0]])
        results = self.classify([self.person(11, [200, 0, 250, 80]), self.person(22)],
                                [[0, 0, 0, 1, 0]])
        self.assertEqual(["unknown", "lying"], [item["label_name"] for item in results])
        self.assertEqual([11, 22], [item["track_id"] for item in results])
        self.assertEqual(0.0, results[0]["score"])
        self.assertTrue(results[0]["features"]["stale_feature"])
        recovered = self.classify([self.person(11)], [[0, 1, 0, 0, 0]])[0]
        self.assertEqual(1.0, recovered["score"])
        self.assertEqual(1, recovered["features"]["history_size"])

    def test_empty_frames_expire_state_before_track_id_is_reused(self):
        self.classifier.max_track_age = 1
        self.classify([self.person(8)], [[1, 0, 0, 0, 0]])
        self.assertEqual([], self.classify([], []))
        self.classify([], [])
        self.assertNotIn(8, self.classifier.tracks)
        result = self.classify([self.person(8)], [[0, 0, 0, 0, 1]])[0]
        self.assertEqual(("bending", 1.0), (result["label_name"], result["score"]))

    def test_threshold_and_optional_history_gate_keep_unknown_contract(self):
        self.classifier.min_history = 2
        self.classifier.confidence_threshold = .8
        first = self.classify([self.person(8)], [[.9, .1, 0, 0, 0]])[0]
        self.assertEqual((5, "unknown"), (first["label_id"], first["label_name"]))
        second = self.classify([self.person(8)], [[.9, .1, 0, 0, 0]])[0]
        self.assertEqual("standing", second["label_name"])
        third = self.classify([self.person(8)], [[.2, .2, .2, .2, .2]])[0]
        self.assertEqual("unknown", third["label_name"])
        self.assertFalse(third["features"]["warmup"])

    def test_duplicate_ids_and_nonfinite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.classify([self.person(8), self.person(8)], [])
        with self.assertRaises(ValueError):
            self.classify([self.person(8)], [[np.nan, 0, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
