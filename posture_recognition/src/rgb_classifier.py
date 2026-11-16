"""POLAR ResNet18 classification for the existing tracked RGB-D detections."""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


POSE_CLASSES = ["standing", "sitting", "squatting", "lying", "bending"]
# Keep the labels published by the original ROS node, including 坐下.
CN_LABELS = {"standing": "站立", "sitting": "坐下", "squatting": "蹲",
             "lying": "躺", "bending": "弯腰", "unknown": "未知"}


def crop_person_rgb(frame, bbox, margin=0.05):
    """Crop in original image coordinates; return None for an invalid box."""
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    height, width = frame.shape[:2]
    left = max(0, math.floor(x1 - (x2 - x1) * margin))
    top = max(0, math.floor(y1 - (y2 - y1) * margin))
    right = min(width, math.ceil(x2 + (x2 - x1) * margin))
    bottom = min(height, math.ceil(y2 + (y2 - y1) * margin))
    if right <= left or bottom <= top:
        return None
    return Image.fromarray(np.ascontiguousarray(frame[top:bottom, left:right, ::-1]))


class ResNetPoseClassifier:
    """Classify RGB crops and smooth probabilities independently per ByteTrack ID.

    The result dictionaries retain the TCN adapter's public keys. Depth and
    skeleton features are used only for optional diagnostics, never for logits
    or a geometry-based label override.
    """

    def __init__(self, weights_path, device="auto", batch_size=16,
                 confidence_threshold=0.0, crop_margin=0.05,
                 smoothing_alpha=0.6, min_history=1, max_track_age=15,
                 cpu_threads=4):
        self.batch_size = int(batch_size)
        self.confidence_threshold = float(confidence_threshold)
        self.crop_margin = float(crop_margin)
        self.alpha = float(smoothing_alpha)
        self.min_history = int(min_history)
        self.max_track_age = int(max_track_age)
        if min(self.batch_size, self.min_history, self.max_track_age, int(cpu_threads)) < 1:
            raise ValueError("Batch size, history, track age and CPU threads must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0 <= self.crop_margin <= 0.5 or not 0 < self.alpha <= 1:
            raise ValueError("Invalid crop_margin or smoothing_alpha")

        value = str(device)
        if value == "auto":
            value = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif value.isdigit():
            value = f"cuda:{value}"
        self.device = torch.device(value)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device: cpu explicitly")
        torch.set_num_threads(int(cpu_threads))

        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"POLAR ResNet18 weights not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("architecture") != "resnet18" or
                checkpoint.get("classes") != POSE_CLASSES or
                checkpoint.get("input_size") != 224):
            raise ValueError("Expected the 224x224 POLAR five-class ResNet18 checkpoint")
        self.pose_classes = list(POSE_CLASSES)
        self.input_size = int(checkpoint["input_size"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.padding = tuple(checkpoint["padding_rgb"])
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(POSE_CLASSES))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.reset()

    @classmethod
    def from_config(cls, config, weights_path):
        return cls(
            weights_path, device=config.get("device", "auto"),
            batch_size=config.get("batch_size", 16),
            confidence_threshold=config.get("confidence_threshold", 0.0),
            crop_margin=config.get("crop_margin", 0.05),
            smoothing_alpha=config.get("smoothing_alpha", 0.6),
            min_history=config.get("min_history", 1),
            max_track_age=config.get("max_track_age", 15),
            cpu_threads=config.get("cpu_threads", 4))

    def reset(self):
        self.frame_index = 0
        self.tracks = {}

    def preprocess(self, image):
        """Match POLAR training: RGB, aspect-preserving padding, no center crop."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(self.input_size / image.width, self.input_size / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.input_size, self.input_size), self.padding)
        canvas.paste(image, ((self.input_size - width) // 2,
                            (self.input_size - height) // 2))
        return self.transform(canvas)

    def _predict(self, crops):
        rows = []
        with torch.inference_mode():
            for offset in range(0, len(crops), self.batch_size):
                batch = torch.stack([self.preprocess(crop)
                                     for crop in crops[offset:offset + self.batch_size]])
                probabilities = self.model(batch.to(self.device)).float().softmax(dim=1)
                rows.append(probabilities.cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, 5), dtype=np.float32)

    @staticmethod
    def _geometry_details(detection, down_axis):
        # Preserve optional diagnostic fields without making RGB classification
        # depend on valid depth, a TCN checkpoint or skeleton feature history.
        points = detection.get("kpts_3d")
        if points is None:
            return {}
        points = np.asarray(points, dtype=float)
        if points.shape != (17, 4) or not np.isfinite(points).all():
            return {}
        from pose_features import features_from_coco
        axis = np.array([0.0, 1.0, 0.0]) if down_axis is None else down_axis
        feature = features_from_coco(points, axis)
        if feature is None:
            return {}
        geometry = feature[-10:]
        return {
            "knee_angle": round(float(geometry[:2].mean()) * 180.0, 1),
            "hip_angle": round(float(geometry[2:4].mean()) * 180.0, 1),
            "torso_angle": round(float(geometry[4]) * 90.0, 1),
            "height_ratio": round(float(geometry[5]) * 5.0, 2),
            "width_ratio": round(float(geometry[6]) * 3.0, 2),
            "valid_ratio": round(float(geometry[8]), 2),
        }

    def classify_batch(self, detections, frame, down_axis=None, debug=False):
        """Return one prediction per detection, in the same order, even for bad crops."""
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0):
            raise ValueError("Expected a nonempty HxWx3 uint8 BGR frame")
        track_ids = [int(item["track_id"]) for item in detections]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("Each detection must have a distinct ByteTrack ID")
        self.frame_index += 1
        for track_id in list(self.tracks):
            if self.frame_index - self.tracks[track_id]["last_seen"] > self.max_track_age:
                del self.tracks[track_id]

        crops, indices = [], []
        for index, detection in enumerate(detections):
            crop = crop_person_rgb(frame, detection["bbox"], self.crop_margin)
            if crop is not None:
                indices.append(index)
                crops.append(crop)
        probabilities = np.asarray(self._predict(crops), dtype=np.float32)
        if (probabilities.shape != (len(crops), 5) or
                not np.isfinite(probabilities).all() or (probabilities < 0).any() or
                (probabilities.sum(axis=1) <= 0).any()):
            raise ValueError("Classifier returned invalid probabilities")
        if len(crops):
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        rows = dict(zip(indices, probabilities))

        results = []
        for index, (detection, track_id) in enumerate(zip(detections, track_ids)):
            state = self.tracks.setdefault(track_id, {
                "probabilities": None, "history_size": 0, "stale_count": 0})
            state["last_seen"] = self.frame_index
            raw = rows.get(index)
            label_id, label_name, score = 5, "unknown", 0.0
            if raw is None:
                # Do not emit a cached posture when the current crop is invalid.
                state["probabilities"] = None
                state["history_size"] = 0
                state["stale_count"] += 1
            else:
                old = state["probabilities"]
                smoothed = raw if old is None else self.alpha * raw + (1 - self.alpha) * old
                state["probabilities"] = smoothed / smoothed.sum()
                state["history_size"] += 1
                state["stale_count"] = 0
                top_index = int(state["probabilities"].argmax())
                score = float(state["probabilities"][top_index])
                if (state["history_size"] >= self.min_history and
                        score >= self.confidence_threshold):
                    label_id, label_name = top_index, self.pose_classes[top_index]
            details = {
                "track_id": track_id,
                "history_size": state["history_size"],
                "warmup": state["history_size"] < self.min_history,
                "stale_feature": raw is None,
                "stale_count": state["stale_count"],
            }
            if debug:
                details.update(self._geometry_details(detection, down_axis))
            results.append({"label_id": label_id, "label_name": label_name,
                            "score": score, "track_id": track_id, "features": details})
        return results
