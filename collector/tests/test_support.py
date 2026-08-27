from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COLLECTOR_DIR))

from camera import Camera, CameraError  # noqa: E402
from canonical import (  # noqa: E402
    COORD_Y,
    COORD_Z,
    JOINT_INDEX,
    SEQUENCE_LENGTH,
    Label,
    canonical_shape,
    parse_uiprmd_filename,
)
from config import Config  # noqa: E402
from dataset_split import discover_uiprmd_txt, split_stats, subject_safe_split  # noqa: E402
from form_analysis import analyze_form, top_detected_issues  # noqa: E402
from form_inference import FormClassifierInference, ModelLoadError, sha256_file  # noqa: E402
from form_model import FormClassifier  # noqa: E402
from live_flow import LiveRepProcessor  # noqa: E402
from pose import PoseDataError, PoseEstimator  # noqa: E402
from preprocessing import assert_canonical_orientation  # noqa: E402
from quality import check_quality  # noqa: E402
from repetition import Phase, RepetitionDetector  # noqa: E402
from storage import StorageManager  # noqa: E402
from ui import HudState, draw_hud  # noqa: E402
from uiprmd_adapter import (  # noqa: E402
    convert_dataset,
    load_uiprmd_skeleton,
    load_uiprmd_skeleton_txt,
    process_file,
)

TXT_SAMPLE_NAME = "A01S01E02C01.txt"


def synthetic_squat_sample() -> np.ndarray:
    sample = np.zeros(canonical_shape(), dtype=np.float32)
    sample[:, :, 3] = 1.0

    coords = {
        "left_shoulder": (0.25, -1.0, 0.0),
        "right_shoulder": (-0.25, -1.0, 0.0),
        "left_hip": (0.25, 0.0, 0.0),
        "right_hip": (-0.25, 0.0, 0.0),
        "left_knee": (0.35, 1.0, 0.0),
        "right_knee": (-0.35, 1.0, 0.0),
        "left_ankle": (0.42, 2.0, 0.0),
        "right_ankle": (-0.42, 2.0, 0.0),
        "left_heel": (0.45, 2.18, 0.0),
        "right_heel": (-0.45, 2.18, 0.0),
        "left_foot_index": (0.45, 2.18, 0.0),
        "right_foot_index": (-0.45, 2.18, 0.0),
    }
    for name, xyz in coords.items():
        sample[:, JOINT_INDEX[name], :3] = xyz

    bottom = slice(25, 36)
    sample[bottom, JOINT_INDEX["left_hip"], 1] = 0.75
    sample[bottom, JOINT_INDEX["right_hip"], 1] = 0.75
    sample[bottom, JOINT_INDEX["left_shoulder"], 1] = -0.25
    sample[bottom, JOINT_INDEX["right_shoulder"], 1] = -0.25
    sample[bottom, JOINT_INDEX["left_knee"], :2] = (0.75, 1.2)
    sample[bottom, JOINT_INDEX["right_knee"], :2] = (-0.75, 1.2)
    return sample


def squat_landmark_frame(knee_x: float) -> np.ndarray:
    """Create a symmetric live-landmark frame; 0.2 is standing, 1.2 is bottom."""
    frame = np.zeros((12, 4), dtype=np.float32)
    frame[:, 3] = 1.0
    for side, names in (
        (1.0, ("left_hip", "left_knee", "left_ankle")),
        (-1.0, ("right_hip", "right_knee", "right_ankle")),
    ):
        hip, knee, ankle = (JOINT_INDEX[name] for name in names)
        frame[hip, :2] = (side * 0.2, 0.0)
        frame[knee, :2] = (side * knee_x, 1.0)
        frame[ankle, :2] = (side * 0.2, 2.0)
    return frame


class FakePoseEstimator:
    def process(self, frame: np.ndarray):
        return type("FakePoseResult", (), {"landmarks": frame, "bbox_area_ratio": 0.25})()


class FakeClassifier:
    def __init__(self):
        self.samples: list[np.ndarray] = []

    def predict(self, sample: np.ndarray) -> dict:
        self.samples.append(sample.copy())
        return {
            "label": "correct",
            "label_id": 1,
            "confidence": 0.95,
            "probabilities": {"incorrect": 0.05, "correct": 0.95},
        }


class FakeCapture:
    def __init__(self, opened: bool, reads: list[tuple[bool, np.ndarray | None]]):
        self.opened = opened
        self.reads = list(reads)
        self.release = MagicMock()
        self.set = MagicMock()

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.reads.pop(0) if self.reads else (False, None)


class FakeMediaPipePose:
    def __init__(self, result: object):
        self.result = result
        self.close = MagicMock()

    def process(self, frame: np.ndarray) -> object:
        return self.result


def synthetic_uiprmd_skeleton(frame_count: int = 8) -> np.ndarray:
    skeleton = np.zeros((frame_count, 22, 3), dtype=np.float32)
    coords = {
        6: (0.30, 140.0, -230.0),
        10: (-0.30, 140.0, -230.0),
        14: (0.30, 90.0, -230.0),
        18: (-0.30, 90.0, -230.0),
        15: (0.38, 50.0, -230.0),
        19: (-0.38, 50.0, -230.0),
        16: (0.44, 10.0, -230.0),
        20: (-0.44, 10.0, -230.0),
        17: (0.48, 5.0, -229.0),
        21: (-0.48, 5.0, -231.0),
    }

    for joint_idx, xyz in coords.items():
        skeleton[:, joint_idx, :] = xyz

    for i in range(frame_count):
        squat_phase = np.sin(np.pi * i / max(frame_count - 1, 1))
        skeleton[i, [14, 18], 1] -= 15.0 * squat_phase
        skeleton[i, [6, 10], 1] -= 15.0 * squat_phase

    return skeleton


def write_uiprmd_txt_fixture(path: Path, frame_count: int = 8) -> Path:
    skeleton = synthetic_uiprmd_skeleton(frame_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, skeleton.reshape(frame_count, -1), fmt="%.6f")
    return path


def write_split_fixture(root: Path) -> None:
    for label_dir, class_code in (("Correct", "01"), ("Incorrect", "02")):
        skeleton_dir = root / label_dir / "Kinect" / "Skeletons"
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        for subject in ("01", "02", "08"):
            for episode in ("01", "02"):
                path = skeleton_dir / f"A01S{subject}E{episode}C{class_code}.txt"
                path.write_text("", encoding="utf-8")


__all__ = [
    "COORD_Y",
    "COORD_Z",
    "JOINT_INDEX",
    "SEQUENCE_LENGTH",
    "Label",
    "canonical_shape",
    "parse_uiprmd_filename",
    "Config",
    "discover_uiprmd_txt",
    "split_stats",
    "subject_safe_split",
    "analyze_form",
    "top_detected_issues",
    "FormClassifierInference",
    "ModelLoadError",
    "sha256_file",
    "FormClassifier",
    "LiveRepProcessor",
    "PoseDataError",
    "PoseEstimator",
    "assert_canonical_orientation",
    "check_quality",
    "Phase",
    "RepetitionDetector",
    "StorageManager",
    "HudState",
    "draw_hud",
    "convert_dataset",
    "load_uiprmd_skeleton",
    "load_uiprmd_skeleton_txt",
    "process_file",
    "TXT_SAMPLE_NAME",
    "Camera",
    "CameraError",
    "np",
    "torch",
    "json",
    "tempfile",
    "unittest",
    "Path",
    "SimpleNamespace",
    "MagicMock",
    "patch",
    "synthetic_squat_sample",
    "squat_landmark_frame",
    "FakePoseEstimator",
    "FakeClassifier",
    "FakeCapture",
    "FakeMediaPipePose",
    "synthetic_uiprmd_skeleton",
    "write_uiprmd_txt_fixture",
    "write_split_fixture",
    "completed_squat",
]


def completed_squat(test_case: unittest.TestCase, detector: RepetitionDetector) -> list[np.ndarray]:
    completed: list[np.ndarray] = []
    for knee_angle in (170, 170, 170, 170, 170, 150, 100, 80, 100, 150, 170, 170, 170, 170, 170):
        candidate = detector.update(squat_landmark_frame(knee_angle))
        if candidate is not None:
            completed.append(candidate)
    test_case.assertEqual(len(completed), 1)
    return completed
