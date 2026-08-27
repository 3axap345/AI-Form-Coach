from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COLLECTOR_DIR))

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
from preprocessing import assert_canonical_orientation  # noqa: E402
from quality import check_quality  # noqa: E402
from repetition import Phase, RepetitionDetector  # noqa: E402
from storage import StorageManager  # noqa: E402
from uiprmd_adapter import (  # noqa: E402
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


class PipelineTests(unittest.TestCase):
    def test_filename_and_label_mapping(self):
        correct = parse_uiprmd_filename("A01S01E02C01.txt")
        incorrect = parse_uiprmd_filename("A01S01E02C02.txt")
        self.assertEqual(correct.activity, "01")
        self.assertEqual(correct.subject, "01")
        self.assertEqual(correct.episode, "02")
        self.assertEqual(correct.label, Label.CORRECT.value)
        self.assertEqual(incorrect.label, Label.INCORRECT.value)

    def test_uiprmd_mapping_and_y_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            raw = load_uiprmd_skeleton_txt(path)
            frames = load_uiprmd_skeleton(path)
            self.assertEqual(raw.shape[1:], (22, 3))
            self.assertEqual(frames.shape[1:], (12, 4))

            raw_left_shoulder_y = raw[0, 6, 1]
            canonical_left_shoulder_y = frames[0, JOINT_INDEX["left_shoulder"], COORD_Y]
            self.assertAlmostEqual(
                canonical_left_shoulder_y,
                -raw_left_shoulder_y,
                places=4,
            )

    def test_preprocessing_shape_orientation_and_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            sample = process_file(path, Config())
            self.assertEqual(sample.shape, canonical_shape())
            self.assertEqual(sample.dtype, np.float32)
            assert_canonical_orientation(sample)
            self.assertLess(abs(float(sample[:, :, COORD_Z].mean())), 2.0)
            self.assertLess(float(np.max(np.abs(sample[:, :, COORD_Z]))), 3.0)
            self.assertTrue(np.all(sample[:, :, 3] == 1.0))

    def test_upside_down_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            sample = process_file(path, Config())
            upside_down = sample.copy()
            upside_down[:, :, COORD_Y] *= -1.0
            with self.assertRaises(ValueError):
                assert_canonical_orientation(upside_down)

    def test_subject_safe_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "UI-PRMD"
            write_split_fixture(root)
            records = discover_uiprmd_txt(root)
            split = subject_safe_split(records, test_subjects=("08",))
            train_subjects = {row["subject"] for row in split["train"]}
            test_subjects = {row["subject"] for row in split["test"]}
            self.assertFalse(train_subjects.intersection(test_subjects))
            stats = split_stats(split)
            self.assertEqual(stats["train"]["total"], 8)
            self.assertEqual(stats["test"]["total"], 4)
            self.assertEqual(stats["train"]["labels"], {"incorrect": 4, "correct": 4})
            self.assertEqual(stats["test"]["labels"], {"incorrect": 2, "correct": 2})

    def test_model_input_shape(self):
        model = FormClassifier()
        x = torch.zeros((2, SEQUENCE_LENGTH, 12, 4), dtype=torch.float32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_inference_output(self):
        model = FormClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            inference = FormClassifierInference(
                path,
                device="cpu",
                expected_sha256=sha256_file(path),
            )
            sample_path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            sample = process_file(sample_path, Config())
            result = inference.predict(sample)
            self.assertIn(result["label"], {"incorrect", "correct"})
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)

    def test_inference_rejects_unexpected_state_dict(self):
        model = FormClassifier()
        state_dict = model.state_dict()
        state_dict.pop("head.1.bias")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(state_dict, path)
            with self.assertRaisesRegex(ModelLoadError, "keys do not match"):
                FormClassifierInference(path, device="cpu")

    def test_inference_rejects_wrong_sha256(self):
        model = FormClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ModelLoadError, "SHA-256 mismatch"):
                FormClassifierInference(path, device="cpu", expected_sha256="0" * 64)

    def test_standing_confirmation_is_required_before_rep_tracking(self):
        cfg = Config(
            standing_confirm_frames=3,
            smoothing_window=1,
            min_rep_duration_sec=0.1,
        )
        detector = RepetitionDetector(cfg)

        self.assertIsNone(detector.update(squat_landmark_frame(1.2), now=0.0))
        self.assertEqual(detector.phase, Phase.STANDING)
        for now in (0.1, 0.2, 0.3):
            self.assertIsNone(detector.update(squat_landmark_frame(0.2), now=now))

        completed = None
        for now, knee_x in ((0.4, 0.7), (0.5, 1.2), (0.6, 1.2), (0.7, 0.7), (0.8, 0.2)):
            completed = detector.update(squat_landmark_frame(knee_x), now=now) or completed

        self.assertIsNotNone(completed)
        self.assertEqual(detector.phase, Phase.STANDING)

    def test_tracking_timeout_and_storage_duration_are_independent(self):
        cfg = Config(
            standing_confirm_frames=1,
            smoothing_window=1,
            max_rep_tracking_duration_sec=0.2,
            max_saved_rep_duration_sec=10.0,
        )
        detector = RepetitionDetector(cfg)
        detector.update(squat_landmark_frame(0.2), now=0.0)
        detector.update(squat_landmark_frame(0.7), now=0.1)
        self.assertEqual(detector.phase, Phase.DESCENDING)
        self.assertIsNone(detector.update(squat_landmark_frame(0.7), now=0.4))
        self.assertEqual(detector.phase, Phase.STANDING)

        report = check_quality(
            [squat_landmark_frame(0.2)] * 3,
            duration_sec=2.0,
            missing_frame_count=0,
            total_expected_frames=3,
            cfg=cfg,
        )
        self.assertTrue(report.passed)

    def test_live_flow_happy_path_saves_one_completed_repetition(self):
        cfg = Config(
            dataset_root="unused",
            standing_confirm_frames=2,
            smoothing_window=1,
            min_rep_duration_sec=0.1,
            min_avg_visibility=0.5,
            min_keypoint_visibility=0.5,
        )
        pose_estimator = FakePoseEstimator()
        detector = RepetitionDetector(cfg)
        classifier = FakeClassifier()

        with tempfile.TemporaryDirectory() as tmp:
            cfg.dataset_root = str(Path(tmp) / "samples")
            storage = StorageManager(cfg)
            processor = LiveRepProcessor(cfg, storage, classifier)
            completed = None
            sequence = (
                (0.0, 0.2),
                (0.1, 0.2),
                (0.2, 0.7),
                (0.3, 1.2),
                (0.4, 1.2),
                (0.5, 0.7),
                (0.6, 0.2),
            )
            for now, knee_x in sequence:
                pose_result = pose_estimator.process(squat_landmark_frame(knee_x))
                completed = detector.update(pose_result.landmarks, now=now) or completed

            self.assertIsNotNone(completed)
            result = processor.process(
                completed,
                missing_frame_count=0,
                total_expected_frames=len(sequence),
                class_name="correct",
                class_id=0,
            )
            storage.close()

            self.assertTrue(result.saved)
            self.assertIsNone(result.rejection_reason)
            self.assertEqual(result.prediction["label"], "correct")
            self.assertEqual(len(classifier.samples), 1)
            self.assertEqual(classifier.samples[0].shape, canonical_shape())
            saved_npy = Path(cfg.dataset_root) / "correct" / "sample_000001.npy"
            saved_json = saved_npy.with_suffix(".json")
            self.assertTrue(saved_npy.exists())
            self.assertEqual(np.load(saved_npy).shape, canonical_shape())
            metadata = json.loads(saved_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["class_label"], "correct")

    def test_form_analysis_shallow_depth(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_hip"], :2] = (0.30, 0.55)
        sample[bottom, JOINT_INDEX["right_hip"], :2] = (-0.30, 0.55)
        sample[bottom, JOINT_INDEX["left_knee"], :2] = (0.34, 1.1)
        sample[bottom, JOINT_INDEX["right_knee"], :2] = (-0.34, 1.1)
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["shallow_depth"]["detected"])
        self.assertIn("Squat depth", analysis["shallow_depth"]["message"])

    def test_form_analysis_excessive_forward_lean(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_shoulder"], 0] += 0.9
        sample[bottom, JOINT_INDEX["right_shoulder"], 0] += 0.9
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["excessive_forward_lean"]["detected"])

    def test_form_analysis_knee_valgus(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_knee"], 0] = 0.08
        sample[bottom, JOINT_INDEX["right_knee"], 0] = -0.08
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["knee_valgus"]["detected"])

    def test_form_analysis_heel_instability(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_heel"], :2] += (0.25, -0.15)
        sample[bottom, JOINT_INDEX["right_heel"], :2] += (-0.25, -0.15)
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["heel_instability"]["detected"])

    def test_top_detected_issues_limits_and_sorts(self):
        analysis = {
            "knee_valgus": {"detected": True, "score": 0.4, "message": "a"},
            "shallow_depth": {"detected": True, "score": 0.9, "message": "b"},
            "heel_instability": {"detected": True, "score": 0.6, "message": "c"},
            "_metrics": {},
        }
        issues = top_detected_issues(analysis, limit=2)
        self.assertEqual([issue["message"] for issue in issues], ["b", "c"])


if __name__ == "__main__":
    unittest.main()
