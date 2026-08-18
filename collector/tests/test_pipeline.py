from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COLLECTOR_DIR.parent
REHAB_DIR = COLLECTOR_DIR / "RehabExerAssess-main"
sys.path.insert(0, str(COLLECTOR_DIR))

from canonical import (  # noqa: E402
    Label,
    COORD_Y,
    COORD_Z,
    JOINT_INDEX,
    SEQUENCE_LENGTH,
    canonical_shape,
    parse_uiprmd_filename,
)
from config import Config  # noqa: E402
from dataset_split import discover_uiprmd_txt, subject_safe_split, split_stats  # noqa: E402
from form_analysis import analyze_form, top_detected_issues  # noqa: E402
from form_inference import FormClassifierInference  # noqa: E402
from form_model import FormClassifier  # noqa: E402
from preprocessing import assert_canonical_orientation  # noqa: E402
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


def write_raw_skeleton_fixture(path: Path) -> Path:
    skeleton = synthetic_uiprmd_skeleton(frame_count=3)
    lines = [str(skeleton.shape[0]), "1", "0 0 0 0 0 0 0 0 0", "22"]

    for frame_idx, frame in enumerate(skeleton):
        if frame_idx > 0:
            lines.extend(["1", "0 0 0 0 0 0 0 0 0", "22"])
        for xyz in frame:
            lines.append(
                "0 0 0 0 0 0 "
                f"{float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")
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
    def test_raw_skeleton_parser_shape(self):
        spec = importlib.util.spec_from_file_location(
            "convert_uiprmd", REHAB_DIR / "convert_uiprmd.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_raw_skeleton_fixture(Path(tmp) / "A01S01E02C01.skeleton")
            raw = module.parse_skeleton(path)
            self.assertEqual(raw.shape[1:], (22, 3))
            self.assertGreater(raw.shape[0], 1)

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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_name": {0: "incorrect", 1: "correct"},
                },
                path,
            )
            inference = FormClassifierInference(path, device="cpu")
            sample_path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            sample = process_file(sample_path, Config())
            result = inference.predict(sample)
            self.assertIn(result["label"], {"incorrect", "correct"})
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)

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
