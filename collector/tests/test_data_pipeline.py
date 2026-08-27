from __future__ import annotations

from test_support import (
    COORD_Y,
    COORD_Z,
    JOINT_INDEX,
    TXT_SAMPLE_NAME,
    Config,
    Label,
    Path,
    assert_canonical_orientation,
    canonical_shape,
    convert_dataset,
    discover_uiprmd_txt,
    load_uiprmd_skeleton,
    load_uiprmd_skeleton_txt,
    np,
    parse_uiprmd_filename,
    process_file,
    split_stats,
    subject_safe_split,
    tempfile,
    unittest,
    write_split_fixture,
    write_uiprmd_txt_fixture,
)


class TestDataPipeline(unittest.TestCase):
    def test_default_config_is_valid(self):
        self.assertIsInstance(Config(), Config)

    def test_config_rejects_invalid_ranges_and_sizes(self):
        invalid_configs = (
            ({"target_fps": 0}, "target_fps must be positive"),
            ({"standing_confirm_frames": 0}, "standing_confirm_frames must be positive"),
            ({"min_avg_visibility": 1.1}, "min_avg_visibility must be between 0 and 1"),
            (
                {"min_bbox_area_ratio": 0.9, "max_bbox_area_ratio": 0.1},
                "min_bbox_area_ratio must not exceed max_bbox_area_ratio",
            ),
            ({"min_rep_duration_sec": -0.1}, "min_rep_duration_sec must be non-negative"),
            ({"reconnect_delay_sec": -0.1}, "reconnect_delay_sec must be non-negative"),
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    Config(**kwargs)

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

    def test_uiprmd_converter_uses_explicit_external_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "converted"
            source = source_root / "Correct" / "Kinect" / "Skeletons" / TXT_SAMPLE_NAME
            write_uiprmd_txt_fixture(source)
            output_root = Path(tmp) / "output"

            counts = convert_dataset(source_root, output_root, Config())

            self.assertEqual(counts, {"correct": 1, "incorrect": 0})
            converted_sample = output_root / "correct" / f"{Path(TXT_SAMPLE_NAME).stem}.npy"
            self.assertTrue(converted_sample.exists())

    def test_uiprmd_converter_rejects_missing_external_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_root = Path(tmp) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "Provide --source-root"):
                convert_dataset(missing_root, Path(tmp) / "output", Config())

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
