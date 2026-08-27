from __future__ import annotations

from test_support import (
    Config,
    FakeClassifier,
    FakePoseEstimator,
    LiveRepProcessor,
    Path,
    Phase,
    RepetitionDetector,
    StorageManager,
    canonical_shape,
    completed_squat,
    json,
    np,
    squat_landmark_frame,
    tempfile,
    unittest,
)


class TestLiveFlow(unittest.TestCase):
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

    def test_incomplete_rep_does_not_save_or_run_feedback(self):
        cfg = Config(dataset_root="unused", standing_confirm_frames=2, smoothing_window=1)
        detector = RepetitionDetector(cfg)
        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.dataset_root = str(Path(tmp) / "samples")
            storage = StorageManager(cfg)
            processor = LiveRepProcessor(cfg, storage, classifier)
            completed = None
            for now, knee_x in ((0.0, 0.2), (0.1, 0.2), (0.2, 0.7), (0.3, 1.2)):
                completed = detector.update(squat_landmark_frame(knee_x), now=now) or completed
            storage.close()

            self.assertIsNone(completed)
            self.assertEqual(classifier.samples, [])
            self.assertEqual(list(Path(cfg.dataset_root).rglob("*.npy")), [])
            self.assertIsNotNone(processor)

    def test_tracking_timeout_does_not_create_a_live_result(self):
        cfg = Config(
            dataset_root="unused",
            standing_confirm_frames=1,
            smoothing_window=1,
            max_rep_tracking_duration_sec=0.2,
        )
        detector = RepetitionDetector(cfg)
        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.dataset_root = str(Path(tmp) / "samples")
            storage = StorageManager(cfg)
            detector.update(squat_landmark_frame(0.2), now=0.0)
            detector.update(squat_landmark_frame(0.7), now=0.1)
            completed = detector.update(squat_landmark_frame(0.7), now=0.4)
            storage.close()

            self.assertIsNone(completed)
            self.assertEqual(detector.phase, Phase.STANDING)
            self.assertEqual(classifier.samples, [])
            self.assertEqual(list(Path(cfg.dataset_root).rglob("*.npy")), [])

    def test_quality_rejection_stops_preprocessing_feedback_and_storage(self):
        cfg = Config(
            dataset_root="unused",
            standing_confirm_frames=2,
            smoothing_window=1,
            min_rep_duration_sec=0.1,
            min_keypoint_visibility=0.5,
        )
        completed = completed_squat(self, RepetitionDetector(cfg))
        for frame in completed.frames:
            frame[:, 3] = 0.0
        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.dataset_root = str(Path(tmp) / "samples")
            storage = StorageManager(cfg)
            result = LiveRepProcessor(cfg, storage, classifier).process(
                completed,
                missing_frame_count=0,
                total_expected_frames=7,
                class_name="correct",
                class_id=0,
            )
            storage.close()

            self.assertFalse(result.saved)
            self.assertIn("visibility too low", result.rejection_reason)
            self.assertIsNone(result.sample)
            self.assertEqual(classifier.samples, [])
            self.assertEqual(list(Path(cfg.dataset_root).rglob("*.npy")), [])
