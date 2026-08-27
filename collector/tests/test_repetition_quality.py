from __future__ import annotations

from test_support import (
    JOINT_INDEX,
    Config,
    Phase,
    RepetitionDetector,
    check_quality,
    squat_landmark_frame,
    unittest,
)


class TestRepetitionQuality(unittest.TestCase):
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

    def test_default_standing_confirmation_boundary_is_five_frames(self):
        cfg = Config(smoothing_window=1)
        self.assertEqual(cfg.standing_confirm_frames, 5)
        detector = RepetitionDetector(cfg)

        for now in range(4):
            detector.update(squat_landmark_frame(0.2), now=float(now))
        detector.update(squat_landmark_frame(0.7), now=4.0)
        self.assertEqual(detector.phase, Phase.STANDING)

        detector = RepetitionDetector(cfg)
        for now in range(5):
            detector.update(squat_landmark_frame(0.2), now=float(now))
        detector.update(squat_landmark_frame(0.7), now=5.0)
        self.assertEqual(detector.phase, Phase.DESCENDING)

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

    def test_quality_gate_boundaries_match_the_default_contract(self):
        cfg = Config()
        self.assertEqual(cfg.min_rep_duration_sec, 0.4)
        self.assertEqual(cfg.max_saved_rep_duration_sec, 5.0)
        self.assertEqual(cfg.max_missing_frame_ratio, 0.15)
        self.assertEqual(cfg.min_keypoint_visibility, 0.4)
        self.assertEqual(cfg.min_avg_visibility, 0.6)
        frames = [squat_landmark_frame(0.2) for _ in range(3)]

        accepted_at_minimum = check_quality(frames, 0.4, 0, 20, cfg)
        accepted_at_maximum = check_quality(frames, 5.0, 3, 20, cfg)
        too_short = check_quality(frames, 0.39, 0, 20, cfg)
        too_long = check_quality(frames, 5.01, 0, 20, cfg)
        too_many_missing = check_quality(frames, 1.0, 4, 20, cfg)

        self.assertTrue(accepted_at_minimum.passed)
        self.assertTrue(accepted_at_maximum.passed)
        self.assertIn("too short", too_short.reason)
        self.assertIn("too long", too_long.reason)
        self.assertIn("too many missing", too_many_missing.reason)

    def test_quality_visibility_gates_are_independent(self):
        cfg = Config()
        critical_visibility_failure = squat_landmark_frame(0.2)
        for joint_name in (
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):
            critical_visibility_failure[JOINT_INDEX[joint_name], 3] = 0.39
        low_overall_visibility = squat_landmark_frame(0.2)
        low_overall_visibility[:, 3] = 0.5

        critical_report = check_quality(
            [critical_visibility_failure] * 3,
            duration_sec=1.0,
            missing_frame_count=0,
            total_expected_frames=3,
            cfg=cfg,
        )
        overall_report = check_quality(
            [low_overall_visibility] * 3,
            duration_sec=1.0,
            missing_frame_count=0,
            total_expected_frames=3,
            cfg=cfg,
        )

        self.assertIn("critical keypoint visibility", critical_report.reason)
        self.assertIn("overall visibility", overall_report.reason)
