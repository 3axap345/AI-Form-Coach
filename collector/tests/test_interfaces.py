from __future__ import annotations

from test_support import (
    Camera,
    CameraError,
    Config,
    FakeCapture,
    FakeMediaPipePose,
    HudState,
    PoseDataError,
    PoseEstimator,
    SimpleNamespace,
    draw_hud,
    np,
    patch,
    unittest,
)


class TestInterfaces(unittest.TestCase):
    def test_camera_releases_unusable_backends(self):
        cfg = Config(windows_backends=(1, 2))
        captures = [FakeCapture(False, []), FakeCapture(True, [(False, None)])]
        with patch("camera.cv2.VideoCapture", side_effect=captures):
            with self.assertRaises(CameraError):
                Camera(cfg)
        for capture in captures:
            capture.release.assert_called_once()

    def test_camera_failed_read_attempts_reconnect(self):
        probe_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = FakeCapture(True, [(True, probe_frame), (False, None)])
        with patch("camera.cv2.VideoCapture", return_value=capture):
            camera = Camera(Config(windows_backends=(1,)))
        with patch.object(camera, "_reconnect", return_value=(False, None)) as reconnect:
            self.assertEqual(camera.read(), (False, None))
        reconnect.assert_called_once()
        camera.release()
        capture.release.assert_called_once()

    def test_pose_wrapper_handles_missing_pose_and_closes(self):
        estimator = object.__new__(PoseEstimator)
        pose = FakeMediaPipePose(SimpleNamespace(pose_landmarks=None))
        estimator._pose = pose
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch("pose.cv2.cvtColor", return_value=frame):
            self.assertIsNone(estimator.process(frame))
        estimator.close()
        pose.close.assert_called_once()

    def test_pose_wrapper_rejects_incomplete_landmarks(self):
        estimator = object.__new__(PoseEstimator)
        landmark = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
        pose_landmarks = SimpleNamespace(landmark=[landmark])
        estimator._pose = FakeMediaPipePose(SimpleNamespace(pose_landmarks=pose_landmarks))
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch("pose.cv2.cvtColor", return_value=frame):
            with self.assertRaisesRegex(PoseDataError, "missing required landmarks"):
                estimator.process(frame)

    def test_hud_displays_runtime_diagnostics(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HudState(
            class_name="correct",
            samples_saved=4,
            rejected=1,
            knee_angle=128.0,
            form_issues=[{"message": "Knees moving inward"}],
        )
        with patch("ui._draw_panel"), patch("ui._put") as put:
            draw_hud(frame, state)
        texts = [call.args[1] for call in put.call_args_list]
        self.assertIn("Class: correct  Saved: 4  Rejected: 1", texts)
        self.assertIn("Knee angle: 128 deg", texts)
        self.assertIn("Feedback: Knees moving inward", texts)
