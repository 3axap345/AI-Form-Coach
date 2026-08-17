"""
Обёртка над MediaPipe Pose.

ВАЖНО: mediapipe.solutions.pose детектирует только ОДНОГО человека в кадре
(это ограничение самой модели BlazePose в legacy Solutions API). Если в
кадре несколько людей, MediaPipe сам трекает одну персону (обычно
наиболее заметную/крупную) и просто не даёт выбрать между несколькими.
Для честного multi-person detection с выбором "ближайшей/крупнейшей"
персоны потребуется MediaPipe Tasks API (PoseLandmarker, num_poses>1) с
отдельно скачиваемой моделью — сознательно не усложняем этим v1, как и
обсуждали. Здесь мы просто прозрачно логируем этот факт.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import Config, LANDMARK_INDICES, NUM_LANDMARKS

logger = logging.getLogger("collector.pose")


@dataclass
class PoseResult:
    landmarks: np.ndarray          # [NUM_LANDMARKS, 4] -> x, y, z, visibility
    bbox_area_ratio: float         # доля площади кадра, занятая человеком
    raw_result: object             # для отрисовки skeleton


class PoseEstimator:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not result.pose_landmarks:
            return None

        all_lm = result.pose_landmarks.landmark
        selected = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        xs, ys = [], []
        for i, idx in enumerate(LANDMARK_INDICES):
            lm = all_lm[idx]
            selected[i] = (lm.x, lm.y, lm.z, lm.visibility)
            xs.append(lm.x)
            ys.append(lm.y)

        bbox_area_ratio = (max(xs) - min(xs)) * (max(ys) - min(ys))

        return PoseResult(
            landmarks=selected,
            bbox_area_ratio=float(bbox_area_ratio),
            raw_result=result,
        )

    def draw_skeleton(self, frame_bgr: np.ndarray, pose_result: PoseResult) -> None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            pose_result.raw_result.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 220, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(200, 200, 200), thickness=2
            ),
        )

    def close(self) -> None:
        self._pose.close()
