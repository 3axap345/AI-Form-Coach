"""
Централизованная конфигурация Data Collector.
Все пороги и magic numbers живут здесь, чтобы не расползаться по проекту.
"""

from dataclasses import dataclass

import cv2
from canonical import (
    CANONICAL_JOINTS,
    MEDIAPIPE_LANDMARK_INDICES,
    NUM_JOINTS,
    SEQUENCE_LENGTH,
)

# Индексы landmarks в MediaPipe Pose (полный список из 33 точек), которые
# реально нужны для анализа приседания. Лицо, кисти и прочие точки не
# сохраняются — они не несут сигнала для техники squat и только увеличивают
# объём датасета.
SELECTED_LANDMARKS = MEDIAPIPE_LANDMARK_INDICES

LANDMARK_NAMES = list(CANONICAL_JOINTS)
LANDMARK_INDICES = list(SELECTED_LANDMARKS.values())
NUM_LANDMARKS = NUM_JOINTS

# class_id -> (label_index, label_name)
EXERCISE_CLASSES = {
    ord("1"): (0, "correct"),
    ord("2"): (1, "shallow"),
    ord("3"): (2, "knees_in"),
    ord("4"): (3, "back_bent"),
    ord("5"): (4, "heels_up"),
    ord("6"): (5, "wrong_stance"),
}


@dataclass
class Config:
    # --- Camera ---
    camera_index: int = 0
    target_width: int = 1280
    target_height: int = 720
    target_fps: int = 30
    # Backends пробуются по порядку на Windows. CAP_DSHOW обычно быстрее
    # открывается и даёт меньшую задержку, но менее стабилен на некоторых
    # встроенных/новых камерах — тогда используем fallback на MSMF, затем ANY.
    windows_backends: tuple = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 1.0

    # --- MediaPipe Pose ---
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0 = lite (быстрее), 1 = full, 2 = heavy (точнее)

    # --- Sequence / dataset ---
    sequence_length: int = SEQUENCE_LENGTH  # итоговая длина sample после resampling
    dataset_root: str = "dataset/squat"

    # --- Repetition detection (state machine) ---
    standing_knee_angle: float = 160.0  # колено выпрямлено => считаем что стоим
    bottom_knee_angle: float = 100.0  # порог для фиксации нижней точки приседа
    hysteresis: float = 8.0  # люфт вокруг порогов, чтобы убрать дребезг
    standing_confirm_frames: int = 5  # столько кадров подряд в STANDING перед новым циклом
    min_rep_duration_sec: float = 0.4
    max_rep_tracking_duration_sec: float = 5.0  # reset the FSM if a repetition stalls
    smoothing_window: int = 5  # окно скользящего среднего по углам

    # --- Quality checks ---
    max_saved_rep_duration_sec: float = 5.0  # longest repetition accepted for storage
    min_avg_visibility: float = 0.6
    min_keypoint_visibility: float = 0.4  # для критичных точек (колено/таз/лодыжка)
    max_missing_frame_ratio: float = 0.15  # доля кадров без обнаруженной позы
    min_bbox_area_ratio: float = 0.05  # человек слишком далеко от камеры
    max_bbox_area_ratio: float = 0.85  # человек слишком близко к камере

    # --- Async saving ---
    save_queue_maxsize: int = 32

    # --- Model inference ---
    enable_form_inference: bool = True
    form_model_path: str = "models/squat_binary/best_model.pt"
    form_model_sha256: str | None = None  # optional expected hash for a deployed model

    # --- Rule-based form analysis ---
    form_analysis_bottom_window: int = 9
    form_analysis_smoothing_window: int = 5
    form_analysis_min_visibility: float = 0.4
    shallow_depth_knee_angle: float = 115.0
    excessive_forward_lean_angle: float = 28.0
    knee_valgus_ratio_drop: float = 0.18
    heel_instability_threshold: float = 0.12
    max_form_issues_displayed: int = 3

    # --- UI ---
    window_name: str = "AI Form Coach - Data Collector"
    show_skeleton: bool = True
    flash_frames_on_rep: int = 6  # сколько кадров держать визуальную вспышку

    def __post_init__(self) -> None:
        positive_values = {
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_fps": self.target_fps,
            "reconnect_attempts": self.reconnect_attempts,
            "sequence_length": self.sequence_length,
            "standing_confirm_frames": self.standing_confirm_frames,
            "max_rep_tracking_duration_sec": self.max_rep_tracking_duration_sec,
            "max_saved_rep_duration_sec": self.max_saved_rep_duration_sec,
            "smoothing_window": self.smoothing_window,
            "save_queue_maxsize": self.save_queue_maxsize,
            "form_analysis_bottom_window": self.form_analysis_bottom_window,
            "form_analysis_smoothing_window": self.form_analysis_smoothing_window,
            "max_form_issues_displayed": self.max_form_issues_displayed,
            "flash_frames_on_rep": self.flash_frames_on_rep,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        unit_interval_values = {
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "min_avg_visibility": self.min_avg_visibility,
            "min_keypoint_visibility": self.min_keypoint_visibility,
            "max_missing_frame_ratio": self.max_missing_frame_ratio,
            "min_bbox_area_ratio": self.min_bbox_area_ratio,
            "max_bbox_area_ratio": self.max_bbox_area_ratio,
            "form_analysis_min_visibility": self.form_analysis_min_visibility,
        }
        for name, value in unit_interval_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        if self.min_bbox_area_ratio > self.max_bbox_area_ratio:
            raise ValueError("min_bbox_area_ratio must not exceed max_bbox_area_ratio")
        if self.min_rep_duration_sec < 0:
            raise ValueError("min_rep_duration_sec must be non-negative")
        if self.reconnect_delay_sec < 0:
            raise ValueError("reconnect_delay_sec must be non-negative")


CONFIG = Config()
