"""
Нормализация landmarks и вычисление biomechanical features (углы),
используемых repetition detector'ом.
"""
import numpy as np

from config import LANDMARK_NAMES, COORD_X, COORD_Y, COORD_Z, COORD_VIS

_IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def normalize_frame(frame_landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Нормализует один кадр landmarks к canonical representation.

    Origin: mid-hip. Scale: torso length, единый на весь sequence.
    X/Y/Z масштабируются одинаково; visibility остаётся confidence-каналом.
    """
    mid_hip = (frame_landmarks[_IDX["left_hip"]] + frame_landmarks[_IDX["right_hip"]]) / 2.0
    out = frame_landmarks.copy()
    out[:, COORD_X] = (frame_landmarks[:, COORD_X] - mid_hip[COORD_X]) / scale
    out[:, COORD_Y] = (frame_landmarks[:, COORD_Y] - mid_hip[COORD_Y]) / scale
    out[:, COORD_Z] = (frame_landmarks[:, COORD_Z] - mid_hip[COORD_Z]) / scale
    return out


def compute_scale(frame_landmarks: np.ndarray) -> float:
    """Torso length (shoulder_mid -> hip_mid) как масштаб тела."""
    mid_shoulder = (
        frame_landmarks[_IDX["left_shoulder"]] + frame_landmarks[_IDX["right_shoulder"]]
    ) / 2.0
    mid_hip = (frame_landmarks[_IDX["left_hip"]] + frame_landmarks[_IDX["right_hip"]]) / 2.0
    dist = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
    return float(dist) if dist > 1e-6 else 1e-6


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Угол в точке b (в градусах), образованный векторами b->a и b->c."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def knee_angle(frame_landmarks: np.ndarray) -> float:
    """Средний угол колена (hip-knee-ankle) по обеим ногам."""
    left = _angle(
        frame_landmarks[_IDX["left_hip"]],
        frame_landmarks[_IDX["left_knee"]],
        frame_landmarks[_IDX["left_ankle"]],
    )
    right = _angle(
        frame_landmarks[_IDX["right_hip"]],
        frame_landmarks[_IDX["right_knee"]],
        frame_landmarks[_IDX["right_ankle"]],
    )
    return (left + right) / 2.0


def hip_vertical_position(frame_landmarks: np.ndarray) -> float:
    """Y-координата центра таза (растёт вниз в image space)."""
    mid_hip = (frame_landmarks[_IDX["left_hip"]] + frame_landmarks[_IDX["right_hip"]]) / 2.0
    return float(mid_hip[COORD_Y])


def average_visibility(frame_landmarks: np.ndarray, names=None) -> float:
    if names is None:
        idxs = range(frame_landmarks.shape[0])
    else:
        idxs = [_IDX[n] for n in names]
    return float(np.mean([frame_landmarks[i, COORD_VIS] for i in idxs]))


CRITICAL_KEYPOINTS = [
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
