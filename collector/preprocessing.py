"""
Приведение сырой последовательности landmarks произвольной длины к
фиксированной длине (config.sequence_length) через линейную интерполяцию
по времени, плюс нормализация координат.

Интерполяция выбрана вместо padding/truncation: padding добавляет
константные "мёртвые" кадры, которые модель может ошибочно интерпретировать
как часть техники, а truncation теряет часть движения. Интерполяция
сохраняет полную динамику приседа независимо от темпа выполнения.
"""
from typing import List

import numpy as np

from config import Config, NUM_LANDMARKS
from landmarks import normalize_frame, compute_scale
from canonical import JOINT_INDEX, COORD_Y


def resample_sequence(frames: List[np.ndarray], target_length: int) -> np.ndarray:
    """
    frames: список кадров [NUM_LANDMARKS, 4], произвольная длина >= 2.
    Возвращает np.ndarray [target_length, NUM_LANDMARKS, 4].
    """
    raw = np.stack(frames, axis=0)  # [T, NUM_LANDMARKS, 4]
    t_raw = np.linspace(0.0, 1.0, num=raw.shape[0])
    t_target = np.linspace(0.0, 1.0, num=target_length)

    resampled = np.empty((target_length, raw.shape[1], raw.shape[2]), dtype=np.float32)
    for lm_idx in range(raw.shape[1]):
        for coord_idx in range(raw.shape[2]):
            resampled[:, lm_idx, coord_idx] = np.interp(
                t_target, t_raw, raw[:, lm_idx, coord_idx]
            )
    return resampled


def build_sample(frames: List[np.ndarray], cfg: Config) -> np.ndarray:
    """
    Полный препроцессинг сырого repetition в готовый sample:
    1. Resample до фиксированной длины.
    2. Нормализация относительно центра таза + единый scale по первому кадру
       (он соответствует началу движения, близко к STANDING).
    """
    resampled = resample_sequence(frames, cfg.sequence_length)
    scale = compute_scale(resampled[0])
    normalized = np.stack(
        [normalize_frame(resampled[i], scale) for i in range(resampled.shape[0])],
        axis=0,
    )
    return normalized.astype(np.float32)


def orientation_diagnostics(sample: np.ndarray) -> dict:
    """
    Возвращает простую биомеханическую диагностику для canonical Y-down.
    Отрицательные shoulder_vs_hip означают, что плечи выше таза; положительные
    knee_vs_hip/foot_vs_knee означают, что колени/стопы ниже по экранной оси.
    """
    mid_shoulder_y = np.mean(
        sample[:, [JOINT_INDEX["left_shoulder"], JOINT_INDEX["right_shoulder"]], COORD_Y],
        axis=1,
    )
    mid_hip_y = np.mean(
        sample[:, [JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]], COORD_Y],
        axis=1,
    )
    mid_knee_y = np.mean(
        sample[:, [JOINT_INDEX["left_knee"], JOINT_INDEX["right_knee"]], COORD_Y],
        axis=1,
    )
    mid_foot_y = np.mean(
        sample[:, [JOINT_INDEX["left_foot_index"], JOINT_INDEX["right_foot_index"]], COORD_Y],
        axis=1,
    )
    return {
        "shoulder_vs_hip": float(np.median(mid_shoulder_y - mid_hip_y)),
        "knee_vs_hip": float(np.median(mid_knee_y - mid_hip_y)),
        "foot_vs_hip": float(np.median(mid_foot_y - mid_hip_y)),
        "foot_vs_knee": float(np.median(mid_foot_y - mid_knee_y)),
    }


def assert_canonical_orientation(sample: np.ndarray, initial_fraction: float = 0.2) -> None:
    initial_len = max(1, int(round(sample.shape[0] * initial_fraction)))
    diag = orientation_diagnostics(sample[:initial_len])
    if not (
        diag["shoulder_vs_hip"] < 0
        and diag["foot_vs_hip"] > 0
        and diag["foot_vs_knee"] > 0
    ):
        raise ValueError(f"Skeleton orientation is not canonical Y-down: {diag}")
