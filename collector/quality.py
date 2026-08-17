"""
Проверка качества repetition перед сохранением в датасет.
Каждая функция возвращает (passed: bool, reason: str), reason пуст если passed.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import Config, COORD_VIS
from landmarks import average_visibility, CRITICAL_KEYPOINTS


@dataclass
class QualityReport:
    passed: bool
    reason: str
    score: float  # 0.0-1.0, грубая метрика качества (средняя visibility)


def check_quality(
    raw_frames: List[np.ndarray],
    duration_sec: float,
    missing_frame_count: int,
    total_expected_frames: int,
    cfg: Config,
) -> QualityReport:
    # 1. Длительность
    if duration_sec < cfg.min_rep_duration_sec:
        return QualityReport(False, "repetition too short (likely false trigger)", 0.0)
    if duration_sec > cfg.max_rep_duration_sec:
        return QualityReport(False, "repetition too long (likely stalled movement)", 0.0)

    # 2. Доля пропущенных кадров (человек не обнаружен MediaPipe)
    if total_expected_frames > 0:
        missing_ratio = missing_frame_count / total_expected_frames
        if missing_ratio > cfg.max_missing_frame_ratio:
            return QualityReport(
                False, f"too many missing frames ({missing_ratio:.0%})", 0.0
            )

    if len(raw_frames) < 3:
        return QualityReport(False, "incomplete repetition (too few frames)", 0.0)

    # 3. Средняя visibility по критичным точкам (колени/таз/лодыжки) по всей sequence
    critical_vis = [average_visibility(f, CRITICAL_KEYPOINTS) for f in raw_frames]
    min_critical_vis = float(np.min(critical_vis))
    avg_critical_vis = float(np.mean(critical_vis))

    if min_critical_vis < cfg.min_keypoint_visibility:
        return QualityReport(
            False,
            f"critical keypoint visibility too low (min={min_critical_vis:.2f})",
            avg_critical_vis,
        )

    overall_vis = float(np.mean([average_visibility(f) for f in raw_frames]))
    if overall_vis < cfg.min_avg_visibility:
        return QualityReport(
            False, f"overall visibility too low (avg={overall_vis:.2f})", overall_vis
        )

    return QualityReport(True, "", overall_vis)


def is_duplicate(new_sample: np.ndarray, last_sample: Optional[np.ndarray], threshold: float = 0.02) -> bool:
    """
    Простая проверка на дубликат: среднеквадратичная разница между
    нормализованными sequences. Достаточно для v1 — усложнять до DTW
    имеет смысл только если дубликаты реально появятся на практике.
    """
    if last_sample is None or last_sample.shape != new_sample.shape:
        return False
    mse = float(np.mean((new_sample[:, :, :2] - last_sample[:, :, :2]) ** 2))
    return mse < threshold
