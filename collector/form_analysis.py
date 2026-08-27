"""
Rule-based squat form diagnostics on canonical [60, 12, 4] samples.

These rules are engineering heuristics for coaching feedback. They are not a
medical assessment and should be calibrated with real target-camera data.
"""

from __future__ import annotations

import numpy as np
from canonical import COORD_VIS, JOINT_INDEX, canonical_shape
from config import Config

ISSUE_MESSAGES = {
    "knee_valgus": "Knees moving inward",
    "shallow_depth": "Squat depth appears shallow",
    "excessive_forward_lean": "Excessive forward lean",
    "heel_instability": "Heel instability",
}


def _joint(sample: np.ndarray, name: str) -> np.ndarray:
    return sample[:, JOINT_INDEX[name], :]


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _visibility(sample: np.ndarray, names: list[str]) -> np.ndarray:
    return np.mean(
        np.stack([_joint(sample, name)[:, COORD_VIS] for name in names], axis=1),
        axis=1,
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float32) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a[:, :2] - b[:, :2]
    bc = c[:, :2] - b[:, :2]
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9
    cos_angle = np.clip(np.sum(ba * bc, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle)).astype(np.float32)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _bottom_indices(hip_y: np.ndarray, cfg: Config) -> np.ndarray:
    center = int(np.argmax(_smooth(hip_y, cfg.form_analysis_smoothing_window)))
    half = max(1, cfg.form_analysis_bottom_window // 2)
    start = max(0, center - half)
    end = min(len(hip_y), center + half + 1)
    return np.arange(start, end)


def _standing_indices(length: int) -> np.ndarray:
    count = max(3, int(round(length * 0.2)))
    return np.arange(0, min(length, count))


def _issue(score: float, threshold: float, message: str) -> dict:
    raw_score = 0.0 if not np.isfinite(score) else float(max(0.0, score))
    return {
        "detected": bool(raw_score >= threshold),
        "score": float(min(1.0, raw_score)),
        "message": message,
    }


def analyze_form(sample: np.ndarray, cfg: Config) -> dict:
    expected = canonical_shape(cfg.sequence_length)
    if tuple(sample.shape) != expected:
        raise ValueError(f"expected sample shape {expected}, got {sample.shape}")

    left_knee = _angle(
        _joint(sample, "left_hip"), _joint(sample, "left_knee"), _joint(sample, "left_ankle")
    )
    right_knee = _angle(
        _joint(sample, "right_hip"), _joint(sample, "right_knee"), _joint(sample, "right_ankle")
    )
    knee_visibility = _visibility(
        sample,
        ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
    )
    knee_angles = _smooth((left_knee + right_knee) / 2.0, cfg.form_analysis_smoothing_window)
    mid_hip = _midpoint(_joint(sample, "left_hip"), _joint(sample, "right_hip"))
    bottom = _bottom_indices(mid_hip[:, 1], cfg)
    standing = _standing_indices(len(sample))

    visibility_ok = knee_visibility >= cfg.form_analysis_min_visibility
    bottom_weights = knee_visibility[bottom] * visibility_ok[bottom]

    # Depth: shallow squats keep a large hip-knee-ankle angle at the bottom.
    bottom_knee_angle = _weighted_mean(knee_angles[bottom], bottom_weights)
    shallow_score = bottom_knee_angle / max(cfg.shallow_depth_knee_angle, 1e-6)

    # Torso lean: mid-hip -> mid-shoulder angle away from vertical-up in x/y.
    mid_shoulder = _midpoint(_joint(sample, "left_shoulder"), _joint(sample, "right_shoulder"))
    torso = mid_shoulder[:, :2] - mid_hip[:, :2]
    torso_angle = np.degrees(np.arctan2(np.abs(torso[:, 0]), np.abs(torso[:, 1]) + 1e-9))
    torso_visibility = _visibility(
        sample, ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    )
    torso_bottom = _weighted_mean(torso_angle[bottom], torso_visibility[bottom])
    lean_score = torso_bottom / max(cfg.excessive_forward_lean_angle, 1e-6)

    # Valgus: knees narrow relative to foot width at bottom compared to standing.
    knee_width = np.abs(_joint(sample, "left_knee")[:, 0] - _joint(sample, "right_knee")[:, 0])
    foot_width = np.abs(
        _joint(sample, "left_foot_index")[:, 0] - _joint(sample, "right_foot_index")[:, 0]
    )
    width_visibility = _visibility(
        sample,
        ["left_knee", "right_knee", "left_foot_index", "right_foot_index"],
    )
    ratio = knee_width / (foot_width + 1e-6)
    standing_ratio = _weighted_mean(ratio[standing], width_visibility[standing])
    bottom_ratio = _weighted_mean(ratio[bottom], width_visibility[bottom])
    ratio_drop = (standing_ratio - bottom_ratio) / max(standing_ratio, 1e-6)
    valgus_score = ratio_drop / max(cfg.knee_valgus_ratio_drop, 1e-6)

    # Heel instability: heel moves relative to ankle/foot_index across the rep.
    side_scores = []
    for side in ("left", "right"):
        heel = _joint(sample, f"{side}_heel")[:, :2]
        ankle = _joint(sample, f"{side}_ankle")[:, :2]
        foot = _joint(sample, f"{side}_foot_index")[:, :2]
        heel_rel_ankle = heel - ankle
        heel_rel_foot = heel - foot
        baseline = np.mean(
            np.concatenate([heel_rel_ankle[standing], heel_rel_foot[standing]], axis=0), axis=0
        )
        bottom_rel = np.concatenate([heel_rel_ankle[bottom], heel_rel_foot[bottom]], axis=0)
        displacement = np.linalg.norm(bottom_rel - baseline, axis=1)
        side_scores.append(float(np.median(displacement)))
    heel_motion = max(side_scores) if side_scores else 0.0
    heel_score = heel_motion / max(cfg.heel_instability_threshold, 1e-6)

    return {
        "knee_valgus": _issue(valgus_score, 1.0, ISSUE_MESSAGES["knee_valgus"]),
        "shallow_depth": _issue(shallow_score, 1.0, ISSUE_MESSAGES["shallow_depth"]),
        "excessive_forward_lean": _issue(lean_score, 1.0, ISSUE_MESSAGES["excessive_forward_lean"]),
        "heel_instability": _issue(heel_score, 1.0, ISSUE_MESSAGES["heel_instability"]),
        "_metrics": {
            "bottom_knee_angle": float(bottom_knee_angle),
            "torso_lean_angle": float(torso_bottom),
            "knee_width_ratio_drop": float(ratio_drop),
            "heel_motion": float(heel_motion),
            "bottom_frames": bottom.tolist(),
        },
    }


def top_detected_issues(analysis: dict, limit: int = 3) -> list[dict]:
    issues = [
        {"key": key, **value}
        for key, value in analysis.items()
        if not key.startswith("_") and value.get("detected")
    ]
    return sorted(issues, key=lambda item: item["score"], reverse=True)[:limit]
