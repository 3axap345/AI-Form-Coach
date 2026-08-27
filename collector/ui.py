"""
Отрисовка чистого HUD поверх кадра.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HudState:
    class_name: str = "-"
    fps: float = 0.0
    reps_detected: int = 0
    samples_saved: int = 0
    rejected: int = 0
    phase: str = "-"
    knee_angle: Optional[float] = None
    recording: bool = False
    warning: Optional[str] = None  # напр. "too far", "no person"
    last_reject_reason: Optional[str] = None
    prediction: Optional[str] = None
    prediction_confidence: Optional[float] = None
    form_issues: Optional[list[dict]] = None
    flash: bool = False  # визуальный сигнал при новом повторении


def _put(img, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_panel(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 22, 28), thickness=-1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 90, 105), thickness=1)


def _status_text(state: HudState) -> str:
    if state.flash:
        return "Processing"
    if state.warning == "no person detected":
        return "Tracking"
    if not state.recording:
        return "Ready"
    if state.phase in ("DESCENDING", "BOTTOM", "ASCENDING"):
        return "Squatting"
    return "Ready"


def draw_status(frame: np.ndarray, state: HudState, x: int, y: int) -> int:
    status = _status_text(state)
    color = {
        "Tracking": (0, 190, 255),
        "Ready": (80, 220, 120),
        "Squatting": (255, 210, 80),
        "Processing": (180, 160, 255),
    }.get(status, (255, 255, 255))

    _put(frame, "STATUS", (x, y), scale=0.5, color=(190, 200, 210), thickness=1)
    _put(frame, status, (x, y + 35), scale=0.9, color=color, thickness=2)
    return y + 68


def draw_rep_counter(frame: np.ndarray, state: HudState, x: int, y: int) -> int:
    _put(frame, "REPS", (x, y), scale=0.5, color=(190, 200, 210), thickness=1)
    _put(
        frame, str(state.reps_detected), (x, y + 48), scale=1.6, color=(255, 255, 255), thickness=3
    )
    return y + 86


def draw_last_prediction(frame: np.ndarray, state: HudState, x: int, y: int) -> int:
    _put(frame, "LAST REP", (x, y), scale=0.5, color=(190, 200, 210), thickness=1)

    if state.prediction is None:
        _put(frame, "--", (x, y + 42), scale=1.1, color=(210, 215, 220), thickness=2)
        return y + 76

    label = state.prediction.upper()
    is_correct = state.prediction == "correct"
    label_color = (80, 230, 120) if is_correct else (60, 90, 255)
    cv2.rectangle(frame, (x, y + 16), (x + 210, y + 58), label_color, thickness=-1)
    _put(frame, label, (x + 12, y + 48), scale=0.9, color=(255, 255, 255), thickness=2)

    confidence = state.prediction_confidence
    if confidence is not None:
        _put(
            frame,
            f"Confidence: {confidence:.0%}",
            (x, y + 88),
            scale=0.62,
            color=(230, 235, 240),
            thickness=1,
        )
        return y + 110
    return y + 74


def draw_diagnostics(frame: np.ndarray, state: HudState, x: int, y: int) -> None:
    _put(
        frame,
        f"Class: {state.class_name}  Saved: {state.samples_saved}  Rejected: {state.rejected}",
        (x, y),
        scale=0.5,
        color=(200, 210, 220),
    )
    y += 24
    if state.knee_angle is not None:
        _put(frame, f"Knee angle: {state.knee_angle:.0f} deg", (x, y), scale=0.5)
        y += 24
    for issue in state.form_issues or []:
        _put(frame, f"Feedback: {issue['message']}", (x, y), scale=0.5, color=(70, 190, 255))
        y += 24


def draw_hud(frame: np.ndarray, state: HudState) -> None:
    h, w = frame.shape[:2]

    if state.flash:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), thickness=15)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    panel_x, panel_y = 18, 18
    panel_w = min(360, max(300, w // 4))
    panel_h = min(440, h - panel_y - 18)
    _draw_panel(frame, panel_x, panel_y, panel_w, panel_h)

    x = panel_x + 18
    y = panel_y + 34
    _put(frame, "AI FORM COACH", (x, y), scale=0.72, color=(255, 255, 255), thickness=2)
    y += 42
    y = draw_status(frame, state, x, y)
    y = draw_rep_counter(frame, state, x, y)
    y = draw_last_prediction(frame, state, x, y)
    draw_diagnostics(frame, state, x, y)

    rec_color = (70, 80, 255) if state.recording else (160, 165, 170)
    rec_text = "REC" if state.recording else "PAUSED"
    _put(
        frame,
        rec_text,
        (panel_x + panel_w - 95, panel_y + 34),
        color=rec_color,
        scale=0.65,
        thickness=2,
    )

    if state.last_reject_reason:
        _put(frame, f"Rejected: {state.last_reject_reason}", (15, h - 20), color=(0, 165, 255))

    _put(
        frame,
        "[1-6] class  [SPACE] rec  [Z] undo  [R] reset  [Q] quit",
        (15, h - 45),
        scale=0.5,
        color=(200, 200, 200),
    )
