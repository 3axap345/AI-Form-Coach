"""
State machine для автоматического обнаружения повторений приседания.

STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

Основной сигнал: сглаженный knee angle (hip-knee-ankle). Гистерезис и
подтверждение нескольких standing-кадров защищают переходы от шума.
"""

import collections
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np
from config import Config
from landmarks import knee_angle

logger = logging.getLogger("collector.repetition")


class Phase(Enum):
    STANDING = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()


@dataclass
class CompletedRep:
    frames: List[np.ndarray]  # сырые (ненормализованные) landmark-кадры за весь rep
    duration_sec: float
    min_knee_angle: float


class RepetitionDetector:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._phase = Phase.STANDING
        self._angle_history: collections.deque[float] = collections.deque(
            maxlen=cfg.smoothing_window
        )
        self._standing_streak = 0
        self._rep_buffer: List[np.ndarray] = []
        self._rep_start_time: Optional[float] = None
        self._min_angle_in_rep: float = 999.0
        self._prev_smoothed_angle: Optional[float] = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def current_angle(self) -> Optional[float]:
        """Последний сглаженный knee angle — для отображения в UI."""
        return self._prev_smoothed_angle

    def _smoothed_angle(self, raw_angle: float) -> float:
        self._angle_history.append(raw_angle)
        return float(np.mean(self._angle_history))

    def update(
        self, frame_landmarks: np.ndarray, now: Optional[float] = None
    ) -> Optional[CompletedRep]:
        """
        Скармливаем один кадр landmarks. Возвращает CompletedRep, если в этом
        кадре только что завершилось полное повторение, иначе None.
        """
        now = now if now is not None else time.time()
        raw_angle = knee_angle(frame_landmarks)
        angle = self._smoothed_angle(raw_angle)

        cfg = self._cfg
        completed: Optional[CompletedRep] = None

        if self._phase == Phase.STANDING:
            standing_threshold = cfg.standing_knee_angle - cfg.hysteresis
            if angle >= standing_threshold:
                self._standing_streak += 1
            elif self._standing_streak >= cfg.standing_confirm_frames:
                # Начинаем новое повторение
                self._phase = Phase.DESCENDING
                self._rep_buffer = [frame_landmarks]
                self._rep_start_time = now
                self._min_angle_in_rep = angle
                self._standing_streak = 0
                logger.debug("Phase -> DESCENDING (angle=%.1f)", angle)
            else:
                # A valid start needs consecutive standing frames beforehand.
                self._standing_streak = 0

        elif self._phase == Phase.DESCENDING:
            self._rep_buffer.append(frame_landmarks)
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            # Нижняя точка: угол опустился ниже bottom-порога, либо скорость
            # изменения угла сменила знак (локальный минимум) — берём то, что
            # сработает раньше.
            reached_bottom_threshold = angle < cfg.bottom_knee_angle + cfg.hysteresis
            local_minimum = (
                self._prev_smoothed_angle is not None
                and angle > self._prev_smoothed_angle
                and self._min_angle_in_rep < cfg.standing_knee_angle - cfg.hysteresis
            )
            if reached_bottom_threshold or local_minimum:
                self._phase = Phase.BOTTOM
                logger.debug("Phase -> BOTTOM (angle=%.1f)", angle)

        elif self._phase == Phase.BOTTOM:
            self._rep_buffer.append(frame_landmarks)
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            previous_angle = self._prev_smoothed_angle
            if previous_angle is not None and angle > previous_angle:
                self._phase = Phase.ASCENDING
                logger.debug("Phase -> ASCENDING (angle=%.1f)", angle)

        elif self._phase == Phase.ASCENDING:
            self._rep_buffer.append(frame_landmarks)
            if angle > cfg.standing_knee_angle - cfg.hysteresis:
                # Повторение завершено
                rep_start_time = self._rep_start_time
                if rep_start_time is None:
                    raise RuntimeError("repetition reached ASCENDING without a start time")
                duration = now - rep_start_time
                completed = CompletedRep(
                    frames=self._rep_buffer,
                    duration_sec=duration,
                    min_knee_angle=self._min_angle_in_rep,
                )
                logger.info(
                    "Repetition completed: duration=%.2fs min_angle=%.1f frames=%d",
                    duration,
                    self._min_angle_in_rep,
                    len(self._rep_buffer),
                )
                self._phase = Phase.STANDING
                self._rep_buffer = []
                self._standing_streak = 0

        self._prev_smoothed_angle = angle

        # Защита от "застрявших" повторений (слишком долго не возвращаемся в standing)
        if self._phase != Phase.STANDING and self._rep_start_time is not None:
            elapsed = now - self._rep_start_time
            if elapsed > cfg.max_rep_tracking_duration_sec:
                logger.warning(
                    "Repetition timeout after %.1fs, resetting FSM without saving", elapsed
                )
                self._reset()

        return completed

    def _reset(self) -> None:
        self._phase = Phase.STANDING
        self._rep_buffer = []
        self._rep_start_time = None
        self._min_angle_in_rep = 999.0
        self._standing_streak = 0
