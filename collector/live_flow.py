"""Application-level processing of one completed live repetition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
from config import Config
from form_analysis import analyze_form, top_detected_issues
from preprocessing import build_sample
from quality import check_quality, is_duplicate
from repetition import CompletedRep
from storage import StorageManager

logger = logging.getLogger("collector.live_flow")


class FormClassifier(Protocol):
    def predict(self, sample: np.ndarray) -> dict: ...


@dataclass
class LiveRepResult:
    saved: bool
    rejection_reason: str | None
    sample: np.ndarray | None
    prediction: dict | None
    form_issues: list[dict]


class LiveRepProcessor:
    """Run quality, preprocessing, feedback, duplicate checks, and storage."""

    def __init__(
        self,
        cfg: Config,
        storage: StorageManager,
        classifier: FormClassifier | None = None,
    ):
        self._cfg = cfg
        self._storage = storage
        self._classifier = classifier

    def process(
        self,
        completed: CompletedRep,
        missing_frame_count: int,
        total_expected_frames: int,
        class_name: str,
        class_id: int,
    ) -> LiveRepResult:
        report = check_quality(
            raw_frames=completed.frames,
            duration_sec=completed.duration_sec,
            missing_frame_count=missing_frame_count,
            total_expected_frames=total_expected_frames,
            cfg=self._cfg,
        )
        if not report.passed:
            return LiveRepResult(False, report.reason, None, None, [])

        sample = build_sample(completed.frames, self._cfg)
        prediction = None
        form_issues: list[dict] = []
        if self._classifier is not None:
            try:
                prediction = self._classifier.predict(sample)
                analysis = analyze_form(sample, self._cfg)
                form_issues = top_detected_issues(
                    analysis,
                    limit=self._cfg.max_form_issues_displayed,
                )
            except (RuntimeError, ValueError) as error:
                logger.exception("Form prediction failed: %s", error)

        last_sample = self._storage.last_sample_for_class(class_name)
        if is_duplicate(sample, last_sample):
            return LiveRepResult(
                False,
                "duplicate of previous sample",
                sample,
                prediction,
                form_issues,
            )

        metadata = {
            "class_label": class_name,
            "class_id": class_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_frame_count": len(completed.frames),
            "fps": self._cfg.target_fps,
            "duration_sec": round(completed.duration_sec, 3),
            "quality_score": round(report.score, 3),
            "rejection_reason": None,
        }
        self._storage.save_sample(class_name, sample, metadata)
        return LiveRepResult(True, None, sample, prediction, form_issues)
