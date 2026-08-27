"""
AI Form Coach - Data Collector

Entry point. Захватывает видео, гоняет через MediaPipe Pose, детектирует
повторения приседания через state machine, проверяет качество и асинхронно
сохраняет валидные samples в датасет.

Управление:
  1-6    выбор класса (correct/shallow/knees_in/back_bent/heels_up/wrong_stance)
  SPACE  start/stop recording
  Z      undo последнего сохранённого sample
  R      сброс счётчиков (reps/saved/rejected) для текущей сессии
  Q      выход
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from camera import Camera, CameraError
from config import CONFIG, EXERCISE_CLASSES
from form_analysis import analyze_form, top_detected_issues
from pose import PoseEstimator
from preprocessing import build_sample
from quality import check_quality, is_duplicate
from repetition import Phase, RepetitionDetector
from storage import StorageManager
from ui import HudState, draw_hud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collector.main")


PHASE_LABELS = {
    Phase.STANDING: "STANDING",
    Phase.DESCENDING: "DESCENDING",
    Phase.BOTTOM: "BOTTOM",
    Phase.ASCENDING: "ASCENDING",
}


class Session:
    """Счётчики и текущее состояние UI-сессии (сбрасываются по 'R')."""

    def __init__(self):
        self.reps_detected = 0
        self.samples_saved = 0
        self.rejected = 0


def main() -> None:
    cfg = CONFIG

    try:
        camera = Camera(cfg)
    except CameraError as e:
        logger.error(str(e))
        logger.error(
            "Проверьте, что камера не занята другим приложением, и настройки сети/драйверов."
        )
        return

    pose_estimator = PoseEstimator(cfg)
    storage = StorageManager(cfg)
    classifier = None
    if cfg.enable_form_inference:
        model_path = Path(cfg.form_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if model_path.exists():
            try:
                from form_inference import FormClassifierInference

                classifier = FormClassifierInference(
                    model_path,
                    expected_sha256=cfg.form_model_sha256,
                )
                logger.info("Form classifier loaded: %s", model_path)
            except Exception:
                logger.exception("Failed to load form classifier; live collection will continue")
        else:
            logger.info("Form classifier model not found, inference disabled: %s", model_path)

    current_class_id = 0
    current_class_name = EXERCISE_CLASSES[ord("1")][1]
    recording = False
    session = Session()

    detector = RepetitionDetector(cfg)
    missing_frame_count = 0
    expected_frame_count = 0

    flash_counter = 0
    last_reject_reason = None
    last_prediction = None
    last_form_issues = []
    fps_smoother = []
    prev_time = time.time()

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                logger.error("Camera unavailable, stopping.")
                break

            frame = cv2.flip(frame, 1)  # зеркалим для естественного взаимодействия
            pose_result = pose_estimator.process(frame)

            warning = None
            knee_angle_display = None
            phase_label = "-"

            if pose_result is None:
                warning = "no person detected"
                if recording:
                    missing_frame_count += 1
                    expected_frame_count += 1
            else:
                if cfg.show_skeleton:
                    pose_estimator.draw_skeleton(frame, pose_result)

                if pose_result.bbox_area_ratio < cfg.min_bbox_area_ratio:
                    warning = "person too far from camera"
                elif pose_result.bbox_area_ratio > cfg.max_bbox_area_ratio:
                    warning = "person too close to camera"

                phase_label = PHASE_LABELS[detector.phase]

                if recording:
                    expected_frame_count += 1
                    completed = detector.update(pose_result.landmarks)
                    knee_angle_display = detector.current_angle

                    if completed is not None:
                        session.reps_detected += 1

                        report = check_quality(
                            raw_frames=completed.frames,
                            duration_sec=completed.duration_sec,
                            missing_frame_count=missing_frame_count,
                            total_expected_frames=expected_frame_count,
                            cfg=cfg,
                        )

                        if report.passed:
                            sample = build_sample(completed.frames, cfg)
                            if classifier is not None:
                                try:
                                    last_prediction = classifier.predict(sample)
                                    analysis = analyze_form(sample, cfg)
                                    last_form_issues = top_detected_issues(
                                        analysis,
                                        limit=cfg.max_form_issues_displayed,
                                    )
                                    logger.info(
                                        "Form prediction: %s (confidence=%.3f)",
                                        last_prediction["label"],
                                        last_prediction["confidence"],
                                    )
                                except Exception:
                                    logger.exception("Form prediction failed")
                                    last_prediction = None
                                    last_form_issues = []

                            last_sample = storage.last_sample_for_class(current_class_name)
                            if is_duplicate(sample, last_sample):
                                session.rejected += 1
                                last_reject_reason = "duplicate of previous sample"
                                logger.warning("Sample rejected: %s", last_reject_reason)
                            else:
                                metadata = {
                                    "class_label": current_class_name,
                                    "class_id": current_class_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "original_frame_count": len(completed.frames),
                                    "fps": cfg.target_fps,
                                    "duration_sec": round(completed.duration_sec, 3),
                                    "quality_score": round(report.score, 3),
                                    "rejection_reason": None,
                                }
                                storage.save_sample(current_class_name, sample, metadata)
                                session.samples_saved += 1
                                flash_counter = cfg.flash_frames_on_rep
                                last_reject_reason = None
                        else:
                            session.rejected += 1
                            last_reject_reason = report.reason
                            logger.warning("Sample rejected: %s", report.reason)

                        # Новый цикл замера missing/expected для следующего повторения
                        missing_frame_count = 0
                        expected_frame_count = 0

            # --- FPS ---
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_smoother.append(1.0 / dt)
                if len(fps_smoother) > 30:
                    fps_smoother.pop(0)
            fps = sum(fps_smoother) / len(fps_smoother) if fps_smoother else 0.0

            hud = HudState(
                class_name=current_class_name,
                fps=fps,
                reps_detected=session.reps_detected,
                samples_saved=session.samples_saved,
                rejected=session.rejected,
                phase=phase_label,
                knee_angle=knee_angle_display,
                recording=recording,
                warning=warning,
                last_reject_reason=last_reject_reason,
                prediction=last_prediction["label"] if last_prediction else None,
                prediction_confidence=last_prediction["confidence"] if last_prediction else None,
                form_issues=last_form_issues,
                flash=flash_counter > 0,
            )
            draw_hud(frame, hud)
            if flash_counter > 0:
                flash_counter -= 1

            cv2.imshow(cfg.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in EXERCISE_CLASSES:
                current_class_id, current_class_name = EXERCISE_CLASSES[key]
                logger.info("Class switched to: %s", current_class_name)
            elif key == ord(" "):
                recording = not recording
                logger.info("Recording %s", "STARTED" if recording else "STOPPED")
                if recording:
                    missing_frame_count = 0
                    expected_frame_count = 0
            elif key in (ord("z"), ord("Z"), 8):  # 8 = Backspace
                removed_class = storage.undo_last()
                if removed_class:
                    session.samples_saved = max(0, session.samples_saved - 1)
                    logger.info("Undo: last sample removed (class=%s)", removed_class)
                else:
                    logger.info("Nothing to undo")
            elif key in (ord("r"), ord("R")):
                session = Session()
                logger.info("Session counters reset")
            elif key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break

    finally:
        pose_estimator.close()
        storage.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
