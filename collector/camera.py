"""
Обёртка над cv2.VideoCapture с fallback между backend'ами на Windows
и автоматическим переподключением при обрыве камеры.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from config import Config

logger = logging.getLogger("collector.camera")


class CameraError(Exception):
    """Камеру не удалось открыть ни с одним из backend'ов."""


class Camera:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._active_backend: Optional[int] = None
        self._open()

    def _open(self) -> None:
        """Пробует backend'ы по порядку, пока один не откроется успешно."""
        for backend in self._cfg.windows_backends:
            cap = cv2.VideoCapture(self._cfg.camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.target_height)
                cap.set(cv2.CAP_PROP_FPS, self._cfg.target_fps)
                # Небольшой буфер уменьшает задержку (не копим старые кадры)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Проверяем, что кадры реально читаются, а не просто "открыт"
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    self._active_backend = backend
                    logger.info("Camera opened with backend=%s", backend)
                    return
                cap.release()
            else:
                cap.release()
        raise CameraError(
            f"Не удалось открыть камеру index={self._cfg.camera_index} "
            f"ни с одним из backend'ов: {self._cfg.windows_backends}"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Читает кадр. При обрыве пытается переподключиться."""
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if ok:
            return True, frame

        logger.warning("Camera read failed, attempting reconnect...")
        return self._reconnect()

    def _reconnect(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self._cfg.reconnect_attempts + 1):
            logger.warning("Reconnect attempt %d/%d", attempt, self._cfg.reconnect_attempts)
            time.sleep(self._cfg.reconnect_delay_sec)
            try:
                self._open()
            except CameraError:
                continue
            ok, frame = self._cap.read()
            if ok:
                logger.info("Camera reconnected successfully")
                return True, frame

        logger.error("Camera reconnect failed after %d attempts", self._cfg.reconnect_attempts)
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
