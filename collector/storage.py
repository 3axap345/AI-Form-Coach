"""
Асинхронное сохранение samples на диск (отдельный поток + очередь), чтобы
disk I/O никогда не блокировал detection loop. Плюс undo последнего sample.
"""

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from config import Config

logger = logging.getLogger("collector.storage")


@dataclass
class SaveTask:
    class_name: str
    sample: np.ndarray
    metadata: Dict[str, Any]
    npy_path: Path
    json_path: Path
    cancelled: threading.Event = field(default_factory=threading.Event)


class StorageManager:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._root = Path(cfg.dataset_root)
        self._queue: "queue.Queue[Optional[SaveTask]]" = queue.Queue(maxsize=cfg.save_queue_maxsize)
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._history: List[SaveTask] = []  # порядок сохранений, для undo
        self._last_sample_by_class: Dict[str, np.ndarray] = {}

        self._root.mkdir(parents=True, exist_ok=True)
        self._init_counters()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _init_counters(self) -> None:
        """При старте сканирует уже существующий датасет, чтобы продолжить
        нумерацию, а не перезаписать существующие samples."""
        if not self._root.exists():
            return
        for class_dir in self._root.iterdir():
            if not class_dir.is_dir():
                continue
            max_idx = 0
            for f in class_dir.glob("sample_*.npy"):
                try:
                    max_idx = max(max_idx, int(f.stem.split("_")[-1]))
                except ValueError:
                    continue
            self._counters[class_dir.name] = max_idx

    def last_sample_for_class(self, class_name: str) -> Optional[np.ndarray]:
        return self._last_sample_by_class.get(class_name)

    def save_sample(
        self, class_name: str, sample: np.ndarray, metadata: Dict[str, Any]
    ) -> SaveTask:
        class_dir = self._root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            next_idx = self._counters.get(class_name, 0) + 1
            self._counters[class_name] = next_idx

        npy_path = class_dir / f"sample_{next_idx:06d}.npy"
        json_path = class_dir / f"sample_{next_idx:06d}.json"

        task = SaveTask(class_name, sample, metadata, npy_path, json_path)
        self._history.append(task)
        self._last_sample_by_class[class_name] = sample

        try:
            self._queue.put_nowait(task)
        except queue.Full:
            logger.warning("Save queue full — writing synchronously (may cause a brief hitch)")
            self._write(task)

        return task

    def undo_last(self) -> Optional[str]:
        """Отменяет последний сохранённый sample (даже если ещё в очереди на
        запись). Возвращает имя класса удалённого sample или None, если
        отменять нечего."""
        if not self._history:
            return None
        task = self._history.pop()
        task.cancelled.set()

        with self._lock:
            current_idx = self._extract_index(task.npy_path)
            if self._counters.get(task.class_name, 0) == current_idx:
                self._counters[task.class_name] = current_idx - 1

        if task.npy_path.exists():
            task.npy_path.unlink()
        if task.json_path.exists():
            task.json_path.unlink()

        logger.info("Undo: removed sample %s", task.npy_path.name)
        return task.class_name

    @staticmethod
    def _extract_index(npy_path: Path) -> int:
        return int(npy_path.stem.split("_")[-1])

    def _write(self, task: SaveTask) -> None:
        if task.cancelled.is_set():
            return
        np.save(task.npy_path, task.sample)
        with open(task.json_path, "w", encoding="utf-8") as f:
            json.dump(task.metadata, f, ensure_ascii=False, indent=2)
        logger.info("Sample saved: %s", task.npy_path.name)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                self._write(task)
            except Exception:
                logger.exception("Failed to save sample %s", task.npy_path)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(None)
        self._worker_thread.join(timeout=5.0)
