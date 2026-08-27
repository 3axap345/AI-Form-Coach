from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COLLECTOR_DIR))

from canonical import canonical_shape  # noqa: E402
from config import Config  # noqa: E402
from storage import StorageManager  # noqa: E402


class StorageManagerTests(unittest.TestCase):
    def test_save_close_and_undo_last_remove_persisted_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(dataset_root=str(Path(tmp) / "samples"))
            storage = StorageManager(cfg)
            sample = np.zeros(canonical_shape(), dtype=np.float32)
            task = storage.save_sample("correct", sample, {"class_label": "correct"})

            storage.close()

            self.assertTrue(task.npy_path.exists())
            self.assertEqual(np.load(task.npy_path).shape, canonical_shape())
            self.assertEqual(
                json.loads(task.json_path.read_text(encoding="utf-8")),
                {"class_label": "correct"},
            )

            self.assertEqual(storage.undo_last(), "correct")
            self.assertFalse(task.npy_path.exists())
            self.assertFalse(task.json_path.exists())
            self.assertIsNone(storage.undo_last())


if __name__ == "__main__":
    unittest.main()
