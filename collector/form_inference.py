from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from canonical import LABEL_TO_NAME, canonical_shape
from form_model import FormClassifier, predict_probabilities


class FormClassifierInference:
    def __init__(self, model_path: Path, device: Optional[str] = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_to_name = {
            int(k): v for k, v in checkpoint.get("label_to_name", LABEL_TO_NAME).items()
        }
        self.model = FormClassifier().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, sample: np.ndarray) -> dict:
        expected = canonical_shape()
        if tuple(sample.shape) != expected:
            raise ValueError(f"expected sample shape {expected}, got {sample.shape}")

        tensor = torch.from_numpy(sample.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = predict_probabilities(self.model, tensor)[0].cpu().numpy()
        label_id = int(np.argmax(probs))
        probabilities = {
            self.label_to_name.get(i, str(i)): float(probs[i])
            for i in range(len(probs))
        }
        return {
            "label": self.label_to_name.get(label_id, str(label_id)),
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probabilities": probabilities,
        }
