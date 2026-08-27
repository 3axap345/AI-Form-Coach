from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from canonical import LABEL_TO_NAME, canonical_shape
from form_model import FormClassifier, predict_probabilities


class ModelLoadError(ValueError):
    """Raised when a model file is unsafe, incompatible, or corrupted."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a model file without loading it."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_state_dict(state_dict: object, model: FormClassifier) -> None:
    if not isinstance(state_dict, Mapping):
        raise ModelLoadError("model file must contain a PyTorch state_dict mapping")

    expected_keys = set(model.state_dict())
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ModelLoadError(
            "model state_dict keys do not match FormClassifier "
            f"(missing={missing}, unexpected={unexpected})"
        )


class FormClassifierInference:
    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
        expected_sha256: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.model_path.is_file():
            raise ModelLoadError(f"model file does not exist: {self.model_path}")
        if expected_sha256 is None:
            raise ModelLoadError("an expected model SHA-256 is required before loading weights")
        expected_sha256 = expected_sha256.lower()
        if len(expected_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha256
        ):
            raise ModelLoadError("configured model SHA-256 must be 64 hexadecimal characters")
        actual_sha256 = sha256_file(self.model_path)
        if actual_sha256 != expected_sha256:
            raise ModelLoadError(
                f"model SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )

        self.model = FormClassifier().to(self.device)
        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except (pickle.UnpicklingError, RuntimeError, ValueError, OSError) as error:
            raise ModelLoadError(
                f"could not safely load model weights from {self.model_path}: {error}"
            ) from error

        _validate_state_dict(state_dict, self.model)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ModelLoadError(f"model weights are incompatible: {error}") from error

        self.label_to_name = LABEL_TO_NAME
        self.model.eval()

    def predict(self, sample: np.ndarray) -> dict:
        expected = canonical_shape()
        if tuple(sample.shape) != expected:
            raise ValueError(f"expected sample shape {expected}, got {sample.shape}")

        tensor = torch.from_numpy(sample.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = predict_probabilities(self.model, tensor)[0].cpu().numpy()
        label_id = int(np.argmax(probs))
        probabilities = {
            self.label_to_name.get(i, str(i)): float(probs[i]) for i in range(len(probs))
        }
        return {
            "label": self.label_to_name.get(label_id, str(label_id)),
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probabilities": probabilities,
        }
