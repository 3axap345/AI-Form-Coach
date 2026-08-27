from __future__ import annotations

from test_support import (
    SEQUENCE_LENGTH,
    TXT_SAMPLE_NAME,
    Config,
    FormClassifier,
    FormClassifierInference,
    ModelLoadError,
    Path,
    process_file,
    sha256_file,
    tempfile,
    torch,
    unittest,
    write_uiprmd_txt_fixture,
)

_unsafe_pickle_loaded = False


def _mark_unsafe_pickle_loaded() -> None:
    """Benign payload used to prove the safe loader never executes pickle code."""
    global _unsafe_pickle_loaded
    _unsafe_pickle_loaded = True


class _UnsafePicklePayload:
    def __reduce__(self):
        return _mark_unsafe_pickle_loaded, ()


class TestInference(unittest.TestCase):
    def test_model_input_shape(self):
        model = FormClassifier()
        x = torch.zeros((2, SEQUENCE_LENGTH, 12, 4), dtype=torch.float32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_inference_output(self):
        model = FormClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            inference = FormClassifierInference(
                path,
                device="cpu",
                expected_sha256=sha256_file(path),
            )
            sample_path = write_uiprmd_txt_fixture(Path(tmp) / TXT_SAMPLE_NAME)
            sample = process_file(sample_path, Config())
            result = inference.predict(sample)
            self.assertIn(result["label"], {"incorrect", "correct"})
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)

    def test_inference_rejects_unexpected_state_dict(self):
        model = FormClassifier()
        state_dict = model.state_dict()
        state_dict.pop("head.1.bias")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(state_dict, path)
            with self.assertRaisesRegex(ModelLoadError, "keys do not match"):
                FormClassifierInference(path, device="cpu", expected_sha256=sha256_file(path))

    def test_inference_requires_expected_sha256(self):
        model = FormClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)

            with self.assertRaisesRegex(ModelLoadError, "SHA-256 is required"):
                FormClassifierInference(path, device="cpu")

    def test_inference_rejects_arbitrary_pickle_without_executing_it(self):
        global _unsafe_pickle_loaded
        _unsafe_pickle_loaded = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.pt"
            torch.save(_UnsafePicklePayload(), path)

            with self.assertRaisesRegex(ModelLoadError, "could not safely load"):
                FormClassifierInference(path, device="cpu", expected_sha256=sha256_file(path))

        self.assertFalse(_unsafe_pickle_loaded)

    def test_inference_rejects_wrong_sha256(self):
        model = FormClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ModelLoadError, "SHA-256 mismatch"):
                FormClassifierInference(path, device="cpu", expected_sha256="0" * 64)
