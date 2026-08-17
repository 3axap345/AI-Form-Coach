from pathlib import Path
import numpy as np

from canonical import (
    CANONICAL_JOINTS,
    FOOT_POLICY,
    NUM_JOINTS,
    UIPRMD_TO_CANONICAL,
    parse_uiprmd_filename,
)
from config import Config
from preprocessing import build_sample, assert_canonical_orientation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_uiprmd_skeleton_txt(path: Path) -> np.ndarray:
    """
    Загружает UI-PRMD .txt, созданный convert_uiprmd.py.

    Формат:
        [T, 22, 3]

    Возвращает:
        [T, 22, 3] raw Kinect world coordinates (Y-up).
    """
    data = np.loadtxt(path, dtype=np.float32)

    # В convert_uiprmd.py данные сохраняются flattened:
    # [T, 22 * 3]
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 22 * 3:
        raise ValueError(
            f"{path.name}: expected 66 columns, got {data.shape[1]}"
        )

    return data.reshape(data.shape[0], 22, 3)


def uiprmd_to_canonical_frames(skeleton: np.ndarray) -> np.ndarray:
    """
    UI-PRMD Kinect [T,22,3] -> canonical live frames [T,12,4].

    Kinect absolute skeleton coordinates are Y-up. Canonical coordinates use
    image-style Y-down, so only Y is inverted before shared body-centered
    preprocessing. Z remains in Kinect depth units here; build_sample makes it
    mid-hip relative and torso-normalized.
    """
    if skeleton.ndim != 3 or skeleton.shape[1:] != (22, 3):
        raise ValueError(f"expected [T,22,3], got {skeleton.shape}")

    result = np.zeros(
        (skeleton.shape[0], NUM_JOINTS, 4),
        dtype=np.float32,
    )

    for our_idx, name in enumerate(CANONICAL_JOINTS):
        ui_idx = UIPRMD_TO_CANONICAL[name]
        result[:, our_idx, 0:3] = skeleton[:, ui_idx, :]

    # UI-PRMD Kinect world space is Y-up; canonical live representation is Y-down.
    result[:, :, 1] *= -1.0
    result[:, :, 3] = 1.0

    return result


def load_uiprmd_skeleton(path: Path) -> np.ndarray:
    """Backward-compatible helper returning canonical live frames [T,12,4]."""
    return uiprmd_to_canonical_frames(load_uiprmd_skeleton_txt(path))


def process_file(path: Path, cfg: Config) -> np.ndarray:
    """
    UI-PRMD файл -> готовый sample:

        [T, 22, 3]
              ↓
        [T, 12, 4]
              ↓
        resample
              ↓
        normalize
              ↓
        [60, 12, 4]
    """

    parse_uiprmd_filename(path)
    frames = load_uiprmd_skeleton(path)

    if len(frames) < 2:
        raise ValueError(
            f"{path.name}: too few frames ({len(frames)})"
        )

    sample = build_sample(
        [frames[i] for i in range(len(frames))],
        cfg,
    )
    assert_canonical_orientation(sample)
    return sample


def main():

    cfg = Config()

    source_root = PROJECT_ROOT / "collector" / "RehabExerAssess-main" / "data" / "UI-PRMD"

    output_root = PROJECT_ROOT / "collector" / "dataset" / "squat_uiprmd"

    print(FOOT_POLICY)

    splits = {
        "correct": source_root / "Correct" / "Kinect" / "Skeletons",
        "incorrect": source_root / "Incorrect" / "Kinect" / "Skeletons",
    }

    for label, source_dir in splits.items():

        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(source_dir.glob("*.txt"))

        print(f"\n{label.upper()}: {len(files)} files")

        success = 0
        failed = 0

        for path in files:

            try:
                sample = process_file(path, cfg)

                output_path = output_dir / (
                    path.stem + ".npy"
                )

                np.save(output_path, sample)

                success += 1

                print(
                    f"{path.name}: "
                    f"{sample.shape} -> {output_path.name}"
                )

            except Exception as e:

                failed += 1

                print(
                    f"ERROR {path.name}: {e}"
                )

        print(
            f"{label}: "
            f"success={success}, failed={failed}"
        )


if __name__ == "__main__":
    main()
