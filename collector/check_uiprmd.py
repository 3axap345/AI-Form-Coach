from pathlib import Path

import numpy as np
from preprocessing import orientation_diagnostics

ROOT = Path(__file__).resolve().parent / "dataset" / "squat_uiprmd"


for label in ["correct", "incorrect"]:
    files = sorted((ROOT / label).glob("*.npy"))

    print(f"\n{label.upper()}")
    print("Files:", len(files))

    if not files:
        continue

    sample = np.load(files[0])

    print("Shape:", sample.shape)
    print("dtype:", sample.dtype)
    print("Min:", sample[:, :, :3].min())
    print("Max:", sample[:, :, :3].max())
    print("Visibility:", sample[:, :, 3].min(), sample[:, :, 3].max())
    print("Orientation:", orientation_diagnostics(sample))

    print("First frame:")
    print(sample[0])
