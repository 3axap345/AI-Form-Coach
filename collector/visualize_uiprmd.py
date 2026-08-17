from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import orientation_diagnostics
from canonical import CANONICAL_CONNECTIONS, CANONICAL_JOINTS


ROOT = Path(__file__).resolve().parent / "dataset" / "squat_uiprmd"

JOINT_NAMES = list(CANONICAL_JOINTS)
CONNECTIONS = list(CANONICAL_CONNECTIONS)


def draw_frame(frame, title):
    plt.figure(figsize=(8, 8))

    x = frame[:, 0]
    y = frame[:, 1]

    for a, b in CONNECTIONS:
        plt.plot(
            [x[a], x[b]],
            [y[a], y[b]],
            marker="o",
        )

    plt.scatter(x, y)

    for i, name in enumerate(JOINT_NAMES):
        plt.text(x[i], y[i], f" {i}:{name}")

    # Canonical coordinates use image-style Y-down.
    plt.gca().invert_yaxis()

    plt.axis("equal")
    plt.grid(True)
    plt.title(title)

    plt.show()


def main():

    files = sorted((ROOT / "correct").glob("*.npy"))

    if not files:
        print("No files found")
        return

    # Берём первый correct sample
    path = files[0]

    sample = np.load(path)

    print("File:", path)
    print("Shape:", sample.shape)
    print("Orientation diagnostics:", orientation_diagnostics(sample))

    # Первый кадр
    draw_frame(
        sample[0],
        f"UI-PRMD Correct — frame 0 — {path.name}",
    )

    # Средний кадр
    draw_frame(
        sample[len(sample) // 2],
        f"UI-PRMD Correct — middle frame — {path.name}",
    )

    # Последний кадр
    draw_frame(
        sample[-1],
        f"UI-PRMD Correct — last frame — {path.name}",
    )


if __name__ == "__main__":
    main()
