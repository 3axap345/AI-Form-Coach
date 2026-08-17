"""
Single source of truth for the AI Form Coach skeleton representation.

Canonical sample shape:
    [T=60, V=12, C=4]

Coordinate system:
    Body-centered, normalized, image-style axes:
    - x: positive to the subject's right in the processed frame
    - y: positive downward
    - z: relative depth, centered at mid-hip and scaled like x/y
    - visibility: confidence in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple


SEQUENCE_LENGTH = 60

CANONICAL_JOINTS: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

CANONICAL_CHANNELS: Tuple[str, ...] = ("x", "y", "z", "visibility")

JOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
CHANNEL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_CHANNELS)}

NUM_JOINTS = len(CANONICAL_JOINTS)
NUM_CHANNELS = len(CANONICAL_CHANNELS)

COORD_X = CHANNEL_INDEX["x"]
COORD_Y = CHANNEL_INDEX["y"]
COORD_Z = CHANNEL_INDEX["z"]
COORD_VIS = CHANNEL_INDEX["visibility"]

COORDINATE_SYSTEM = "body_centered_image_y_down"
Z_NORMALIZATION_POLICY = "mid_hip_relative_torso_scale"
SCALE_POLICY = "first_frame_mid_shoulder_to_mid_hip_xy"

CANONICAL_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (3, 5),
    (5, 7),
    (7, 9),
    (9, 11),
)

# MediaPipe Pose 33-landmark indices used by the live collector.
MEDIAPIPE_LANDMARK_INDICES: Dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# UI-PRMD Kinect 22-joint order used by RehabExerAssess connectivity.
UIPRMD_JOINTS: Tuple[str, ...] = (
    "root",
    "lower_spine",
    "upper_spine",
    "neck",
    "head",
    "head_top",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)

UIPRMD_TO_CANONICAL: Dict[str, int] = {
    "left_shoulder": 6,
    "right_shoulder": 10,
    "left_hip": 14,
    "right_hip": 18,
    "left_knee": 15,
    "right_knee": 19,
    "left_ankle": 16,
    "right_ankle": 20,
    "left_heel": 17,
    "right_heel": 21,
    "left_foot_index": 17,
    "right_foot_index": 21,
}

FOOT_POLICY = (
    "UI-PRMD has one foot joint per side; heel and foot_index are duplicated "
    "for compatibility with the live MediaPipe 12-joint contract."
)


class Label(Enum):
    INCORRECT = 0
    CORRECT = 1


LABEL_TO_NAME = {
    Label.INCORRECT.value: "incorrect",
    Label.CORRECT.value: "correct",
}
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}

# Local UI-PRMD Kinect skeleton files use C01/C02. The project directory already
# separates C01 as Correct and C02 as Incorrect; no local upstream document was
# found, so keep this mapping explicit and centralized.
UIPRMD_C_LABELS = {
    "01": Label.CORRECT.value,
    "02": Label.INCORRECT.value,
}

UIPRMD_FILENAME_RE = re.compile(
    r"^A(?P<activity>\d{2})S(?P<subject>\d{2})E(?P<episode>\d{2})C(?P<class_code>\d{2})$"
)


@dataclass(frozen=True)
class UIPRMDMetadata:
    activity: str
    subject: str
    episode: str
    class_code: str
    label: int


def parse_uiprmd_filename(path: str | Path) -> UIPRMDMetadata:
    stem = Path(path).stem
    match = UIPRMD_FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Invalid UI-PRMD filename: {stem}")
    parts = match.groupdict()
    class_code = parts["class_code"]
    if class_code not in UIPRMD_C_LABELS:
        raise ValueError(f"Unsupported UI-PRMD class code C{class_code}: {stem}")
    return UIPRMDMetadata(
        activity=parts["activity"],
        subject=parts["subject"],
        episode=parts["episode"],
        class_code=class_code,
        label=UIPRMD_C_LABELS[class_code],
    )


def canonical_shape(sequence_length: int = SEQUENCE_LENGTH) -> Tuple[int, int, int]:
    return sequence_length, NUM_JOINTS, NUM_CHANNELS


def validate_joint_names(names: Iterable[str]) -> None:
    unknown = [name for name in names if name not in JOINT_INDEX]
    if unknown:
        raise ValueError(f"Unknown canonical joint names: {unknown}")
