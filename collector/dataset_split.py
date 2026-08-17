"""
Dataset discovery, labels, and subject-safe splits for canonical squat data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from canonical import LABEL_TO_NAME, NAME_TO_LABEL, parse_uiprmd_filename


@dataclass(frozen=True)
class SampleRecord:
    path: str
    activity: str
    subject: str
    episode: str
    class_code: str
    label: int
    label_name: str


def discover_uiprmd_txt(source_root: Path, activity: str = "01") -> List[SampleRecord]:
    skeleton_dirs = [
        source_root / "Correct" / "Kinect" / "Skeletons",
        source_root / "Incorrect" / "Kinect" / "Skeletons",
    ]
    records: List[SampleRecord] = []
    for directory in skeleton_dirs:
        for path in sorted(directory.glob(f"A{activity}*.txt")):
            meta = parse_uiprmd_filename(path)
            records.append(
                SampleRecord(
                    path=str(path),
                    activity=meta.activity,
                    subject=meta.subject,
                    episode=meta.episode,
                    class_code=meta.class_code,
                    label=meta.label,
                    label_name=LABEL_TO_NAME[meta.label],
                )
            )
    return sorted(records, key=lambda item: Path(item.path).name)


def subject_safe_split(
    records: Sequence[SampleRecord],
    test_subjects: Iterable[str] = ("08", "09", "10"),
) -> dict:
    test_subject_set = set(test_subjects)
    train = [record for record in records if record.subject not in test_subject_set]
    test = [record for record in records if record.subject in test_subject_set]

    train_subjects = {record.subject for record in train}
    test_subjects_found = {record.subject for record in test}
    overlap = train_subjects.intersection(test_subjects_found)
    if overlap:
        raise ValueError(f"Subject leakage between train/test: {sorted(overlap)}")

    return {
        "label_convention": {
            str(label): name for label, name in sorted(LABEL_TO_NAME.items())
        },
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects_found),
        "train": [asdict(record) for record in train],
        "test": [asdict(record) for record in test],
    }


def split_stats(split: dict) -> dict:
    stats = {}
    for split_name in ("train", "test"):
        rows = split[split_name]
        counts = {name: 0 for name in NAME_TO_LABEL}
        for row in rows:
            counts[row["label_name"]] += 1
        stats[split_name] = {
            "total": len(rows),
            "subjects": sorted({row["subject"] for row in rows}),
            "labels": counts,
        }
    return stats


def save_split_manifest(split: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(split)
    payload["stats"] = split_stats(split)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_split_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
