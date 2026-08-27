from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from canonical import (
    CANONICAL_CHANNELS,
    CANONICAL_JOINTS,
    COORDINATE_SYSTEM,
    LABEL_TO_NAME,
    SEQUENCE_LENGTH,
    Z_NORMALIZATION_POLICY,
)
from config import Config
from dataset_split import (
    discover_uiprmd_txt,
    load_split_manifest,
    save_split_manifest,
    split_stats,
    subject_safe_split,
)
from form_model import FormClassifier
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from uiprmd_adapter import process_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CanonicalSkeletonDataset(Dataset):
    def __init__(self, records: list[dict], cfg: Config):
        self.samples = []
        self.labels = []
        for record in records:
            sample = process_file(Path(record["path"]), cfg)
            self.samples.append(sample.astype(np.float32))
            self.labels.append(int(record["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.samples[index]), torch.tensor(self.labels[index]).long()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            logits = model(samples)
            losses.append(float(loss_fn(logits, labels).item()))
            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def build_or_load_split(args) -> dict:
    if args.split_manifest.exists():
        return load_split_manifest(args.split_manifest)

    records = discover_uiprmd_txt(args.source_root, activity=args.activity)
    split = subject_safe_split(records, test_subjects=args.test_subjects)
    save_split_manifest(split, args.split_manifest)
    return split


def train(args) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = Config(sequence_length=SEQUENCE_LENGTH)

    split = build_or_load_split(args)
    train_dataset = CanonicalSkeletonDataset(split["train"], cfg)
    test_dataset = CanonicalSkeletonDataset(split["test"], cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FormClassifier(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for samples, labels in train_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(samples)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # The inference artifact intentionally contains weights only.  Keep
            # labels, canonical-format metadata, and metrics in JSON below.
            torch.save(model.state_dict(), best_model_path)

        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1']:.3f}"
        )

    result = {
        "best_model": str(best_model_path),
        "device": str(device),
        "split_stats": split_stats(split),
        "history": history,
        "best_f1": best_f1,
        "label_convention": LABEL_TO_NAME,
    }

    with (args.output_dir / "training_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            vars(args) | {"device": str(device)}, f, ensure_ascii=False, indent=2, default=str
        )
    with (args.output_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_name": LABEL_TO_NAME,
                "canonical": {
                    "sequence_length": SEQUENCE_LENGTH,
                    "joints": CANONICAL_JOINTS,
                    "channels": CANONICAL_CHANNELS,
                    "coordinate_system": COORDINATE_SYSTEM,
                    "z_normalization_policy": Z_NORMALIZATION_POLICY,
                },
                "best_f1": best_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        default=PROJECT_ROOT / "collector" / "RehabExerAssess-main" / "data" / "UI-PRMD",
    )
    parser.add_argument("--activity", default="01")
    parser.add_argument("--test_subjects", nargs="+", default=["08", "09", "10"])
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=PROJECT_ROOT / "collector" / "models" / "squat_binary" / "split_manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "collector" / "models" / "squat_binary",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
