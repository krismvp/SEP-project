import argparse
import os
import sys
from collections.abc import Sized
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.constants.emotions import CANON_6
from src.data.ferplus_data import make_ferplus_loaders
from src.data.raf_data import make_raf_loaders
from src.models.factory import build_model
from src.training.train_fer2013 import get_device, set_seed, _build_optimizer


def _unwrap_dataset(dataset: Dataset) -> Dataset:
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _get_class_order(dataset: Dataset) -> list[str]:
    base = _unwrap_dataset(dataset)
    classes = getattr(base, "classes", None)
    if classes:
        return list(classes)
    class_to_idx = getattr(base, "class_to_idx", None)
    if class_to_idx:
        if not all(isinstance(v, int) for v in class_to_idx.values()):
            raise ValueError("class_to_idx values must be int indices.")
        num_classes = len(class_to_idx)
        order: list[Optional[str]] = [None] * num_classes
        for name, idx in class_to_idx.items():
            if idx < 0 or idx >= num_classes:
                raise ValueError("class_to_idx contains out-of-range indices.")
            if order[idx] is not None:
                raise ValueError("class_to_idx contains duplicate indices.")
            order[idx] = name
        if any(item is None for item in order):
            raise ValueError("class_to_idx is incomplete; cannot infer class order.")
        return [item for item in order if item is not None]
    raise ValueError("Dataset has no classes or class_to_idx for label verification.")


def _assert_canon6(dataset: Dataset, name: str) -> None:
    classes = _get_class_order(dataset)
    if classes != list(CANON_6):
        raise ValueError(f"{name} classes mismatch: {classes} vs {list(CANON_6)}")


def _safe_len(obj: object) -> int:
    return len(obj) if isinstance(obj, Sized) else 0


class BalancedConcatSampler(Sampler[int]):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        probs: List[float],
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.lengths = [_safe_len(ds) for ds in datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError("All datasets must have a valid __len__ for sampling.")
        self.offsets = [0]
        for n in self.lengths:
            self.offsets.append(self.offsets[-1] + n)
        probs_tensor = torch.tensor(probs, dtype=torch.double)
        probs_tensor = probs_tensor / probs_tensor.sum()
        self.probs = probs_tensor
        self.num_samples = num_samples or sum(self.lengths)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        ds_ids = torch.multinomial(
            self.probs, self.num_samples, replacement=True, generator=gen
        )
        for ds_id in ds_ids.tolist():
            idx = torch.randint(0, self.lengths[ds_id], (1,), generator=gen).item()
            yield self.offsets[ds_id] + idx

    def __len__(self) -> int:
        return self.num_samples


def _mixup_batch(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[indices]
    targets_a = targets
    targets_b = targets[indices]
    return mixed_inputs, targets_a, targets_b, lam


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / max(total, 1)

def _eval_loss_acc(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a mixed-domain FER+ + RAF model with balanced sampling."
    )
    parser.add_argument("--fer-data-dir", default="data/ferplus")
    parser.add_argument("--raf-data-dir", default="data/RAF-DB")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--use-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--domain-probs",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        metavar=("FER", "RAF"),
    )
    parser.add_argument(
        "--selection-metric", choices=["avg", "min"], default="avg"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mixed/ferplus_raf_resnet34_mtcnn_mixup",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    if args.use_mtcnn:
        print(
            f"MTCNN: enabled=True (margin={args.mtcnn_margin}, device={args.mtcnn_device})"
        )

    fer_train, fer_val, _ = make_ferplus_loaders(
        data_dir=args.fer_data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=args.augmentation,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )
    if args.augmentation != "strong":
        print(
            "Warning: RAF loader uses its default augmentation; "
            "FER+ will use the requested augmentation."
        )

    raf_train, raf_val, _, _ = make_raf_loaders(
        data_dir=args.raf_data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )

    if fer_val is None or raf_val is None:
        raise ValueError("Both FER+ and RAF must have val splits for mixed training.")

    _assert_canon6(fer_train.dataset, "FER+")
    _assert_canon6(raf_train.dataset, "RAF")

    print(
        "FER+ samples — "
        f"Train: {_safe_len(fer_train.dataset)} | Val: {_safe_len(fer_val.dataset)}"
    )
    print(
        "RAF samples — "
        f"Train: {_safe_len(raf_train.dataset)} | Val: {_safe_len(raf_val.dataset)}"
    )
    print(f"Class order: {list(CANON_6)}")

    mixed_train = ConcatDataset([fer_train.dataset, raf_train.dataset])
    sampler = BalancedConcatSampler(
        datasets=[fer_train.dataset, raf_train.dataset],
        probs=args.domain_probs,
        seed=args.seed,
    )
    mixed_loader = DataLoader(
        mixed_train,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(CANON_6)
    model = build_model(args.arch, num_classes=num_classes, in_channels=1).to(device)
    optimizer = _build_optimizer(model, args.lr, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = float("-inf")
    best_score_for_patience = float("-inf")
    best_path = os.path.join(args.output_dir, f"{args.arch}_best.pth")
    best_epoch = 0
    epochs_since_improve = 0
    min_delta = 0.05

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss_fer": [],
        "val_loss_raf": [],
        "val_acc_fer": [],
        "val_acc_raf": [],
        "score": [],
    }

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(mixed_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.mixup:
                # MixUp is disabled for explainability; Grad-CAM runs on original images.
                mixed_imgs, y_a, y_b, lam = _mixup_batch(
                    imgs, labels, args.mixup_alpha
                )
                outputs = model(mixed_imgs)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(
                    outputs, y_b
                )
                total += y_a.size(0)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1) if not args.mixup else None

        val_loss_fer, val_acc_fer = _eval_loss_acc(model, fer_val, device, criterion)
        val_loss_raf, val_acc_raf = _eval_loss_acc(model, raf_val, device, criterion)
        val_loss_avg = 0.5 * (val_loss_fer + val_loss_raf)
        if args.selection_metric == "min":
            score = min(val_acc_fer, val_acc_raf)
        else:
            score = 0.5 * (val_acc_fer + val_acc_raf)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss_fer"].append(val_loss_fer)
        history["val_loss_raf"].append(val_loss_raf)
        history["val_acc_fer"].append(val_acc_fer)
        history["val_acc_raf"].append(val_acc_raf)
        history["score"].append(score)

        if train_acc is None:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.4f} | "
                f"Val FER+ Loss = {val_loss_fer:.4f} | Val FER+ Acc = {val_acc_fer:.2f}% | "
                f"Val RAF Loss = {val_loss_raf:.4f} | Val RAF Acc = {val_acc_raf:.2f}% | "
                f"Score = {score:.2f} | Val Loss Avg = {val_loss_avg:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | "
                f"Val FER+ Loss = {val_loss_fer:.4f} | Val FER+ Acc = {val_acc_fer:.2f}% | "
                f"Val RAF Loss = {val_loss_raf:.4f} | Val RAF Acc = {val_acc_raf:.2f}% | "
                f"Score = {score:.2f} | Val Loss Avg = {val_loss_avg:.4f}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if score > best_score_for_patience + min_delta:
            best_score_for_patience = score
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        scheduler.step()

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.arch}_last.pth"))

    epochs = np.array(history["epochs"])
    if epochs.size > 0:
        plt.figure(figsize=(7, 4))
        train_acc = np.array(
            [acc if acc is not None else np.nan for acc in history["train_acc"]]
        )
        if not np.all(np.isnan(train_acc)):
            plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, history["val_acc_fer"], label="Val FER+ Acc")
        plt.plot(epochs, history["val_acc_raf"], label="Val RAF Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Mixed Training Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "accuracy_curves.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss_fer"], label="Val FER+ Loss")
        plt.plot(epochs, history["val_loss_raf"], label="Val RAF Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mixed Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
