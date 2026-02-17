import os
from collections.abc import Sized
from typing import Any, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

from src.constants.emotions import CANON_6
from src.data.ferplus_data import make_ferplus_loaders
from src.data.raf_data import make_raf_loaders
from src.models.factory import build_model
from src.training.train_utils import (
    get_device,
    set_seed,
    _build_optimizer,
    _load_pretrained_backbone,
    _extract_labels,
    _compute_class_weights,
)


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
        sample_weights: Optional[List[List[float]]] = None,
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
        self.sample_weights = None
        if sample_weights is not None:
            if len(sample_weights) != len(self.lengths):
                raise ValueError("sample_weights must match number of datasets.")
            weights_by_ds: List[torch.Tensor] = []
            for weights, length in zip(sample_weights, self.lengths):
                if len(weights) != length:
                    raise ValueError("sample_weights length mismatch for a dataset.")
                weight_tensor = torch.tensor(weights, dtype=torch.double)
                if torch.any(weight_tensor < 0):
                    raise ValueError("sample_weights must be non-negative.")
                if weight_tensor.sum() <= 0:
                    raise ValueError("sample_weights must sum to > 0.")
                weights_by_ds.append(weight_tensor)
            self.sample_weights = weights_by_ds
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
            if self.sample_weights is None:
                idx = torch.randint(0, self.lengths[ds_id], (1,), generator=gen).item()
            else:
                weights = self.sample_weights[ds_id]
                idx = torch.multinomial(weights, 1, replacement=True, generator=gen).item()
            yield self.offsets[ds_id] + idx

    def __len__(self) -> int:
        return self.num_samples


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


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = "",
) -> tuple[float, Optional[float]]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        iterator = tqdm(loader, desc=desc) if desc else loader
        for imgs, labels in iterator:
            imgs, labels = imgs.to(device), labels.to(device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)

    loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return loss, acc


def train_mixed_ferplus_raf(
    fer_data_dir: str,
    raf_data_dir: str,
    arch: str = "resnet18",
    epochs: int = 25,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 64,
    augmentation: str = "strong",
    lr: float = 1e-3,
    pretrained_path: Optional[str] = None,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_weighted_sampler: bool = False,
    use_weighted_loss: bool = True,
    class_weight_power: float = 0.5,
    patience: int = 5,
    seed: int = 42,
    val_split: float = 0.1,
    use_mtcnn: bool = True,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
    domain_probs: Optional[List[float]] = None,
    selection_metric: str = "avg",
    output_dir: str = "outputs/mixed/ferplus_raf_resnet34_mtcnn",
):
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    if domain_probs is None:
        domain_probs = [0.5, 0.5]
    if len(domain_probs) != 2:
        raise ValueError("domain_probs must contain two probabilities: [FER, RAF].")

    fer_train, fer_val, _ = make_ferplus_loaders(
        data_dir=fer_data_dir,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        image_size=image_size,
        augmentation=augmentation,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )
    if augmentation != "strong":
        print(
            "Warning: RAF loader uses its default augmentation; "
            "FER+ will use the requested augmentation."
        )

    raf_train, raf_val, _, _ = make_raf_loaders(
        data_dir=raf_data_dir,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        image_size=image_size,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
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
    num_classes = len(CANON_6)
    use_class_weight_power = class_weight_power > 0

    class_weights = None
    labels_fer: Optional[List[int]] = None
    labels_raf: Optional[List[int]] = None
    if use_class_weight_power and (use_weighted_sampler or use_weighted_loss):
        labels_fer = _extract_labels(fer_train.dataset)
        labels_raf = _extract_labels(raf_train.dataset)
        labels_all = labels_fer + labels_raf
        if labels_all:
            class_weights = _compute_class_weights(
                labels_all, num_classes=num_classes, power=class_weight_power
            )

    sample_weights = None
    if use_weighted_sampler:
        if class_weights is None:
            print(
                "Warning: weighted sampler requested but class weights are disabled. "
                "Set --class-weight-power > 0 to enable."
            )
        else:
            if labels_fer is None or labels_raf is None:
                raise ValueError("Sample labels missing for weighted sampling.")
            sample_weights = [
                [class_weights[label].item() for label in labels_fer],
                [class_weights[label].item() for label in labels_raf],
            ]

    sampler = BalancedConcatSampler(
        datasets=[fer_train.dataset, raf_train.dataset],
        probs=domain_probs,
        seed=seed,
        sample_weights=sample_weights,
    )
    mixed_loader = DataLoader(
        mixed_train,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(arch, num_classes=num_classes, in_channels=1).to(device)
    if pretrained_path:
        _load_pretrained_backbone(model, pretrained_path)
    if backbone_lr is None:
        backbone_lr = lr if pretrained_path is None else lr * 0.1

    optimizer = _build_optimizer(model, backbone_lr, lr, weight_decay)
    weight_tensor = None
    if use_weighted_loss and class_weights is not None:
        weight_tensor = class_weights.to(device)
    if weight_tensor is not None:
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    os.makedirs(output_dir, exist_ok=True)
    best_score = float("-inf")
    best_score_for_patience = float("-inf")
    best_path = os.path.join(output_dir, f"{arch}_best.pth")
    best_epoch = 0
    epochs_since_improve = 0
    min_delta = 0.05

    history: dict[str, Any] = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss_fer": [],
        "val_loss_raf": [],
        "val_acc_fer": [],
        "val_acc_raf": [],
        "score": [],
    }

    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        train_loss, train_acc = _run_epoch(
            model,
            mixed_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            desc=f"Epoch {epoch}/{epochs} [train]",
        )

        val_loss_fer, val_acc_fer = _eval_loss_acc(model, fer_val, device, criterion)
        val_loss_raf, val_acc_raf = _eval_loss_acc(model, raf_val, device, criterion)
        val_loss_avg = 0.5 * (val_loss_fer + val_loss_raf)
        if selection_metric == "min":
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

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(output_dir, f"{arch}_last.pth"))

    history.update(
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "class_names": list(CANON_6),
        }
    )
    return history


__all__ = ["train_mixed_ferplus_raf", "BalancedConcatSampler"]
