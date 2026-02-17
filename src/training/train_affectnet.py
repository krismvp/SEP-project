from __future__ import annotations

import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.constants.emotions import CANON_6
from src.data.affectnet_data import make_affectnet_loaders
from src.models.factory import build_model
from src.training.train_utils import (
    get_device,
    set_seed,
    _build_optimizer,
    _compute_class_weights,
    _extract_labels,
)


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


def train_affectnet(
    data_dir: str,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    image_size: int,
    augmentation: str,
    label_smoothing: float,
    weighted_sampler: bool,
    no_weighted_loss: bool,
    class_weight_power: float,
    patience: int,
    seed: int,
    val_split: float,
    use_mtcnn: bool,
    mtcnn_margin: float,
    mtcnn_device: str | None,
    output_dir: str,
) -> Dict[str, List[float]]:
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_affectnet_loaders(
        data_dir=data_dir,
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
    if val_loader is None:
        raise ValueError("val_split must be > 0 for AffectNet training.")

    num_classes = len(CANON_6)
    use_class_weight_power = class_weight_power > 0

    class_weights = None
    labels = None
    if use_class_weight_power and (weighted_sampler or not no_weighted_loss):
        labels = _extract_labels(train_loader.dataset)
        if labels:
            class_weights = _compute_class_weights(
                labels, num_classes=num_classes, power=class_weight_power
            )

    if weighted_sampler:
        if class_weights is None:
            print(
                "Warning: weighted sampler requested but class weights are disabled. "
                "Set --class-weight-power > 0 to enable."
            )
        else:
            if labels is None:
                raise ValueError("Sample labels missing for weighted sampling.")
            sample_weights = [class_weights[label].item() for label in labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    model = build_model(arch, num_classes=num_classes, in_channels=1).to(device)

    optimizer = _build_optimizer(model, lr, lr, weight_decay)
    weight_tensor = None
    if not no_weighted_loss and class_weights is not None:
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
    best_val_acc = -1.0
    best_val_acc_for_patience = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    min_delta = 0.05
    best_path = os.path.join(output_dir, f"{arch}_best.pth")

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            desc=f"Epoch {epoch}/{epochs} [train]",
        )
        val_loss, val_acc = _eval_loss_acc(model, val_loader, device, criterion)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | "
            f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if val_acc > best_val_acc_for_patience + min_delta:
            best_val_acc_for_patience = val_acc
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        scheduler.step()

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(output_dir, f"{arch}_last.pth"))

    test_loss = None
    test_acc = None
    if test_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        test_loss, test_acc = _eval_loss_acc(model, test_loader, device, criterion)
        print(f"Test Loss = {test_loss:.4f} | Test Acc = {test_acc:.2f}%")

    history.update(
        {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "class_names": list(CANON_6),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )
    return history


__all__ = ["train_affectnet"]
