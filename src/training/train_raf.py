import os
from collections.abc import Sized
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.constants.emotions import CANON_6
from src.data.raf_data import make_raf_loaders
from src.models.factory import build_model
from src.training.train_utils import (
    get_device,
    set_seed,
    _compute_class_weights,
    _build_optimizer,
    _extract_labels,
    _load_pretrained_backbone,
)


def _safe_len(obj: object) -> int:
    """Avoid hard failures when loaders expose objects without __len__."""
    return len(obj) if isinstance(obj, Sized) else 0


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    log_interval: int = 0,
    desc: str = "",
    debug_state: Optional[dict] = None,
) -> Tuple[float, float]:
    """Share train/val loop logic to keep both phases behaviorally consistent."""
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        if desc:
            iterator = tqdm(loader, desc=desc, leave=False)
            for step, (images, labels) in enumerate(iterator, start=1):
                if debug_state is not None and not debug_state.get("printed", False):
                    # One-time debug print helps catch input/model mismatches early.
                    print("conv1:", tuple(model.conv1.weight.shape))
                    print("batch:", tuple(images.shape))
                    print("batch_dtype:", images.dtype)
                    print("batch_range:", (float(images.min()), float(images.max())))
                    debug_state["printed"] = True
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                if train and optimizer is not None:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                if log_interval > 0 and step % log_interval == 0:
                    acc = 100 * correct / max(total, 1)
                    iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
        else:
            for step, (images, labels) in enumerate(loader, start=1):
                if debug_state is not None and not debug_state.get("printed", False):
                    print("conv1:", tuple(model.conv1.weight.shape))
                    print("batch:", tuple(images.shape))
                    print("batch_dtype:", images.dtype)
                    print("batch_range:", (float(images.min()), float(images.max())))
                    debug_state["printed"] = True
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                if train and optimizer is not None:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

    loss = running_loss / max(total, 1)
    acc = 100 * correct / max(total, 1)
    return loss, acc


def train_raf(
    data_dir: str,
    pretrained_path: Optional[str],
    output_dir: str,
    lr: float = 3e-4,
    batch_size: int = 32,
    epochs: int = 30,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    patience: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 64,
    log_interval: int = 50,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    image_dir: Optional[str] = None,
    use_weighted_loss: bool = True,
    use_weighted_sampler: bool = True,
    class_weight_power: float = 0.2,
    label_smoothing: float = 0.05,
    augmentation: str = "strong",
    arch: str = "resnet34",
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
) -> Dict[str, List[float]]:
    """Train RAF with optional pretraining and imbalance-aware sampling/loss."""
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes = make_raf_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        image_size=image_size,
        train_csv=train_csv,
        test_csv=test_csv,
        image_dir=image_dir,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )
    train_size = _safe_len(train_loader.dataset)
    val_size = _safe_len(val_loader.dataset) if val_loader is not None else 0
    test_size = _safe_len(test_loader.dataset) if test_loader is not None else 0
    print(
        f"Train samples: {train_size} | Val samples: {val_size} | Test samples: {test_size}"
    )

    model = build_model(arch, num_classes=num_classes, in_channels=1).to(device)
    if pretrained_path:
        _load_pretrained_backbone(model, pretrained_path)

    labels = _extract_labels(train_loader.dataset)
    class_order = list(CANON_6)
    class_weights = None
    if labels:
        class_weights = _compute_class_weights(
            labels, num_classes=num_classes, power=class_weight_power
        )
    if use_weighted_sampler and class_weights is not None:
        # Oversampling keeps rare classes present without discarding common examples.
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
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    if use_weighted_loss and class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if backbone_lr is None:
        # Smaller LR for pretrained backbones helps preserve useful pretrained features.
        backbone_lr = lr if pretrained_path is None else lr * 0.1

    optimizer = _build_optimizer(model, backbone_lr, lr, weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    min_delta = 1e-4
    # Validation loss often moves slightly; patience should ignore tiny noise.
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, f"{arch}_finetune_best.pth")

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
        "class_order": class_order,
    }
    debug_state = {"printed": False}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            log_interval=log_interval,
            desc=f"Train {epoch}/{epochs}",
            debug_state=debug_state,
        )
        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)

        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            val_loss, val_acc = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                train=False,
                log_interval=log_interval,
                desc=f"Val {epoch}/{epochs}",
            )
            history["val_losses"].append(val_loss)
            history["val_accs"].append(val_acc)

        if val_loader is not None:
            print(
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | "
                f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}%"
            )

        if val_loader is not None:
            if val_loss < best_val_loss - min_delta:
                # Loss is used for selection because it is usually less jumpy than accuracy.
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save(
                    model.state_dict(),
                    best_path,
                )
            else:
                epochs_since_improve += 1

            if patience > 0 and epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    torch.save(model.state_dict(), os.path.join(output_dir, f"{arch}_finetune_last.pth"))

    if val_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))

    test_loss = None
    test_acc = None
    if test_loader is not None:
        running_test_loss = 0.0
        test_correct = 0
        test_total = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Test"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()
        test_loss = running_test_loss / max(test_total, 1)
        test_acc = 100 * test_correct / max(test_total, 1)
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


__all__ = ["train_raf"]
