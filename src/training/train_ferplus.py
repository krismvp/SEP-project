import os
from collections.abc import Sized
from typing import Optional, Dict, List, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.data.ferplus_data import make_ferplus_loaders
from src.models.resnet_small import ResNet18
from src.training.train_fer2013 import (
    get_device,
    infer_in_channels,
    infer_num_classes,
    set_seed,
    _build_optimizer,
    _freeze_backbone,
    _load_pretrained_backbone,
    _unfreeze_backbone,
)


def _label_from_sample(sample: object) -> Optional[int]:
    label = getattr(sample, "label", None)
    if label is not None:
        return int(label)
    if isinstance(sample, (list, tuple)) and len(sample) > 1:
        return int(sample[1])
    return None


def _extract_labels(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
        samples = getattr(base, "samples", None)
        if samples is not None:
            labels = []
            for idx in indices:
                label = _label_from_sample(samples[idx])
                if label is not None:
                    labels.append(label)
            return labels
        targets = getattr(base, "targets", None)
        if targets is not None:
            return [int(targets[idx]) for idx in indices]
        labels = []
        for idx in indices:
            label = _label_from_sample(base[idx])
            if label is None:
                raise ValueError("Dataset samples must provide labels.")
            labels.append(label)
        return labels
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        labels = []
        for sample in samples:
            label = _label_from_sample(sample)
            if label is not None:
                labels.append(label)
        return labels
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(t) for t in targets]
    if not isinstance(dataset, Sized):
        raise ValueError("Dataset must implement __len__ for label extraction.")
    sized_dataset = cast(Sized, dataset)
    labels = []
    for idx in range(len(sized_dataset)):
        label = _label_from_sample(dataset[idx])
        if label is None:
            raise ValueError("Dataset samples must provide labels.")
        labels.append(label)
    return labels


def _compute_class_weights(
    labels: List[int], num_classes: int, power: float = 1.0
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    counts = counts.float().clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    if power != 1.0:
        weights = weights.pow(power)
    return weights


def _get_class_names(dataset, num_classes: int) -> list[str]:
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)
    return [str(i) for i in range(num_classes)]


def train_ferplus(
    data_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    val_split: float,
    seed: int,
    output_dir: str,
    patience: int,
    num_workers: int = 4,
    num_channels: Optional[int] = None,
    image_size: int = 64,
    pretrained_path: Optional[str] = None,
    freeze_epochs: int = 0,
    backbone_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    weight_decay: float = 0.0,
    drop_neutral: bool = False,
    drop_contempt: bool = False,
    confusion_matrix: bool = False,
    use_weighted_loss: bool = True,
    use_weighted_sampler: bool = False,
    class_weight_power: float = 0.5,
    label_smoothing: float = 0.0,
    augmentation: str = "basic",
):
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_ferplus_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        num_channels=num_channels or 1,
        image_size=image_size,
        drop_neutral=drop_neutral,
        drop_contempt=drop_contempt,
        augmentation=augmentation,
    )

    num_classes = infer_num_classes(train_loader.dataset)
    in_channels = num_channels if num_channels is not None else infer_in_channels(train_loader.dataset)
    class_names = _get_class_names(train_loader.dataset, num_classes)

    model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    if pretrained_path:
        _load_pretrained_backbone(model, pretrained_path)

    labels = _extract_labels(train_loader.dataset)
    class_weights = None
    if labels:
        class_weights = _compute_class_weights(
            labels, num_classes=num_classes, power=class_weight_power
        )
    if use_weighted_sampler and class_weights is not None:
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

    head_lr = head_lr if head_lr is not None else lr
    if backbone_lr is None:
        backbone_lr = lr if pretrained_path is None else lr * 0.1

    if freeze_epochs > 0 and pretrained_path:
        _freeze_backbone(model)
    optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_since_improve = 0

    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "resnet18_best.pth")

    for epoch in range(epochs):
        if freeze_epochs > 0 and pretrained_path and epoch == freeze_epochs:
            _unfreeze_backbone(model)
            optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay)
            print("Unfroze backbone for full fine-tuning.")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = 100 * correct / max(total, 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | "
            f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improve += 1

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "resnet18_last.pth"))

    if test_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)

    test_loss = None
    test_acc = None
    cm = None
    if test_loader is not None:
        running_test_loss = 0.0
        test_correct = 0
        test_total = 0
        if confusion_matrix:
            cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
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
                if cm is not None:
                    labels_cpu = labels.detach().cpu()
                    preds_cpu = preds.detach().cpu()
                    indices = labels_cpu * num_classes + preds_cpu
                    cm += torch.bincount(
                        indices, minlength=num_classes * num_classes
                    ).reshape(num_classes, num_classes)
        test_loss = running_test_loss / max(test_total, 1)
        test_acc = 100 * test_correct / max(test_total, 1)
        print(f"Test Loss = {test_loss:.4f} | Test Acc = {test_acc:.2f}%")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist() if isinstance(cm, torch.Tensor) else None,
    }


__all__ = ["train_ferplus"]
