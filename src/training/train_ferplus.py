import os
from typing import Optional, Dict

import torch
import torch.nn as nn
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
    )

    num_classes = infer_num_classes(train_loader.dataset)
    in_channels = num_channels if num_channels is not None else infer_in_channels(train_loader.dataset)

    model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    if pretrained_path:
        _load_pretrained_backbone(model, pretrained_path)
    criterion = nn.CrossEntropyLoss()

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
            torch.save(model.state_dict(), os.path.join(output_dir, "resnet18_best.pth"))
        else:
            epochs_since_improve += 1

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "resnet18_last.pth"))

    if test_loader is not None:
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Test"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()
        test_loss = test_loss / max(test_total, 1)
        test_acc = 100 * test_correct / max(test_total, 1)
        print(f"Test Loss = {test_loss:.4f} | Test Acc = {test_acc:.2f}%")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }


__all__ = ["train_ferplus"]
