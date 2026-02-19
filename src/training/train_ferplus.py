import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.ferplus_data import make_ferplus_loaders
from src.models.factory import build_model
from src.training.train_utils import (
    get_device,
    infer_num_classes,
    set_seed,
    _compute_class_weights,
    _build_optimizer,
    _extract_labels,
    _load_pretrained_backbone,
)


def _get_class_names(dataset, num_classes: int) -> list[str]:
    """Return stable class names for logging, even for wrapped datasets."""
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)
    return [str(i) for i in range(num_classes)]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = "",
    debug_state: Optional[dict] = None,
) -> tuple[float, float]:
    """Use one loop for train and val so both phases stay behaviorally aligned."""
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
            if debug_state is not None and not debug_state.get("printed", False):
                # Printed once to catch shape/range issues early without noisy logs each step.
                print("conv1:", tuple(model.conv1.weight.shape))
                print("batch:", tuple(imgs.shape))
                print("batch_dtype:", imgs.dtype)
                print("batch_range:", (float(imgs.min()), float(imgs.max())))
                debug_state["printed"] = True
            imgs, labels = imgs.to(device), labels.to(device)

            if train and optimizer is not None:
                optimizer.zero_grad()
            outputs = model(imgs)
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


def train_ferplus(
    data_dir: str,
    pretrained_path: Optional[str],
    output_dir: str,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 25,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    patience: int = 5,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = 64,
    log_interval: int = 50,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    image_dir: Optional[str] = None,
    use_weighted_loss: bool = True,
    use_weighted_sampler: bool = False,
    class_weight_power: float = 0.5,
    label_smoothing: float = 0.05,
    augmentation: str = "basic",
    arch: str = "resnet18",
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
):
    """Train FER+ with optional pretraining and class-imbalance controls."""
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_ferplus_loaders(
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
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    test_size = len(test_loader.dataset) if test_loader is not None else 0
    print(f"FER+ samples - Train: {train_size} | Val: {val_size} | Test: {test_size}")

    num_classes = infer_num_classes(train_loader.dataset)
    in_channels = 1
    class_names = _get_class_names(train_loader.dataset, num_classes)

    model = build_model(arch, num_classes=num_classes, in_channels=in_channels).to(device)
    if pretrained_path:
        _load_pretrained_backbone(model, pretrained_path)

    labels = _extract_labels(train_loader.dataset)
    class_weights = None
    if labels:
        class_weights = _compute_class_weights(
            labels, num_classes=num_classes, power=class_weight_power
        )
    if use_weighted_sampler and class_weights is not None:
        # Oversampling avoids throwing away rare classes while keeping batch size unchanged.
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
        # Lower backbone LR after loading pretrained features to avoid destroying them.
        backbone_lr = lr if pretrained_path is None else lr * 0.1

    optimizer = _build_optimizer(model, backbone_lr, lr, weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    min_delta = 1e-4
    # FER+ val loss tends to move in small steps; a small threshold keeps early stop stable.
    printed_debug = False

    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, f"{arch}_best.pth")

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            desc=f"Epoch {epoch+1}/{epochs} [train]",
            debug_state={"printed": printed_debug},
        )
        printed_debug = True
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            train=False,
            desc=f"Epoch {epoch+1}/{epochs} [val]",
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | "
            f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.2f}%"
        )

        if val_loss < best_val_loss - min_delta:
            # Loss is used for model selection because it is smoother than accuracy.
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improve += 1

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    torch.save(model.state_dict(), os.path.join(output_dir, f"{arch}_last.pth"))

    if test_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)

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
    }


__all__ = ["train_ferplus"]
