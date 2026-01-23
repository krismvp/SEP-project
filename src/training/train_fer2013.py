import os
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.data_loader import make_fer_loaders
from src.models.resnet_small import ResNet18


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_num_classes(dataset) -> int:
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    raise ValueError("Unable to infer number of classes from dataset.")


def infer_in_channels(dataset) -> int:
    sample, _ = dataset[0]
    if hasattr(sample, "shape"):
        return sample.shape[0]
    raise ValueError("Unable to infer input channels from dataset sample.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pretrained_backbone(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a state_dict key.")
    conv1_weight = state.get("conv1.weight")
    if isinstance(conv1_weight, torch.Tensor):
        in_channels = _get_conv1_in_channels(model)
        state = _adapt_conv1(state, in_channels)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"Missing keys in checkpoint (expected for fc): {missing}")


def _adapt_conv1(state: Dict[str, torch.Tensor], in_channels: int) -> Dict[str, torch.Tensor]:
    weight = state["conv1.weight"]
    if weight.shape[1] == in_channels:
        return state
    if in_channels == 1 and weight.shape[1] == 3:
        state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
        return state
    if in_channels == 3 and weight.shape[1] == 1:
        state["conv1.weight"] = weight.repeat(1, 3, 1, 1) / 3.0
        return state
    return state


def _get_conv1_in_channels(model: nn.Module) -> int:
    conv1 = getattr(model, "conv1", None)
    if isinstance(conv1, nn.Conv2d):
        return int(conv1.in_channels)
    raise ValueError("Model does not have a Conv2d conv1 layer.")


def _freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def _unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return optim.Adam(param_groups, weight_decay=weight_decay)


def train_fer2013(
    data_path: str,
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

    train_loader, val_loader = make_fer_loaders(
        data_path,
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

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }


__all__ = ["train_fer2013"]
