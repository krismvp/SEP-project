import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

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


def train(
    data_path: str,
    batch_size: int,
    epochs: int,
    lr: float,
    val_split: float,
    seed: int,
    output_dir: str,
    patience: int,
) -> None:
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_fer_loaders(
        data_path, batch_size=batch_size, val_split=val_split, seed=seed
    )

    num_classes = infer_num_classes(train_loader.dataset)
    in_channels = infer_in_channels(train_loader.dataset)

    model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_epoch = 0
    epochs_since_improve = 0

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
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
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_running_loss / max(val_total, 1)
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

    epochs_ran = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, epochs_ran + 1), train_accs, label="Train Acc")
    axes[0].plot(range(1, epochs_ran + 1), val_accs, label="Val Acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, epochs_ran + 1), train_losses, label="Train Loss")
    axes[1].plot(range(1, epochs_ran + 1), val_losses, label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best epoch by val accuracy: {best_epoch}")
    print(f"Saved plots to: {plot_path}")
    print(f"Saved metrics to: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on FER data.")
    parser.add_argument("--data-path", default="data/FER13")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
