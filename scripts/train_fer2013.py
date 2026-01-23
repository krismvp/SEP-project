import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.train_fer2013 import train_fer2013


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on FER2013 data.")
    parser.add_argument("--data-path", default="data/FER13")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    num_channels = args.num_channels
    if num_channels is None:
        num_channels = 3 if args.pretrained_path else 1

    history = train_fer2013(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir,
        patience=args.patience,
        num_workers=args.num_workers,
        num_channels=num_channels,
        image_size=args.image_size,
        pretrained_path=args.pretrained_path,
        freeze_epochs=args.freeze_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    epochs_ran = len(history["train_losses"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, epochs_ran + 1), history["train_accs"], label="Train Acc")
    axes[0].plot(range(1, epochs_ran + 1), history["val_accs"], label="Val Acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, epochs_ran + 1), history["train_losses"], label="Train Loss")
    axes[1].plot(range(1, epochs_ran + 1), history["val_losses"], label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Best epoch by val accuracy: {history['best_epoch']}")
    print(f"Saved plots to: {plot_path}")


if __name__ == "__main__":
    main()
