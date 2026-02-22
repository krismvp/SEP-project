import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def main() -> None:
    """Main training function: Parse arguments, run AffectNet training, and plot results."""
    parser = argparse.ArgumentParser(
        description="Train on AffectNet (ImageFolder format)."
    )
    parser.add_argument("--data-dir", default="data/AffectNet")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weighted-sampler", dest="weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.set_defaults(weighted_sampler=True)
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--use-mtcnn", action="store_true", default=True)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument("--output-dir", default="outputs/pretrain/affectnet_best_generalization")
    args = parser.parse_args()

    if args.no_mtcnn:
        args.use_mtcnn = False

    from src.training.train_affectnet import train_affectnet

    if args.use_mtcnn:
        print(
            f"MTCNN: enabled=True (margin={args.mtcnn_margin}, device={args.mtcnn_device})"
        )

    history = train_affectnet(
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=args.augmentation,
        label_smoothing=args.label_smoothing,
        weighted_sampler=args.weighted_sampler,
        no_weighted_loss=args.no_weighted_loss,
        class_weight_power=args.class_weight_power,
        patience=args.patience,
        seed=args.seed,
        val_split=args.val_split,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
        output_dir=args.output_dir,
    )

    epochs = np.array(history["epochs"])
    if epochs.size == 0:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    train_acc = np.array(
        [acc if acc is not None else np.nan for acc in history["train_acc"]]
    )
    if not np.all(np.isnan(train_acc)):
        axes[0].plot(epochs, train_acc, label="Train Acc")
    axes[0].plot(epochs, history["val_acc"], label="Val Acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
