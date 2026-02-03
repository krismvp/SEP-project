import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.training.train_ferplus import train_ferplus


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on FER+ data.")
    parser.add_argument("--data-dir", default="data/ferplus")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/ferplus")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--use-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    args = parser.parse_args()

    if args.use_mtcnn:
        print(
            f"MTCNN: enabled=True (margin={args.mtcnn_margin}, device={args.mtcnn_device})"
        )

    history = train_ferplus(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir,
        patience=args.patience,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pretrained_path=args.pretrained_path,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        use_weighted_loss=not args.no_weighted_loss,
        use_weighted_sampler=args.weighted_sampler,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        augmentation=args.augmentation,
        arch=args.arch,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )
    class_names = history.get("class_names") or []
    if class_names:
        print(f"Class order: {class_names}")

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
