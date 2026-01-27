import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.train_raf import train_raf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune ResNet-18 on RAF-DB."
    )
    parser.add_argument("--data-dir", default="data/RAF-DB")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    history = train_raf(
        data_dir=args.data_dir,
        pretrained_path=args.pretrained_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        log_interval=args.log_interval,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        use_weighted_loss=not args.no_weighted_loss,
        use_weighted_sampler=args.weighted_sampler,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
    )
    class_order = history.get("class_order") or []
    if class_order:
        print(f"Class order: {class_order}")

    epochs_ran = len(history["train_losses"])
    if epochs_ran == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, epochs_ran + 1), history["train_accs"], label="Train Acc")
    if history["val_accs"]:
        axes[0].plot(range(1, epochs_ran + 1), history["val_accs"], label="Val Acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, epochs_ran + 1), history["train_losses"], label="Train Loss")
    if history["val_losses"]:
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

    print(f"Saved plots to: {plot_path}")


if __name__ == "__main__":
    main()
