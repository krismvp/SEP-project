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

from src.training.train_mixed_ferplus_raf import train_mixed_ferplus_raf




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a mixed-domain FER+ + RAF model with balanced sampling."
    )
    parser.add_argument("--fer-data-dir", default="data/ferplus")
    parser.add_argument("--raf-data-dir", default="data/RAF-DB")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--use-mtcnn", action="store_true", default=True)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--domain-probs",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        metavar=("FER", "RAF"),
    )
    parser.add_argument("--selection-metric", choices=["avg", "min"], default="avg")
    parser.add_argument(
        "--output-dir",
        default="outputs/mixed/ferplus_raf_resnet34_mtcnn_mixup",
    )
    args = parser.parse_args()
    if args.no_mtcnn:
        args.use_mtcnn = False

    if args.use_mtcnn:
        print(
            f"MTCNN: enabled=True (margin={args.mtcnn_margin}, device={args.mtcnn_device})"
        )

    history = train_mixed_ferplus_raf(
        fer_data_dir=args.fer_data_dir,
        raf_data_dir=args.raf_data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=args.augmentation,
        lr=args.lr,
        pretrained_path=args.pretrained_path,
        freeze_epochs=args.freeze_epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=args.weighted_sampler,
        use_weighted_loss=not args.no_weighted_loss,
        class_weight_power=args.class_weight_power,
        patience=args.patience,
        seed=args.seed,
        val_split=args.val_split,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
        mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        domain_probs=args.domain_probs,
        selection_metric=args.selection_metric,
        output_dir=args.output_dir,
    )

    epochs = np.array(history["epochs"])
    if epochs.size > 0:
        plt.figure(figsize=(7, 4))
        train_acc = np.array(
            [acc if acc is not None else np.nan for acc in history["train_acc"]]
        )
        if not np.all(np.isnan(train_acc)):
            plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, history["val_acc_fer"], label="Val FER+ Acc")
        plt.plot(epochs, history["val_acc_raf"], label="Val RAF Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Mixed Training Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "accuracy_curves.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss_fer"], label="Val FER+ Loss")
        plt.plot(epochs, history["val_loss_raf"], label="Val RAF Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mixed Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()



if __name__ == "__main__":
    main()
