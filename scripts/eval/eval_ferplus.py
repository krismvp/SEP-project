import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.ferplus_data import make_ferplus_loaders
from src.models.factory import build_model


def _unwrap_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _get_class_names(dataset, num_classes: int) -> list[str]:
    base = _unwrap_dataset(dataset)
    classes = getattr(base, "classes", None)
    if classes:
        return list(classes)
    return [str(i) for i in range(num_classes)]


def _adapt_conv1_to_grayscale(state: dict) -> dict:
    weight = state.get("conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 4:
        if int(weight.shape[1]) == 3:
            state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FER+ model on test set.")
    parser.add_argument("--data-dir", default="data/ferplus")
    parser.add_argument("--weights", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/ferplus_eval")
    parser.add_argument("--use-mtcnn", action="store_true", default=True)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    args = parser.parse_args()
    if args.no_mtcnn:
        args.use_mtcnn = False

    if args.use_mtcnn:
        print(
            f"MTCNN: enabled=True (margin={args.mtcnn_margin}, device={args.mtcnn_device})"
        )

    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a state_dict key.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    state = _adapt_conv1_to_grayscale(state)
    in_channels = 1

    _, _, test_loader = make_ferplus_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=42,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )
    _, val_loader, _ = make_ferplus_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=42,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )

    eval_loader = test_loader if args.split == "test" else val_loader
    if eval_loader is None:
        raise FileNotFoundError(f"FER+ {args.split} folder not found.")

    class_names = _get_class_names(eval_loader.dataset, 0)
    num_classes = len(class_names)
    if num_classes == 0:
        fc_weight = state.get("fc.weight")
        if isinstance(fc_weight, torch.Tensor) and fc_weight.ndim == 2:
            num_classes = int(fc_weight.shape[0])
            class_names = [str(i) for i in range(num_classes)]
        else:
            raise ValueError("Unable to infer number of classes for evaluation.")
    model = build_model(args.arch, num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(state)
    model.eval()

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    correct = 0
    total = 0
    printed_debug = False

    with torch.no_grad():
        for imgs, labels in eval_loader:
            if not printed_debug:
                print("conv1:", tuple(model.conv1.weight.shape))
                print("batch:", tuple(imgs.shape))
                print("batch_dtype:", imgs.dtype)
                print("batch_range:", (float(imgs.min()), float(imgs.max())))
                print("class_names:", class_names)
                printed_debug = True
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            indices = labels_cpu * num_classes + preds_cpu
            cm += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    acc = 100 * correct / max(total, 1)
    print(f"Test Acc: {acc:.2f}%")
    title_base = f"FER+ {args.split} Acc: {acc:.2f}%"

    cm_array = cm.numpy()
    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.png")
    csv_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(title_base)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    np.savetxt(csv_path, cm_array, delimiter=",", fmt="%d")

    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_array.astype(float),
        row_sums,
        out=np.zeros_like(cm_array, dtype=float),
        where=row_sums > 0,
    )
    fig_norm, ax_norm = plt.subplots(figsize=(6, 5))
    im_norm = ax_norm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax_norm.set_xlabel("Predicted")
    ax_norm.set_ylabel("True")
    ax_norm.set_xticks(range(len(class_names)))
    ax_norm.set_yticks(range(len(class_names)))
    ax_norm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_norm.set_yticklabels(class_names)
    ax_norm.set_title(f"{title_base} (Normalized)")
    fig_norm.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()
    fig_norm.savefig(cm_norm_path, dpi=150)
    plt.close(fig_norm)

    np.savetxt(csv_norm_path, cm_norm * 100.0, delimiter=",", fmt="%.2f")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")


if __name__ == "__main__":
    main()
