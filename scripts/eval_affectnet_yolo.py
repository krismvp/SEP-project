import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.affectnet_yolo_data import make_affectnet_yolo_loader
from src.models.resnet_small import ResNet18


def _adapt_conv1_to_grayscale(state: dict) -> dict:
    weight = state.get("conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 4:
        if int(weight.shape[1]) == 3:
            state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
    return state


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a FER+ ResNet18 checkpoint on AffectNet YOLO-format splits."
    )
    parser.add_argument("--data-dir", default="data/AffectNet")
    parser.add_argument("--weights", required=True, help="Path to resnet18_best.pth")
    parser.add_argument("--split", choices=["train", "test", "val", "valid"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/affectnet_eval")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a state_dict key.")
    state = _strip_module_prefix(state)
    state = _adapt_conv1_to_grayscale(state)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    in_channels = 1

    dataset, loader = make_affectnet_yolo_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    class_names = dataset.class_names
    num_classes = len(class_names)
    if class_names:
        print(f"Class order: {class_names}")
    fc_weight = state.get("fc.weight")
    if isinstance(fc_weight, torch.Tensor) and fc_weight.ndim == 2:
        expected_classes = int(fc_weight.shape[0])
        if expected_classes != num_classes:
            raise ValueError(
                f"Checkpoint expects {expected_classes} classes, but dataset has {num_classes}. "
                "Check drop flags or use a matching checkpoint."
            )

    model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(state)
    model.eval()

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    correct = 0
    total = 0
    printed_debug = False

    with torch.no_grad():
        for imgs, labels in loader:
            if not printed_debug:
                print("conv1:", tuple(model.conv1.weight.shape))
                print("batch:", tuple(imgs.shape))
                print("batch_dtype:", imgs.dtype)
                print("batch_range:", (float(imgs.min()), float(imgs.max())))
                print("class_names:", dataset.class_names)
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

    os.makedirs(args.output_dir, exist_ok=True)
    cm_array = cm.numpy()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.png")
    csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    csv_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
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
    fig_norm.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()
    fig_norm.savefig(cm_norm_path, dpi=150)
    plt.close(fig_norm)

    np.savetxt(csv_norm_path, cm_norm * 100.0, delimiter=",", fmt="%.2f")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")

    stats = dataset.stats
    if any(stats.values()):
        print(
            "Dataset filter stats:",
            f"missing_images={stats['missing_images']},",
            f"empty_labels={stats['empty_labels']},",
            f"dropped_labels={stats['dropped_labels']},",
            f"multi_label_files={stats['multi_label_files']}",
        )


if __name__ == "__main__":
    main()
