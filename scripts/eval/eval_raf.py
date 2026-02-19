import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.constants.emotions import normalize_emotion  # Emotion name normalization
from src.data.raf_data import make_raf_loaders  # RAF-DB data loader
from src.models.factory import build_model  # Model factory


def _unwrap_dataset(dataset):
    """Extract the base dataset from potentially wrapped DataLoader objects."""
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _get_class_names(dataset, num_classes: int) -> list[str]:
    """Retrieve class names from dataset, fallback to numeric indices if not available."""
    base = _unwrap_dataset(dataset)
    classes = getattr(base, "classes", None)
    if classes:
        return list(classes)
    return [str(i) for i in range(num_classes)]


def _sample_to_path(sample) -> str:
    """Extract file path from a dataset sample (supports multiple sample formats)."""
    if isinstance(sample, (list, tuple)) and sample:
        return str(sample[0])
    path = getattr(sample, "path", None)
    if path is not None:
        return str(path)
    raise ValueError("Unable to extract filepath from dataset sample.")


def _extract_paths(dataset) -> list[str]:
    """Extract all sample file paths from dataset for CSV export."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
        base_paths = _extract_paths(base)
        return [base_paths[idx] for idx in indices]
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [_sample_to_path(sample) for sample in samples]
    imgs = getattr(dataset, "imgs", None)
    if imgs is not None:
        return [_sample_to_path(sample) for sample in imgs]
    raise ValueError("Dataset does not expose sample paths for CSV export.")


def _pretty_label(name: str) -> str:
    """Normalize emotion names to pretty-printed format for CSV headers."""
    norm = normalize_emotion(name)
    if norm == "happy":
        return "happiness"
    if norm == "sad":
        return "sadness"
    return norm


def _resolve_pred_order(
    class_names: list[str], desired_order: list[str]
) -> tuple[list[str], list[int]]:
    """Map desired prediction order to actual class indices. Returns CSV header and reordering indices."""
    norm_to_idx = {normalize_emotion(name): idx for idx, name in enumerate(class_names)}
    header: list[str] = []
    order_indices: list[int] = []
    missing: list[str] = []
    for name in desired_order:
        norm = normalize_emotion(name)
        idx = norm_to_idx.get(norm)
        if idx is None:
            missing.append(name)
            continue
        header.append(name)
        order_indices.append(idx)
    if missing:
        print(
            "Warning: prediction CSV order not found; using dataset class order."
        )
        header = [_pretty_label(name) for name in class_names]
        order_indices = list(range(len(class_names)))
    return header, order_indices


def _adapt_conv1_to_grayscale(state: dict) -> dict:
    """Adapt conv1 layer from RGB (3-channel) to grayscale (1-channel) by averaging weights."""
    weight = state.get("conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 4:
        if int(weight.shape[1]) == 3:
            state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
    return state


def _strip_module_prefix(state: dict) -> dict:
    """Remove 'module.' prefix from state_dict keys (from DataParallel models)."""
    if not state:
        return state
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


DEFAULT_PRED_ORDER = [
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]


def main() -> None:
    """Main evaluation function: Load model, evaluate on RAF-DB dataset, save predictions and confusion matrices."""
    parser = argparse.ArgumentParser(description="Evaluate RAF-DB model.")
    parser.add_argument("--data-dir", default="data/RAF-DB")
    parser.add_argument("--weights", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", default="outputs/raf_eval")
    parser.add_argument("--use-mtcnn", action="store_true", default=True)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument(
        "--preds-order",
        nargs="+",
        default=DEFAULT_PRED_ORDER,
        help="CSV column order for probability export.",
    )
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
    state = _strip_module_prefix(state)
    state = _adapt_conv1_to_grayscale(state)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    _, val_loader, test_loader, num_classes = make_raf_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=42,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )

    eval_loader = test_loader if args.split == "test" else val_loader
    if eval_loader is None:
        raise FileNotFoundError(f"RAF {args.split} split not found.")

    class_names = _get_class_names(eval_loader.dataset, num_classes)
    if class_names:
        print(f"Class order: {class_names}")

    model = build_model(args.arch, num_classes=num_classes, in_channels=1).to(device)
    model.load_state_dict(state)
    model.eval()  

    os.makedirs(args.output_dir, exist_ok=True)
    preds_path = os.path.join(args.output_dir, "predictions.csv")
    paths = _extract_paths(eval_loader.dataset)
    header, order_indices = _resolve_pred_order(class_names, args.preds_order)
    if len(paths) != len(eval_loader.dataset):
        raise ValueError("Prediction CSV path count does not match dataset size.")

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    correct = 0
    total = 0
    printed_debug = False
    path_offset = 0

    with open(preds_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath"] + header)

        with torch.no_grad():  
            for imgs, labels in eval_loader:
                if not printed_debug:
                    print("conv1:", tuple(model.conv1.weight.shape))
                    print("batch:", tuple(imgs.shape))
                    print("batch_dtype:", imgs.dtype)
                    print("batch_range:", (float(imgs.min()), float(imgs.max())))
                    printed_debug = True
                
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                preds = outputs.argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                labels_cpu = labels.detach().cpu()
                preds_cpu = preds.detach().cpu()
                indices = labels_cpu * num_classes + preds_cpu
                cm += torch.bincount(
                    indices, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)

                batch_size = labels.size(0)
                batch_paths = paths[path_offset : path_offset + batch_size]
                path_offset += batch_size
                for path, prob in zip(batch_paths, probs):
                    row = [path] + [f"{prob[idx]:.4f}" for idx in order_indices]
                    writer.writerow(row)

    acc = 100 * correct / max(total, 1)
    print(f"Test Acc: {acc:.2f}%")
    title_base = f"RAF-DB {args.split} Acc: {acc:.2f}%"

    cm_array = cm.numpy()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_norm_path = os.path.join(args.output_dir, "confusion_matrix_normalized.png")

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

    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")

if __name__ == "__main__":
    main()
