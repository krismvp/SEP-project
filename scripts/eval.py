import argparse
import json
import os
from pyexpat import model
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import matplotlib.pyplot as plt

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.resnet_small import ResNet18
from src.data.data_loader import make_fer_loaders
from src.data.transforms import fer_eval_transforms


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained ResNet model on FER ImageFolder dataset.")
    ap.add_argument("--data-path", default="data/FER13", help="Dataset root path (expects train/ and test/ subfolders)")
    ap.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test", help="Which split to evaluate")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=6)
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--save-misclassified", action="store_true", help="Save misclassified samples to CSV")
    ap.add_argument("--max-misclassified", type=int, default=50, help="Max rows to store in misclassified.csv")
    return ap.parse_args()


def ensure_outdir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_model(weights_path: str, num_classes: int, in_channels: int, device: torch.device) -> torch.nn.Module:
    model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    state = torch.load(weights_path, map_location=device)

    # --- adapt 3-channel checkpoint to 1-channel model (grayscale) ---
    if "conv1.weight" in state:
        w = state["conv1.weight"]  # shape either [64,3,3,3] or [64,1,3,3]
        if w.ndim == 4 and w.shape[1] == 3 and model.conv1.weight.shape[1] == 1:
            state["conv1.weight"] = w.mean(dim=1, keepdim=True)  # -> [64,1,3,3]
    model.load_state_dict(state)
    model.eval()
    return model


def build_loader(data_path: str, split: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    """
    Returns (loader, class_names)
    - split=test: reads ImageFolder(data_path/test)
    - split=val or train: uses make_fer_loaders(data_path) which reads data_path/train and splits into train/val
    """
    if split == "test":
        ds = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=fer_eval_transforms())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return loader, ds.classes

    # train/val come from your existing pipeline (splitting the train folder)
    train_loader, val_loader = make_fer_loaders(data_path, batch_size=batch_size, val_split=0.1, seed=42)
    loader = val_loader if split == "val" else train_loader

    # Try to infer class names from underlying ImageFolder dataset
    class_names = None
    ds = loader.dataset
    if isinstance(ds, Subset) and hasattr(ds, "dataset") and hasattr(ds.dataset, "classes"):
        class_names = ds.dataset.classes
    elif hasattr(ds, "classes"):
        class_names = ds.classes
    else:
        # fallback to numeric labels
        class_names = [str(i) for i in range(len(set(getattr(ds, "targets", []))))]  # may be empty

    if not class_names:
        class_names = [str(i) for i in range(7)]  # safe fallback

    return loader, class_names


def get_sample_path_from_dataset(dataset, local_index: int) -> Optional[str]:
    """
    Attempts to return the original filepath for a sample.
    Works for:
    - torchvision.datasets.ImageFolder
    - torch.utils.data.Subset(ImageFolder)
    """
    try:
        if isinstance(dataset, datasets.ImageFolder):
            return dataset.samples[local_index][0]
        if isinstance(dataset, Subset):
            base = dataset.dataset
            base_idx = dataset.indices[local_index]
            if isinstance(base, datasets.ImageFolder):
                return base.samples[base_idx][0]
    except Exception:
        return None
    return None


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []
    paths: List[Optional[str]] = []

    softmax = torch.nn.Softmax(dim=1)

    dataset = loader.dataset
    seen = 0

    for batch_imgs, batch_labels in loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_imgs)
        probs = softmax(logits)

        confs, preds = torch.max(probs, dim=1)

        # collect
        y_true.extend(batch_labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_conf.extend(confs.detach().cpu().tolist())

        # paths (best effort)
        bs = batch_imgs.size(0)
        for i in range(bs):
            p = get_sample_path_from_dataset(dataset, seen + i)
            paths.append(p)
        seen += bs

    return y_true, y_pred, y_conf, paths


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    total = max(len(y_true), 1)
    return correct / total


def per_class_accuracy(cm: List[List[int]]) -> List[float]:
    accs = []
    for i in range(len(cm)):
        total = sum(cm[i])
        correct = cm[i][i]
        accs.append((correct / total) if total > 0 else 0.0)
    return accs


def save_confusion_matrix_png(cm: List[List[int]], class_names: List[str], out_path: str):
    cm = np.array(cm, dtype=float)

    # normalize rows (True labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n = len(class_names)
    cm = cm[:n, :n]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_metrics_json(metrics: dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_misclassified_csv(
    y_true: List[int],
    y_pred: List[int],
    y_conf: List[float],
    paths: List[Optional[str]],
    out_path: str,
    max_rows: int = 50,
):
    # collect misclassified with highest confidence first
    rows = []
    for t, p, c, path in zip(y_true, y_pred, y_conf, paths):
        if t != p:
            rows.append((c, t, p, path if path else ""))

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:max_rows]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("confidence,true_label,pred_label,path\n")
        for c, t, p, path in rows:
            # basic CSV escaping
            path_esc = path.replace('"', '""')
            f.write(f'{c:.6f},{t},{p},"{path_esc}"\n')


def main():
    args = parse_args()
    out_dir = ensure_outdir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    loader, class_names = build_loader(
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    # Load model
    model = load_model(
        weights_path=args.weights,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        device=device,
    )

    # Run eval
    y_true, y_pred, y_conf, paths = run_eval(model, loader, device)

    # Metrics
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=args.num_classes)
    acc = accuracy(y_true, y_pred)
    pc_acc = per_class_accuracy(cm)

    metrics = {
        "split": args.split,
        "weights": args.weights,
        "num_classes": args.num_classes,
        "in_channels": args.in_channels,
        "batch_size": args.batch_size,
        "accuracy": acc,
        "per_class_accuracy": pc_acc,
    }

    # Save
    metrics_path = os.path.join(out_dir, "metrics.json")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    save_metrics_json(metrics, metrics_path)
    save_confusion_matrix_png(cm, class_names, cm_path)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")

    if args.save_misclassified:
        mis_path = os.path.join(out_dir, "misclassified.csv")
        save_misclassified_csv(y_true, y_pred, y_conf, paths, mis_path, max_rows=args.max_misclassified)
        print(f"Saved misclassified samples to: {mis_path}")


if __name__ == "__main__":
    main()
