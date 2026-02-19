import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.constants.emotions import CANON_6, normalize_emotion  
from src.data.transforms import fer_eval_transforms  
from src.models.factory import build_model  

DEFAULT_PRED_ORDER = [
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]

DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
DEFAULT_WEIGHTS = ROOT / "outputs/mixed/ferplus_raf_best_generalization/resnet34_best.pth"


class FolderImageDataset(Dataset):
    """Dataset for loading images from a folder and processing them for emotion classification."""
    def __init__(
        self,
        folder: str,
        transform,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> None:
        """Initialize the dataset."""
        self.root = Path(folder)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.root}")

        ext_set = {_normalize_ext(ext) for ext in (extensions or DEFAULT_EXTENSIONS)}
        if recursive:
            paths = [
                path
                for path in self.root.rglob("*")
                if path.is_file() and path.suffix.lower() in ext_set
            ]
        else:
            paths = [
                path
                for path in self.root.iterdir()
                if path.is_file() and path.suffix.lower() in ext_set
            ]
        self.paths = sorted(paths, key=lambda p: str(p).lower())
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {self.root} with extensions: {sorted(ext_set)}"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        """Load and transform an image."""
        path = self.paths[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)


def _normalize_ext(ext: str) -> str:
    """Normalize file extension to lowercase with leading dot."""
    cleaned = ext.strip().lower()
    if not cleaned:
        raise ValueError("Empty extension is not allowed.")
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def _strip_module_prefix(state: dict) -> dict:
    """Remove 'module.' prefix from state_dict keys (from DataParallel models)."""
    if not state or not any(key.startswith("module.") for key in state):
        return state
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def _adapt_conv1_to_grayscale(state: dict) -> dict:
    """Adapt conv1 layer from RGB (3-channel) to grayscale (1-channel) by averaging weights."""
    weight = state.get("conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 4 and int(weight.shape[1]) == 3:
        state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
    return state


def _extract_state_dict(checkpoint) -> dict:
    """Extract model state dictionary from checkpoint in various formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a state_dict key.")
    return checkpoint


def _infer_num_classes(state: dict) -> int:
    """Infer number of emotion classes from the model's fully connected layer."""
    fc_weight = state.get("fc.weight")
    if isinstance(fc_weight, torch.Tensor) and fc_weight.ndim == 2:
        return int(fc_weight.shape[0])
    raise ValueError(
        "Unable to infer num_classes from checkpoint. Pass --num-classes explicitly."
    )


def _resolve_pred_order(
    class_names: list[str], desired_order: list[str]
) -> tuple[list[str], list[int]]:
    """Map desired prediction order to actual class indices for CSV export."""
    norm_to_idx = {normalize_emotion(name): idx for idx, name in enumerate(class_names)}
    header: list[str] = []
    order_indices: list[int] = []
    for name in desired_order:
        norm = normalize_emotion(name)
        idx = norm_to_idx.get(norm)
        if idx is None:
            raise ValueError(
                f"Label '{name}' from --preds-order is not present in canonical class names: {class_names}"
            )
        header.append(name)
        order_indices.append(idx)
    return header, order_indices


def _pick_device(name: str | None) -> torch.device:
    """Select computation device (GPU, Apple Silicon, or CPU)."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Main function: Load model, process images in folder, and export emotion predictions to CSV."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run emotion classification on all images in a folder and export scores to CSV."
    )
    parser.add_argument("input_dir", help="Folder containing images to classify.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to checkpoint .pth",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/folder_predictions.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--preds-order",
        nargs="+",
        default=DEFAULT_PRED_ORDER,
        help="CSV column order for scores.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-mtcnn", action="store_true", default=True)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Image file extensions to include.",
    )
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    if args.no_mtcnn:
        args.use_mtcnn = False
    if args.no_recursive:
        args.recursive = False

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}. "
            "Pass --weights to override the default ferplus_raf_best_generalization checkpoint."
        )

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    state = _strip_module_prefix(state)
    state = _adapt_conv1_to_grayscale(state)

    num_classes = args.num_classes or _infer_num_classes(state)
    if num_classes != len(CANON_6):
        raise ValueError(
            f"Expected a 6-class canonical checkpoint, but inferred num_classes={num_classes}."
        )
    class_names = list(CANON_6)

    header, order_indices = _resolve_pred_order(class_names, args.preds_order)
    device = _pick_device(args.device)

    model = build_model(args.arch, num_classes=num_classes, in_channels=1).to(device)
    model.load_state_dict(state)
    model.eval()  

    transform = fer_eval_transforms(
        image_size=args.image_size,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
    )
    dataset = FolderImageDataset(
        folder=args.input_dir,
        transform=transform,
        recursive=args.recursive,
        extensions=args.extensions,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath"] + header)
        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                for path, prob in zip(paths, probs):
                    writer.writerow([path] + [f"{prob[idx]:.4f}" for idx in order_indices])
                    total += 1

    print(f"Processed {total} images")
    print(f"CSV saved to: {output_path}")

if __name__ == "__main__":
    main()