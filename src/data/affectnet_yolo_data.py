from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import fer_eval_transforms
from src.constants.emotions import CANON_6, CLASS_TO_IDX, normalize_emotion


AFFECTNET_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def _resolve_split_dir(data_dir: str, split: str) -> Path:
    root = Path(data_dir)
    candidates = [split]
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")

    for name in candidates:
        if (root / name).is_dir():
            return root / name

    if (root / "images").is_dir() and (root / "labels").is_dir():
        return root

    raise FileNotFoundError(
        "AffectNet split not found. Expected split folder (e.g. test/) or images/labels "
        f"under: {root}"
    )


def _build_class_mapping() -> Dict[int, int]:
    class_map: Dict[int, int] = {}
    for idx, name in enumerate(AFFECTNET_CLASSES):
        mapped = normalize_emotion(name)
        if mapped in {"neutral", "contempt"}:
            continue
        if mapped not in CLASS_TO_IDX:
            continue
        class_map[idx] = CLASS_TO_IDX[mapped]
    return class_map


class AffectNetYoloDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        image_size: int = 64,
        transform=None,
    ):
        split_root = _resolve_split_dir(data_dir, split)
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/ and labels/ under {split_root}."
            )

        self.class_names = list(CANON_6)
        class_map = _build_class_mapping()
        self.transform = transform or fer_eval_transforms(image_size=image_size)

        image_map: Dict[str, Path] = {}
        for img_path in image_dir.iterdir():
            if img_path.is_file():
                image_map.setdefault(img_path.stem, img_path)

        self.samples: List[Tuple[Path, int]] = []
        stats = {
            "missing_images": 0,
            "empty_labels": 0,
            "dropped_labels": 0,
            "multi_label_files": 0,
        }

        label_files = sorted(label_dir.glob("*.txt"))
        for label_path in label_files:
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                stats["empty_labels"] += 1
                continue
            if len(lines) > 1:
                stats["multi_label_files"] += 1

            parts = lines[0].split()
            if not parts:
                stats["empty_labels"] += 1
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                stats["empty_labels"] += 1
                continue

            if cls not in class_map:
                stats["dropped_labels"] += 1
                continue

            img_path = image_map.get(label_path.stem)
            if img_path is None:
                stats["missing_images"] += 1
                continue

            self.samples.append((img_path, class_map[cls]))

        if not self.samples:
            raise ValueError("No AffectNet samples found after filtering.")

        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, target


def make_affectnet_yolo_loader(
    data_dir: str,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 64,
) -> Tuple[AffectNetYoloDataset, DataLoader]:
    dataset = AffectNetYoloDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


__all__ = ["AffectNetYoloDataset", "make_affectnet_yolo_loader", "AFFECTNET_CLASSES"]
