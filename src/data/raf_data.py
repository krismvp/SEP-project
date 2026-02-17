from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .transforms import raf_eval_transforms, raf_train_transforms
from src.constants.emotions import CANON_6, CLASS_TO_IDX, normalize_emotion


@dataclass
class RAFSample:
    path: Path
    label: int


RAF_LABEL_TO_NAME = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}


class RAFDBCsvDataset(Dataset):
    """RAF-DB dataset using CSV labels (train_labels.csv/test_labels.csv)."""

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        transform=None,
        allow_empty: bool = False,
    ):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        self.samples = _load_csv_samples(self.csv_path, self.image_dir)
        if not self.samples and not allow_empty:
            raise FileNotFoundError(f"No RAF-DB samples found in {csv_path}.")
        self.samples, self.num_classes, stats = _normalize_samples(self.samples)
        if stats["fallback_used"] > 0:
            print(
                f"Warning: RAF label_id+1 fallback used for {stats['fallback_used']} samples. "
                "Check label integrity."
            )
        self.classes = list(CANON_6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample.label


def make_raf_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    image_size: int = 64,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    image_dir: Optional[str] = None,
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], int]:
    root = Path(data_dir)
    train_transform = raf_train_transforms(
        image_size=image_size,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )
    eval_transform = raf_eval_transforms(
        image_size=image_size,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )

    train_csv_path, test_csv_path = _find_csv_files(root, train_csv, test_csv)
    if train_csv_path is not None:
        train_image_dir = Path(image_dir) if image_dir else _find_csv_image_dir(root, "train")
        train_base = RAFDBCsvDataset(
            csv_path=train_csv_path,
            image_dir=train_image_dir,
            transform=train_transform,
        )
        num_classes = _infer_num_classes(train_base)

        if val_split > 0:
            eval_base = RAFDBCsvDataset(
                csv_path=train_csv_path,
                image_dir=train_image_dir,
                transform=eval_transform,
            )
            train_idx, val_idx = _split_indices(len(train_base), val_split, seed)
            train_dataset = Subset(train_base, train_idx)
            val_dataset = Subset(eval_base, val_idx) if val_idx else None
        else:
            train_dataset = train_base
            val_dataset = None

        test_dataset = None
        if test_csv_path is not None:
            if image_dir:
                test_image_dir = Path(image_dir)
            else:
                try:
                    test_image_dir = _find_csv_image_dir(root, "test")
                except FileNotFoundError:
                    test_image_dir = None
            if test_image_dir is not None:
                test_dataset = RAFDBCsvDataset(
                    csv_path=test_csv_path,
                    image_dir=test_image_dir,
                    transform=eval_transform,
                    allow_empty=True,
                )
                if len(test_dataset) == 0:
                    test_dataset = None
    else:
        raise FileNotFoundError(
            "RAF-DB not found. Provide train_labels.csv/test_labels.csv."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, num_classes


def _split_indices(num_samples: int, val_split: float, seed: int):
    val_size = int(num_samples * val_split)
    if val_size <= 0:
        return list(range(num_samples)), []
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    return indices[:-val_size], indices[-val_size:]


def _infer_num_classes(dataset: Dataset) -> int:
    if isinstance(dataset, Subset):
        return _infer_num_classes(dataset.dataset)
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return len(list(classes))
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("Unable to infer number of classes: dataset has no samples.")
    labels = set()
    for sample in list(samples):
        if isinstance(sample, RAFSample):
            labels.add(sample.label)
        elif isinstance(sample, (list, tuple)) and len(sample) > 1:
            labels.add(sample[1])
    if not labels:
        raise ValueError("Unable to infer number of classes from samples.")
    return len(labels)


def _normalize_samples(
    samples: List[RAFSample],
) -> Tuple[List[RAFSample], int, dict]:
    if not samples:
        return samples, 0, {"fallback_used": 0}
    normalized = []
    fallback_used = 0
    for sample in samples:
        label_id = sample.label
        name = RAF_LABEL_TO_NAME.get(label_id)
        if name is None:
            name = RAF_LABEL_TO_NAME.get(label_id + 1)
            if name is not None:
                fallback_used += 1
        if name is None:
            continue
        name = normalize_emotion(name)
        if name == "neutral":
            continue
        if name not in CLASS_TO_IDX:
            continue
        normalized.append(RAFSample(sample.path, CLASS_TO_IDX[name]))
    return normalized, len(CANON_6), {"fallback_used": fallback_used}


def _find_csv_files(
    root: Path,
    train_csv: Optional[str],
    test_csv: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    if train_csv:
        train_path = Path(train_csv)
    else:
        train_path = root / "train_labels.csv"
        if not train_path.exists():
            train_path = root / "DATASET" / "train_labels.csv"
        if not train_path.exists():
            train_path = None

    if test_csv:
        test_path = Path(test_csv)
    else:
        test_path = root / "test_labels.csv"
        if not test_path.exists():
            test_path = root / "DATASET" / "test_labels.csv"
        if not test_path.exists():
            test_path = None

    return train_path if train_path and train_path.exists() else None, test_path if test_path and test_path.exists() else None


def _find_csv_image_dir(root: Path, split: str) -> Path:
    candidates = [
        root / "DATASET" / split,
        root / split,
        root / "Image" / split,
        root / "images" / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find RAF-DB image directory for split '{split}'.")


def _load_csv_samples(csv_path: Path, image_dir: Path) -> List[RAFSample]:
    samples: List[RAFSample] = []
    file_map: Optional[dict[str, Path]] = None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            filename = row.get("image") or row.get("filename") or row.get("file_name")
            label = row.get("label")
            if filename is None or label is None:
                values = list(row.values())
                if len(values) >= 2:
                    filename, label = values[0], values[1]
                else:
                    continue
            try:
                label_int = int(label)
            except ValueError:
                continue
            path = _resolve_csv_image_path(
                image_dir=image_dir,
                filename=filename,
                label=label_int,
                file_map=file_map,
            )
            if path is None and file_map is None:
                file_map = _build_file_map(image_dir)
                path = file_map.get(filename)
            if path is None:
                continue
            samples.append(RAFSample(path, label_int))
    return samples


def _resolve_csv_image_path(
    image_dir: Path,
    filename: str,
    label: int,
    file_map: Optional[dict[str, Path]],
) -> Optional[Path]:
    candidates = [
        image_dir / filename,
        image_dir / str(label) / filename,
        image_dir / "train" / str(label) / filename,
        image_dir / "test" / str(label) / filename,
        image_dir / "DATASET" / "train" / str(label) / filename,
        image_dir / "DATASET" / "test" / str(label) / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if file_map is not None:
        return file_map.get(filename)
    return None


def _build_file_map(image_dir: Path) -> dict[str, Path]:
    return {path.name: path for path in image_dir.rglob("*.jpg")}


__all__ = ["make_raf_loaders"]
