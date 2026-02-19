from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .transforms import fer_eval_transforms, fer_train_transforms
from src.constants.emotions import CANON_6, CLASS_TO_IDX, normalize_emotion


def _find_split_dir(root: Path, names: list[str]) -> Optional[Path]:
    """Find first matching split directory to tolerate naming variants."""
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ferplus_root(data_dir: str) -> Path:
    """Resolve FER+ root across direct and nested extraction layouts."""
    root = Path(data_dir)
    if _find_split_dir(root, ["train", "Training"]) is not None:
        return root
    default_root = root / "fer2013plus" / "fer2013"
    if _find_split_dir(default_root, ["train", "Training"]) is not None:
        return default_root
    raise FileNotFoundError(
        "FER+ not found. Expected train/ (or Training/) under: "
        f"{root} or {default_root}"
    )


class _MappedImageFolder(Dataset):
    """ImageFolder view with labels remapped to the shared CANON_6 taxonomy."""
    def __init__(self, base: datasets.ImageFolder, samples: list[tuple[str, int]]):
        self.loader = base.loader
        self.transform = base.transform
        self.target_transform = base.target_transform
        self.classes = list(CANON_6)
        self.class_to_idx = CLASS_TO_IDX.copy()

        if not samples:
            raise ValueError("No samples found after mapping FER+ classes.")
        self.samples = samples
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target


def _unwrap_subset(dataset: Dataset):
    """Expose underlying dataset and optional index filter for remapping."""
    if isinstance(dataset, Subset):
        return dataset.dataset, set(dataset.indices)
    return dataset, None


def _map_ferplus_to_canon(dataset: datasets.ImageFolder) -> Dataset:
    """Drop unsupported labels and normalize class names to canonical indices."""
    base, indices = _unwrap_subset(dataset)
    if not hasattr(base, "samples") or not hasattr(base, "classes"):
        return dataset
    samples: list[tuple[str, int]] = []
    for idx, (path, target) in enumerate(base.samples):
        if indices is not None and idx not in indices:
            continue
        class_name = base.classes[target]
        mapped = normalize_emotion(class_name)
        # Neutral/contempt are intentionally excluded to keep 6-class training aligned.
        if mapped in {"neutral", "contempt"}:
            continue
        if mapped not in CLASS_TO_IDX:
            continue
        samples.append((path, CLASS_TO_IDX[mapped]))
    return _MappedImageFolder(base, samples)


def _split_indices(num_samples: int, val_split: float, seed: int):
    """Create deterministic train/val split for reproducible comparisons."""
    val_size = int(num_samples * val_split)
    if val_size <= 0:
        return list(range(num_samples)), []
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    return indices[:-val_size], indices[-val_size:]


def make_ferplus_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    image_size: int = 64,
    augmentation: str = "basic",
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build FER+ loaders with canonical label mapping and optional explicit val/test."""
    root = _resolve_ferplus_root(data_dir)
    train_root = _find_split_dir(root, ["train", "Training"])
    if train_root is None:
        raise FileNotFoundError("FER+ train folder not found.")
    val_root = _find_split_dir(root, ["val", "Validation"])
    test_root = _find_split_dir(root, ["test", "Test"])

    train_transform = fer_train_transforms(
        image_size=image_size,
        augmentation=augmentation,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )
    eval_transform = fer_eval_transforms(
        image_size=image_size,
        use_mtcnn=use_mtcnn,
        mtcnn_margin=mtcnn_margin,
        mtcnn_device=mtcnn_device,
    )

    if val_root is not None:
        train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        train_dataset = _map_ferplus_to_canon(train_dataset)
        val_dataset = datasets.ImageFolder(root=str(val_root), transform=eval_transform)
        val_dataset = _map_ferplus_to_canon(val_dataset)
    else:
        train_base = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        eval_base = datasets.ImageFolder(root=str(train_root), transform=eval_transform)
        train_base = _map_ferplus_to_canon(train_base)
        eval_base = _map_ferplus_to_canon(eval_base)
        train_idx, val_idx = _split_indices(len(train_base), val_split, seed)
        train_dataset = Subset(train_base, train_idx)
        val_dataset = Subset(eval_base, val_idx) if val_idx else Subset(eval_base, [])

    test_dataset = None
    if test_root is not None:
        test_dataset = datasets.ImageFolder(root=str(test_root), transform=eval_transform)
        test_dataset = _map_ferplus_to_canon(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
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

    return train_loader, val_loader, test_loader


__all__ = ["make_ferplus_loaders"]
