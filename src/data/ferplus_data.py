from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from .transforms import fer_eval_transforms, fer_train_transforms


def _find_split_dir(root: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ferplus_root(data_dir: str) -> Path:
    root = Path(data_dir)
    candidates = [
        root,
        root / "fer2013plus" / "fer2013",
        root / "fer2013plus",
        root / "fer2013",
    ]
    for candidate in candidates:
        if _find_split_dir(candidate, ["train", "Training"]) is not None:
            return candidate
    raise FileNotFoundError(
        "FER+ not found. Expected train/ (or Training/) under: "
        f"{', '.join(str(c) for c in candidates)}"
    )


def make_ferplus_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    num_channels: int = 1,
    image_size: int = 64,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    root = _resolve_ferplus_root(data_dir)
    train_root = _find_split_dir(root, ["train", "Training"])
    if train_root is None:
        raise FileNotFoundError("FER+ train folder not found.")
    val_root = _find_split_dir(root, ["val", "Validation"])
    test_root = _find_split_dir(root, ["test", "Test"])

    train_transform = fer_train_transforms(num_channels=num_channels, image_size=image_size)
    eval_transform = fer_eval_transforms(num_channels=num_channels, image_size=image_size)

    if val_root is not None:
        train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(val_root), transform=eval_transform)
    else:
        base_dataset = datasets.ImageFolder(root=str(train_root))
        train_size = int((1 - val_split) * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_split, val_split_set = random_split(
            base_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(train_root), transform=eval_transform)
        train_dataset = Subset(train_dataset, train_split.indices)
        val_dataset = Subset(val_dataset, val_split_set.indices)

    test_dataset = None
    if test_root is not None:
        test_dataset = datasets.ImageFolder(root=str(test_root), transform=eval_transform)

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
