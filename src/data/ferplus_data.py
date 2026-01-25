from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
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


class _FilteredImageFolder(Dataset):
    def __init__(self, base: datasets.ImageFolder, keep_classes: list[str]):
        self.loader = base.loader
        self.transform = base.transform
        self.target_transform = base.target_transform
        self.classes = list(keep_classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        keep_indices = {
            base.class_to_idx[name]
            for name in keep_classes
            if name in base.class_to_idx
        }
        samples = []
        targets = []
        for path, target in base.samples:
            if target in keep_indices:
                class_name = base.classes[target]
                new_target = self.class_to_idx[class_name]
                samples.append((path, new_target))
                targets.append(new_target)
        if not samples:
            raise ValueError("No samples found after filtering FER+ classes.")
        self.samples = samples
        self.targets = targets

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


def _filter_classes(
    dataset: datasets.ImageFolder, drop_neutral: bool, drop_contempt: bool
) -> Dataset:
    drop = set()
    if drop_neutral:
        drop.add("neutral")
    if drop_contempt:
        drop.add("contempt")
    if not drop:
        return dataset
    keep_classes = [name for name in dataset.classes if name.lower() not in drop]
    if keep_classes == list(dataset.classes):
        return dataset
    return _FilteredImageFolder(dataset, keep_classes)


def _split_indices(num_samples: int, val_split: float, seed: int):
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
    num_channels: int = 1,
    image_size: int = 64,
    drop_neutral: bool = False,
    drop_contempt: bool = False,
    augmentation: str = "basic",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    root = _resolve_ferplus_root(data_dir)
    train_root = _find_split_dir(root, ["train", "Training"])
    if train_root is None:
        raise FileNotFoundError("FER+ train folder not found.")
    val_root = _find_split_dir(root, ["val", "Validation"])
    test_root = _find_split_dir(root, ["test", "Test"])

    train_transform = fer_train_transforms(
        num_channels=num_channels, image_size=image_size, augmentation=augmentation
    )
    eval_transform = fer_eval_transforms(num_channels=num_channels, image_size=image_size)

    if val_root is not None:
        train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        train_dataset = _filter_classes(train_dataset, drop_neutral, drop_contempt)
        val_dataset = datasets.ImageFolder(root=str(val_root), transform=eval_transform)
        val_dataset = _filter_classes(val_dataset, drop_neutral, drop_contempt)
    else:
        train_base = datasets.ImageFolder(root=str(train_root), transform=train_transform)
        eval_base = datasets.ImageFolder(root=str(train_root), transform=eval_transform)
        train_base = _filter_classes(train_base, drop_neutral, drop_contempt)
        eval_base = _filter_classes(eval_base, drop_neutral, drop_contempt)
        train_idx, val_idx = _split_indices(len(train_base), val_split, seed)
        train_dataset = Subset(train_base, train_idx)
        val_dataset = Subset(eval_base, val_idx) if val_idx else Subset(eval_base, [])

    test_dataset = None
    if test_root is not None:
        test_dataset = datasets.ImageFolder(root=str(test_root), transform=eval_transform)
        test_dataset = _filter_classes(test_dataset, drop_neutral, drop_contempt)

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
