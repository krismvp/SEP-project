from typing import Any, Dict, List, Protocol, Tuple

import csv
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class _ImageDataset(Protocol):
    """Minimal dataset contract used by the SimCLR wrapper."""
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...


def _resolve_celeba_root(data_dir: str) -> Path:
    """Support common CelebA folder layouts to reduce setup friction."""
    root = Path(data_dir)
    candidates = [
        root,
        root / "celeba",
        root / "celebA",
        root / "celebA" / "celeba",
        root / "celeba" / "celeba",
    ]
    for candidate in candidates:
        if (candidate / "img_align_celeba").exists() and (
            (candidate / "list_bbox_celeba.csv").exists()
            or (candidate / "list_bbox_celeba.txt").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "CelebA not found. Expected img_align_celeba and list_bbox_celeba.{csv,txt} "
        f"under: {', '.join(str(c) for c in candidates)}"
    )


def _load_bbox_map(root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """Load face boxes from CSV or TXT so both official formats are accepted."""
    csv_path = root / "list_bbox_celeba.csv"
    txt_path = root / "list_bbox_celeba.txt"
    bbox_map: Dict[str, Tuple[int, int, int, int]] = {}

    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["image_id"]
                x1 = int(row["x_1"])
                y1 = int(row["y_1"])
                width = int(row["width"])
                height = int(row["height"])
                bbox_map[image_id] = (x1, y1, width, height)
        return bbox_map

    if txt_path.exists():
        with txt_path.open() as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            image_id, x1, y1, width, height = parts
            bbox_map[image_id] = (int(x1), int(y1), int(width), int(height))
        return bbox_map

    raise FileNotFoundError("Missing list_bbox_celeba.csv or list_bbox_celeba.txt")


def _load_partitions(root: Path) -> Dict[str, int]:
    """Load split assignments if available; fallback to all images otherwise."""
    csv_path = root / "list_eval_partition.csv"
    txt_path = root / "list_eval_partition.txt"
    partitions: Dict[str, int] = {}

    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                partitions[row["image_id"]] = int(row["partition"])
        return partitions

    if txt_path.exists():
        with txt_path.open() as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            image_id, partition = parts
            partitions[image_id] = int(partition)
        return partitions

    return {}


def _crop_with_padding(
    image: Image.Image, bbox: Tuple[int, int, int, int], padding: float
) -> Image.Image:
    """Expand bounding boxes slightly so crops keep context around the face."""
    x1, y1, width, height = bbox
    if width <= 0 or height <= 0:
        return image
    pad_w = max(0, int(width * padding))
    pad_h = max(0, int(height * padding))
    left = max(0, x1 - pad_w)
    top = max(0, y1 - pad_h)
    right = min(image.width, x1 + width + pad_w)
    bottom = min(image.height, y1 + height + pad_h)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


class CelebAUnlabeled(Dataset):
    """CelebA dataset that returns cropped face images only (no labels)."""

    _split_map = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_faces: bool = True,
        padding: float = 0.1,
    ):
        self.root = _resolve_celeba_root(root)
        self.images_dir = self.root / "img_align_celeba"
        nested_dir = self.images_dir / "img_align_celeba"
        if nested_dir.exists():
            self.images_dir = nested_dir
        self.crop_faces = crop_faces
        self.padding = padding
        self.available_images = {
            p.name for p in self.images_dir.iterdir() if p.is_file()
        }
        self.bbox_map = _load_bbox_map(self.root)
        self.partitions = _load_partitions(self.root)
        self.image_ids = self._filter_split(split)
        if not self.image_ids:
            raise FileNotFoundError(
                "No images found for the requested split. Check img_align_celeba "
                "and that list_bbox_celeba matches the image files."
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self.images_dir / image_id).convert("RGB")
        if self.crop_faces:
            bbox = self.bbox_map.get(image_id)
            if bbox is not None:
                image = _crop_with_padding(image, bbox, self.padding)
        return image

    def _filter_split(self, split: str) -> List[str]:
        """Prefer official partitions, but keep an 'all' mode for quick experiments."""
        if split == "all" or not self.partitions:
            return [img_id for img_id in self.bbox_map.keys() if img_id in self.available_images]
        if split not in self._split_map:
            raise ValueError(f"Unknown split '{split}'. Use train, val, test, or all.")
        split_id = self._split_map[split]
        return [
            img_id
            for img_id in self.bbox_map.keys()
            if self.partitions.get(img_id) == split_id and img_id in self.available_images
        ]


class SimCLRDataset(Dataset):
    """Dataset wrapper that returns two augmented views."""

    def __init__(self, base_dataset: _ImageDataset, transform):
        self.base_dataset: _ImageDataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return self.base_dataset.__len__()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Two stochastic views are required for contrastive positive pairs.
        img = self.base_dataset[index]
        return self.transform(img), self.transform(img)


def build_ssl_transform(image_size: int = 64) -> transforms.Compose:
    """Use stronger augmentations so invariances are learned during SSL pretraining."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)],
                p=0.8,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def make_ssl_loader(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = False,
    image_size: int = 64,
    split: str = "train",
    crop_faces: bool = True,
    face_padding: float = 0.1,
) -> DataLoader:
    """Create SSL loader with face-cropped CelebA and paired SimCLR views."""
    base_dataset = CelebAUnlabeled(
        root=data_dir,
        split=split,
        crop_faces=crop_faces,
        padding=face_padding,
    )
    dataset = SimCLRDataset(base_dataset, build_ssl_transform(image_size=image_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
