from collections.abc import Callable
from typing import Any

from torchvision import transforms

Transform = Callable[[Any], Any]


def _maybe_mtcnn_ops(
    use_mtcnn: bool, image_size: int, mtcnn_margin: float, mtcnn_device: str | None
) -> list[Transform]:
    if not use_mtcnn:
        return []
    from src.preprocessing.mtcnn_crop import MTCNNCrop

    return [MTCNNCrop(image_size=image_size, margin=mtcnn_margin, device=mtcnn_device)]


def _fer_norm_stats():
    return [0.5], [0.5]


def fer_train_transforms(
    image_size: int = 64,
    augmentation: str = "basic",
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
):
    mean, std = _fer_norm_stats()
    # basic (previous spec): Grayscale -> RRC(scale=0.9-1.0) -> HFlip -> Rotation(10)
    if augmentation not in {"basic", "strong"}:
        raise ValueError("augmentation must be 'basic' or 'strong'")
    ops: list[Transform] = _maybe_mtcnn_ops(use_mtcnn, image_size, mtcnn_margin, mtcnn_device)
    if augmentation == "basic":
        scale = (0.9, 1.0) if use_mtcnn else (0.75, 1.0)
        ops += [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(18),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6
            ),
        ]
    else:
        scale = (0.9, 1.0) if use_mtcnn else (0.75, 1.0)
        ops += [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(18),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6
            ),
        ]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def fer_eval_transforms(
    image_size: int = 64,
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
):
    mean, std = _fer_norm_stats()
    ops: list[Transform] = _maybe_mtcnn_ops(
        use_mtcnn, image_size, mtcnn_margin, mtcnn_device
    )
    ops.append(transforms.Grayscale(num_output_channels=1))
    if not use_mtcnn:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)

def raf_train_transforms(
    image_size: int = 64,
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
):
    mean, std = _fer_norm_stats()
    ops: list[Transform] = _maybe_mtcnn_ops(
        use_mtcnn, image_size, mtcnn_margin, mtcnn_device
    )
    scale = (0.9, 1.0) if use_mtcnn else (0.75, 1.0)
    ops += [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(image_size, scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(18),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.6
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)

def raf_eval_transforms(
    image_size: int = 64,
    use_mtcnn: bool = False,
    mtcnn_margin: float = 0.25,
    mtcnn_device: str | None = None,
):
    mean, std = _fer_norm_stats()
    ops: list[Transform] = _maybe_mtcnn_ops(
        use_mtcnn, image_size, mtcnn_margin, mtcnn_device
    )
    ops.append(transforms.Grayscale(num_output_channels=1))
    if not use_mtcnn:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)
