from torchvision import transforms


def _fer_norm_stats(num_channels: int):
    if num_channels == 1:
        return [0.5], [0.5]
    if num_channels == 3:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    raise ValueError("num_channels must be 1 or 3")


def fer_train_transforms(
    num_channels: int = 1, image_size: int = 64, augmentation: str = "basic"
):
    mean, std = _fer_norm_stats(num_channels)
    # basic (previous spec): Grayscale -> RRC(scale=0.9-1.0) -> HFlip -> Rotation(10)
    if augmentation not in {"basic", "strong"}:
        raise ValueError("augmentation must be 'basic' or 'strong'")
    if augmentation == "basic":
        ops = [
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    else:
        ops = [
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8
            ),
        ]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def fer_eval_transforms(num_channels: int = 1, image_size: int = 64):
    mean, std = _fer_norm_stats(num_channels)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def raf_train_transforms(image_size: int = 64):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def raf_eval_transforms(image_size: int = 64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
