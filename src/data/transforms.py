from torchvision import transforms


def _fer_norm_stats():
    return [0.5], [0.5]


def fer_train_transforms(image_size: int = 64, augmentation: str = "basic"):
    mean, std = _fer_norm_stats()
    # basic (previous spec): Grayscale -> RRC(scale=0.9-1.0) -> HFlip -> Rotation(10)
    if augmentation not in {"basic", "strong"}:
        raise ValueError("augmentation must be 'basic' or 'strong'")
    if augmentation == "basic":
        ops = [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    else:
        ops = [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8
            ),
        ]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def fer_eval_transforms(image_size: int = 64):
    mean, std = _fer_norm_stats()
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def raf_train_transforms(image_size: int = 64):
    mean, std = _fer_norm_stats()
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def raf_eval_transforms(image_size: int = 64):
    mean, std = _fer_norm_stats()
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
