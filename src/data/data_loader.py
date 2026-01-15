import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from .transforms import fer_train_transforms, fer_eval_transforms

def make_fer_loaders(data_path, batch_size=64, val_split=0.1, seed=42):
    base_dataset = datasets.ImageFolder(root=f"{data_path}/train")
    train_size = int((1 - val_split) * len(base_dataset))
    val_size = len(base_dataset) - train_size

    train_split, val_split = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = datasets.ImageFolder(
        root=f"{data_path}/train",
        transform=fer_train_transforms(),
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_path}/val",
        transform=fer_eval_transforms(),
    )

    train_dataset = Subset(train_dataset, train_split.indices)
    val_dataset = Subset(val_dataset, val_split.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader
  
if __name__ == "__main__":
        train_loader, val_loader = make_fer_loaders("data/FER13")
        images, labels = next(iter(train_loader))
        print(f"Batch-Shape: {images.shape}")
