import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from .transforms import fer_train_transforms, fer_eval_transforms

def make_fer_loaders(
    data_path,
    batch_size=64,
    val_split=0.1,
    seed=42,
    num_workers=4,
    image_size=64,
    augmentation="basic",
    use_mtcnn=False,
    mtcnn_margin=0.25,
    mtcnn_device=None,
):
    train_root = f"{data_path}/train"
    val_root = f"{data_path}/val"

    if os.path.isdir(val_root):
        train_dataset = datasets.ImageFolder(
            root=train_root,
            transform=fer_train_transforms(
                image_size=image_size,
                augmentation=augmentation,
                use_mtcnn=use_mtcnn,
                mtcnn_margin=mtcnn_margin,
                mtcnn_device=mtcnn_device,
            ),
        )
        val_dataset = datasets.ImageFolder(
            root=val_root,
            transform=fer_eval_transforms(
                image_size=image_size,
                use_mtcnn=use_mtcnn,
                mtcnn_margin=mtcnn_margin,
                mtcnn_device=mtcnn_device,
            ),
        )
    else:
        base_dataset = datasets.ImageFolder(root=train_root)
        train_size = int((1 - val_split) * len(base_dataset))
        val_size = len(base_dataset) - train_size

        train_split, val_split = random_split(
            base_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_dataset = datasets.ImageFolder(
            root=train_root,
            transform=fer_train_transforms(
                image_size=image_size,
                augmentation=augmentation,
                use_mtcnn=use_mtcnn,
                mtcnn_margin=mtcnn_margin,
                mtcnn_device=mtcnn_device,
            ),
        )
        val_dataset = datasets.ImageFolder(
            root=train_root,
            transform=fer_eval_transforms(
                image_size=image_size,
                use_mtcnn=use_mtcnn,
                mtcnn_margin=mtcnn_margin,
                mtcnn_device=mtcnn_device,
            ),
        )

        train_dataset = Subset(train_dataset, train_split.indices)
        val_dataset = Subset(val_dataset, val_split.indices)

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

    return train_loader, val_loader
  
if __name__ == "__main__":
    train_loader, val_loader = make_fer_loaders("data/FER13")
    images, labels = next(iter(train_loader))
    print(f"Batch-Shape: {images.shape}")
