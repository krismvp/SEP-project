import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from .transforms import fer_train_transforms, fer_eval_transforms

def make_fer_loaders(data_path, batch_size=64):
    full_train_dataset = datasets.ImageFolder(
        root=f"{data_path}/train", 
        transform=fer_train_transforms()
    )
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
  
if __name__ == "__main__":
        train_loader, val_loader = make_fer_loaders("data/FER13")
        images, labels = next(iter(train_loader))
        print(f"Batch-Shape: {images.shape}")

