import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from .transforms import (
      raf_train_transforms, raf_eval_transforms,
      fer_train_transforms, fer_eval_transforms
)

def make_raf_loaders(data_path, batch_size=64):
    train_data = datasets.ImageFolder(root=f"{data_path}/train", transform=raf_train_transforms())
    val_data = datasets.ImageFolder(root=f"{data_path}/train", transform=raf_eval_transforms())
   
    train_idx, val_idx = train_test_split(
          range(len(train_data)),
          test_size=0.1,
          stratify=train_data.targets,
          random_state=42
    )

    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
     
    val_loader = DataLoader(
        Subset(val_data, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader


def make_fer_loaders(data_path, batch_size=64):
    train_data = datasets.ImageFolder(root=f"{data_path}/train", transform=fer_train_transforms())
    val_data = datasets.ImageFolder(root=f"{data_path}/train", transform=fer_eval_transforms())

    train_idx, val_idx = train_test_split(
          range(len(train_data)),
          test_size=0.1,
          stratify=train_data.targets,
          random_state=42
    )

    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        Subset(val_data, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
  
if __name__ == "__main__":
        train_loader, val_loader = make_raf_loaders("data/DATASET")
        images, labels = next(iter(train_loader))
        print(f"Batch-Shape: {images.shape}")

