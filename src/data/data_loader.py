import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from .transforms import (
      raf_train_transforms, raf_eval_transforms,
      fer_train_transforms, fer_eval_transforms
)
RAF_MAP = {
    '1': 0, '2': 1, '4': 2, '5': 3, '6': 4, '7': 5
}

FER_MAP = {
    'surprise': 0, 'fear': 1, 'happy': 2, 'sad': 3, 'anger': 4, 'neutral': 5
}
def filter_and_map(dataset, mapping):
    indices = []
    for i, (path, original_idx) in enumerate(dataset.imgs):
        folder_name = dataset.classes[original_idx]
        if folder_name in mapping:
            indices.append(i)
            dataset.targets[i] = mapping[folder_name]
    return indices

def make_raf_loaders(data_path, batch_size=64):
    train_data = datasets.ImageFolder(root=f"{data_path}/train", transform=raf_train_transforms())
    val_data = datasets.ImageFolder(root=f"{data_path}/train", transform=raf_eval_transforms())

    valid_indices = filter_and_map(train_data, RAF_MAP)
    _ = filter_and_map(val_data, RAF_MAP)
   
    train_idx, val_idx = train_test_split(
        valid_indices,
        test_size=0.1,
        stratify=[train_data.targets[i] for i in valid_indices],
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

    valid_indices = filter_and_map(train_data, FER_MAP)
    _ = filter_and_map(val_data, FER_MAP)

    train_idx, val_idx = train_test_split(
          valid_indices,
          test_size=0.1,
          stratify=[train_data.targets[i] for i in valid_indices],
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

