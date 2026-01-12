import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FERDataset(Dataset):
    def __init__(self, df, indices, transform=None):
        self.data = df.iloc[indices].reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels = self.data.loc[idx, "pixels"]
        label = int(self.data.loc[idx, "emotion"])
        img = np.array(pixels.split(), dtype=np.uint8).reshape(48,48)
        img = Image.fromarray(img, mode="L")
        if self.transform:
            img = self.transform(img)
        return img, label
    
