import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline_cnn import SimpleCNN


def train():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_data = datasets.ImageFolder("data/FER13/train", transform=transform)
    val_data = datasets.ImageFolder("data/FER13/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(4):
        model.train()
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        acc = 100 * correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
          for imgs, labels in val_loader:
              imgs, labels = imgs.to(device), labels.to(device)
              outputs = model(imgs)
              _, preds = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (preds == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(
            f"Epoch {epoch+1}: "
            f"Train Acc = {acc:.2f}% | "
            f"Val Acc = {val_acc:.2f}%"
        )

    torch.save(model.state_dict(), "baseline_cnn.pth")


if __name__ == "__main__":
    train()
