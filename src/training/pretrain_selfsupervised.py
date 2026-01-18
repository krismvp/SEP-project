import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ssl_data import make_ssl_loader
from src.models.resnet_small import ResNet18


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimCLR(nn.Module):
    """Backbone + projection head for contrastive pretraining."""

    def __init__(self, backbone: nn.Module, projection_dim: int = 128):
        super().__init__()
        if not hasattr(backbone, "fc"):
            raise ValueError("Backbone must have an fc layer to infer feature dim.")
        fc = backbone.fc
        if not isinstance(fc, nn.Linear):
            raise ValueError("Backbone fc must be nn.Linear to infer feature dim.")
        feat_dim = fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        pos_idx = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (
            2 * batch_size
        )
        positives = sim[torch.arange(2 * batch_size, device=z.device), pos_idx]
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    iterator = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for step, (x1, x2) in enumerate(iterator, start=1):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        z1 = model(x1)
        z2 = model(x2)
        loss = loss_fn(z1, z2)

        loss.backward()
        optimizer.step()

        batch_size = x1.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if log_interval > 0 and step % log_interval == 0:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


def pretrain_ssl(
    data_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    projection_dim: int,
    num_workers: int,
    seed: int,
    output_path: str,
    image_size: int = 64,
    split: str = "train",
    crop_faces: bool = True,
    face_padding: float = 0.1,
    log_interval: int = 0,
) -> Dict[str, List[float]]:
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loader = make_ssl_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        image_size=image_size,
        split=split,
        crop_faces=crop_faces,
        face_padding=face_padding,
    )

    backbone = ResNet18(in_channels=3)
    model = SimCLR(backbone=backbone, projection_dim=projection_dim).to(device)

    loss_fn = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device,
            epoch=epoch,
            total_epochs=epochs,
            log_interval=log_interval,
        )
        scheduler.step()
        losses.append(loss)
        print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.backbone.state_dict(), output_path)

    return {"losses": losses}


__all__ = ["pretrain_ssl"]
