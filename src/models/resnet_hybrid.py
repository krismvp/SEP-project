import torch
import torch.nn as nn
from .resnet_small import ResNet18

class ResNetHybrid(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CNN backbone
        self.backbone = ResNet18(
            num_classes=num_classes,
            in_channels=in_channels,
        )

        # project CNN channels -> transformer dimension
        self.proj = nn.Linear(self.backbone.feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features: [B, C, H, W]
        feats = self.backbone.forward_features(x)

        B, C, H, W = feats.shape

        # flatten spatial dims -> tokens
        tokens = feats.view(B, C, H * W).transpose(1, 2)

        tokens = self.proj(tokens)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # global token pooling
        pooled = tokens.mean(dim=1)

        return self.classifier(pooled)
