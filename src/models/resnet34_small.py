import torch
import torch.nn as nn
from .blocks import BasicBlock


class ResNet34(nn.Module):
    """ResNet-34 for small images (e.g., 64x64)."""

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = base_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-34 stages: [3, 4, 6, 3]
        self.layer1 = self._make_layer(BasicBlock, base_channels, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        """Build one stage while keeping downsampling confined to the first block."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hierarchical features and classify after global pooling."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet34_small(num_classes: int = 6, in_channels: int = 1) -> ResNet34:
    """Helper factory for consistency with other model constructors."""
    return ResNet34(num_classes=num_classes, in_channels=in_channels)


__all__ = ["ResNet34", "resnet34_small"]
