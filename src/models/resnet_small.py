import torch
import torch.nn as nn
from .blocks import BasicBlock


class ResNet18(nn.Module):
    """ResNet-18 for small images (e.g., 64x64).

    Notes:
    - Supports grayscale (in_channels=1) or RGB (in_channels=3) inputs.
    - Uses BasicBlock (expansion=1). The code also works with Bottleneck-style
      blocks if you later swap the block and set the correct expansion.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = base_channels

        # For 64x64 inputs we typically keep stride=1 here (unlike ImageNet ResNet).
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

        # 4 stages with 2 blocks each (ResNet-18)
        self.layer1 = self._make_layer(BasicBlock, base_channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        # First block may downsample (stride>1). Remaining blocks keep stride=1.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            # After a block, the number of channels is out_channels * expansion.
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
