from src.models.resnet18_small import ResNet18
from src.models.resnet34_small import resnet34_small


def build_model(arch: str, num_classes: int, in_channels: int = 1):
    """Centralize architecture selection so training code stays config-driven."""
    key = arch.lower()
    if key == "resnet18":
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    if key == "resnet34":
        return resnet34_small(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown architecture: {arch}")


__all__ = ["build_model"]
