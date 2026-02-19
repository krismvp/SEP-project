from __future__ import annotations

from collections.abc import Sized
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def get_device() -> torch.device:
    """Prefer CUDA/MPS automatically so training scripts keep one shared device policy."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_num_classes(dataset) -> int:
    """Infer class count from common dataset wrappers used in our loaders."""
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    raise ValueError("Unable to infer number of classes from dataset.")


def set_seed(seed: int) -> None:
    """Set all relevant torch seeds so experiments are reproducible enough to compare."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_from_sample(sample: object) -> Optional[int]:
    """Normalize different sample formats to one integer label extraction path."""
    label = getattr(sample, "label", None)
    if label is not None:
        return int(label)
    if isinstance(sample, (list, tuple)) and len(sample) > 1:
        return int(sample[1])
    return None


def _extract_labels(dataset) -> List[int]:
    """Collect labels without assuming one specific dataset implementation."""
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
        samples = getattr(base, "samples", None)
        if samples is not None:
            labels: List[int] = []
            for idx in indices:
                label = _label_from_sample(samples[idx])
                if label is None:
                    raise ValueError("Dataset samples must provide labels.")
                labels.append(label)
            return labels
        targets = getattr(base, "targets", None)
        if targets is not None:
            return [int(targets[idx]) for idx in indices]
        labels: List[int] = []
        for idx in indices:
            label = _label_from_sample(base[idx])
            if label is None:
                raise ValueError("Dataset samples must provide labels.")
            labels.append(label)
        return labels
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        labels: List[int] = []
        for sample in samples:
            label = _label_from_sample(sample)
            if label is None:
                raise ValueError("Dataset samples must provide labels.")
            labels.append(label)
        return labels
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(t) for t in targets]
    if not isinstance(dataset, Sized):
        raise ValueError("Dataset must implement __len__ for label extraction.")
    labels: List[int] = []
    for idx in range(len(dataset)):
        label = _label_from_sample(dataset[idx])
        if label is None:
            raise ValueError("Dataset samples must provide labels.")
        labels.append(label)
    return labels


def _compute_class_weights(
    labels: List[int], num_classes: int, power: float = 1.0
) -> torch.Tensor:
    """Use inverse-frequency weights with optional damping via `power`."""
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    counts = counts.float().clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    if power != 1.0:
        weights = weights.pow(power)
    return weights


def _load_pretrained_backbone(model: nn.Module, checkpoint_path: str) -> None:
    """Load reusable backbone weights while intentionally skipping classifier head params."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a state_dict key.")
    state.pop("fc.weight", None)
    state.pop("fc.bias", None)
    conv1_weight = state.get("conv1.weight")
    if isinstance(conv1_weight, torch.Tensor):
        in_channels = _get_conv1_in_channels(model)
        state = _adapt_conv1(state, in_channels)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"Missing keys in checkpoint (expected for fc): {missing}")


def _adapt_conv1(state: Dict[str, torch.Tensor], in_channels: int) -> Dict[str, torch.Tensor]:
    """Adapt first-layer channels so RGB and grayscale checkpoints stay interoperable."""
    weight = state["conv1.weight"]
    if weight.shape[1] == in_channels:
        return state
    if in_channels == 1 and weight.shape[1] == 3:
        state["conv1.weight"] = weight.mean(dim=1, keepdim=True)
        return state
    if in_channels == 3 and weight.shape[1] == 1:
        state["conv1.weight"] = weight.repeat(1, 3, 1, 1) / 3.0
        return state
    return state


def _get_conv1_in_channels(model: nn.Module) -> int:
    """Read expected input channels from the canonical first conv layer."""
    conv1 = getattr(model, "conv1", None)
    if isinstance(conv1, nn.Conv2d):
        return int(conv1.in_channels)
    raise ValueError("Model does not have a Conv2d conv1 layer.")


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze non-classifier layers for head-only warmup/fine-tuning phases."""
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def _unfreeze_backbone(model: nn.Module) -> None:
    """Re-enable full-model training after any temporary backbone freeze."""
    for param in model.parameters():
        param.requires_grad = True


def _build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create split LR groups so new heads can learn faster than pretrained backbones."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return optim.AdamW(param_groups, weight_decay=weight_decay)


__all__ = [
    "get_device",
    "infer_num_classes",
    "set_seed",
    "_extract_labels",
    "_compute_class_weights",
    "_build_optimizer",
    "_freeze_backbone",
    "_load_pretrained_backbone",
    "_unfreeze_backbone",
]
