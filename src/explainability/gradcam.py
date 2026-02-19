from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    """Container keeps visualization output and prediction context together."""
    cam: torch.Tensor          # (H, W) in [0,1] on CPU
    class_idx: int
    logits: torch.Tensor       # (num_classes,) on CPU


def _find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    """Default to the last conv layer because it is most class-discriminative."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No nn.Conv2d layer found in model.")
    return last


class GradCAM:
    """
    Grad-CAM for image classification models.
    Works with any model that has at least one Conv2d layer.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        self.target_layer = target_layer if target_layer is not None else _find_last_conv2d(model)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # forward hook -> activations
        def fwd_hook(_module, _inp, out):
            self._activations = out

        # backward hook -> gradients w.r.t activations
        def bwd_hook(_module, grad_in, grad_out):
            # grad_out[0] has same shape as activations
            self._gradients = grad_out[0]

        self._h1 = self.target_layer.register_forward_hook(fwd_hook)
        self._h2 = self.target_layer.register_full_backward_hook(bwd_hook)

    def close(self) -> None:
        """Remove hooks explicitly to prevent stale references in long sessions."""
        self._h1.remove()
        self._h2.remove()

    @torch.no_grad()
    def _infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Utility inference path for callers that only need logits."""
        return self.model(x)

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> GradCAMResult:
        """
        x: (1, C, H, W)
        """
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("Expected input x with shape (1, C, H, W).")

        device = next(self.model.parameters()).device
        x = x.to(device)

        # Forward (with grads enabled)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        # Backprop only the selected class score to get class-specific saliency.
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # activations/grads: (1, K, h, w)
        acts = self._activations
        grads = self._gradients

        # weights: global-average-pool over spatial dims -> (1, K, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # weighted sum -> (1, 1, h, w)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        # Use absolute activations so CAM is visible even when contributions are negative.
        cam = cam.abs()

        # upsample to input size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # normalize to [0,1]
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return GradCAMResult(
            cam=cam.detach().cpu(),
            class_idx=class_idx,
            logits=logits[0].detach().cpu(),
        )
