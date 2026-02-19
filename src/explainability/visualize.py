from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def overlay_cam_on_image(
    img_rgb: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a normalized CAM heatmap with the original image for inspection."""
    """
    img_rgb: PIL RGB image
    cam: (H, W) float in [0,1]
    """
    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")

    cam = np.clip(cam, 0.0, 1.0)
    cam_img = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_img).resize(img_rgb.size, resample=Image.BILINEAR)

    cam_np = np.array(cam_img)
    # OpenCV colormap is used for consistent, high-contrast visual debugging.
    heat_bgr = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    base = np.array(img_rgb).astype(np.float32)
    heat = heat_rgb.astype(np.float32)

    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
