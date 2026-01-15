# src/explainability/visualize.py
from __future__ import annotations
from typing import Tuple

import numpy as np
from PIL import Image


def overlay_cam_on_image(
    img_rgb: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    img_rgb: PIL RGB image
    cam: (H, W) float in [0,1]
    """
    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")

    cam = np.clip(cam, 0.0, 1.0)
    cam_img = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_img).resize(img_rgb.size, resample=Image.BILINEAR)

    # simple "heatmap": map grayscale -> red channel (minimal, ohne extra libs)
    heat = np.zeros((cam_img.size[1], cam_img.size[0], 3), dtype=np.uint8)
    heat[..., 0] = np.array(cam_img)  # red intensity

    base = np.array(img_rgb).astype(np.float32)
    heat = heat.astype(np.float32)

    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
