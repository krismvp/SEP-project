from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image
import torch


def _get_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def _center_crop(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


class MTCNNCrop:
    def __init__(self, image_size: int = 64, margin: float = 0.25, device: Optional[str] = None):
        self.image_size = image_size
        self.margin = max(0.0, margin)
        self.device = device or "cpu"
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required for MTCNN preprocessing. "
                "Install it with: python -m pip install -r requirements.txt"
            ) from exc
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.mtcnn.eval()
        self._resample = _get_resample()

    def _largest_bbox(self, boxes) -> Optional[Tuple[float, float, float, float]]:
        if boxes is None or len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(areas.argmax())
        return tuple(float(v) for v in boxes[idx])

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        base = img
        if base.mode != "RGB":
            base = base.convert("RGB")
        width, height = base.size
        with torch.no_grad():
            boxes, _ = self.mtcnn.detect(base)
        bbox = self._largest_bbox(boxes)
        if bbox is None:
            return _center_crop(base).resize((self.image_size, self.image_size), self._resample)

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        x1 -= box_w * self.margin
        x2 += box_w * self.margin
        y1 -= box_h * self.margin
        y2 += box_h * self.margin

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(width), x2)
        y2 = min(float(height), y2)

        if x2 <= x1 or y2 <= y1:
            return _center_crop(base).resize((self.image_size, self.image_size), self._resample)

        cropped = base.crop((int(x1), int(y1), int(x2), int(y2)))
        return cropped.resize((self.image_size, self.image_size), self._resample)


__all__ = ["MTCNNCrop"]
