import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.resnet_small import ResNet18
from src.explainability.gradcam import GradCAM
from src.explainability.visualize import overlay_cam_on_image

def get_eval_transform(img_size: int = 64, in_channels: int = 1):
    try:
        from src.data.transforms import get_transforms
        return get_transforms(
            train=False,
            img_size=img_size,
            grayscale_to_rgb=(in_channels == 3),
        )
    except Exception:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=in_channels),
            transforms.ToTensor(),
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an input image file")
    ap.add_argument("--weights", required=True, help="Path to resnet18_best.pth")
    ap.add_argument("--num-classes", type=int, default=7)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--out", default="outputs/cam_overlay.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    
    tfm = get_eval_transform(args.img_size, args.in_channels)


    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # (1,C,H,W)

    cam = GradCAM(model)  
    result = cam(x)
    cam.close()

    overlay = overlay_cam_on_image(img, result.cam.detach().cpu().numpy())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    overlay.save(args.out)

    print(f"Predicted class idx: {result.class_idx}")
    print(f"Saved CAM overlay to: {args.out}")


if __name__ == "__main__":
    main()
