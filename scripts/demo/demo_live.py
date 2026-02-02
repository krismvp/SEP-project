import argparse
import os
import sys
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.constants.emotions import CANON_6
from src.explainability.gradcam import GradCAM
from src.explainability.visualize import overlay_cam_on_image
from src.models.factory import build_model


def _pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def _build_preprocess(img_size: int, in_channels: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=in_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
        ]
    )


def _expand_box(box: Iterable[float], margin: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    x1 -= box_w * margin
    x2 += box_w * margin
    y1 -= box_h * margin
    y2 += box_h * margin
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)
    return (x1, y1, x2, y2)


def _build_mtcnn(device: str | None):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise ImportError(
            "facenet-pytorch is required for MTCNN-based detection. "
            "Install it with: python -m pip install facenet-pytorch"
        ) from exc
    return MTCNN(keep_all=True, device=device or "cpu")


def _detect_faces_mtcnn(
    frame_bgr: np.ndarray, mtcnn, margin: float, threshold: float
) -> List[Tuple[int, int, int, int]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return []
    width, height = pil_img.size
    out: List[Tuple[int, int, int, int]] = []
    for idx, box in enumerate(boxes):
        score = probs[idx] if probs is not None else None
        if score is not None and float(score) < threshold:
            continue
        out.append(_expand_box(box, margin, width, height))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Live emotion recognition demo.")
    parser.add_argument(
        "--weights",
        default="outputs/mixed/ferplus_raf_resnet34_mtcnn_mixup/resnet34_best.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto if omitted)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror effect")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-threshold", type=float, default=0.9)
    parser.add_argument("--mtcnn-device", type=str, default=None)
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--heatmap", type=_parse_bool, default=True)
    parser.add_argument("--gradcam-every", type=int, default=1)
    args = parser.parse_args()

    device = _pick_device(args.device)
    class_names = [name.title() for name in CANON_6]

    model = build_model(args.arch, num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cam = GradCAM(model)
    preprocess = _build_preprocess(args.img_size, args.in_channels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Live-Demo startet... Drücke 'q' zum Beenden.")

    frame_idx = 0
    last_overlays: list[np.ndarray] = []

    if args.no_mtcnn:
        mtcnn = None
        detect_faces = lambda frame: [(0, 0, frame.shape[1], frame.shape[0])]
    else:
        mtcnn = _build_mtcnn(args.mtcnn_device)
        detect_faces = lambda frame: _detect_faces_mtcnn(
            frame, mtcnn, args.mtcnn_margin, args.mtcnn_threshold
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        faces = detect_faces(frame)
        frame_idx += 1
        do_gradcam = args.heatmap and args.gradcam_every > 0 and (
            frame_idx % args.gradcam_every == 0
        )

        for idx, (x1, y1, x2, y2) in enumerate(faces):
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_roi)

            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            if args.heatmap and do_gradcam:
                outputs = model(input_tensor)
            else:
                with torch.no_grad():
                    outputs = model(input_tensor)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf = float(conf.item())
            pred_idx = int(pred_idx.item())

            if args.heatmap:
                overlay_bgr = None
                if do_gradcam:
                    result = cam(input_tensor, class_idx=pred_idx)
                    overlay = overlay_cam_on_image(pil_img, result.cam.numpy())
                    overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                    if idx >= len(last_overlays):
                        last_overlays.append(overlay_bgr)
                    else:
                        last_overlays[idx] = overlay_bgr
                elif idx < len(last_overlays):
                    overlay_bgr = last_overlays[idx]

                if overlay_bgr is not None:
                    overlay_bgr = cv2.resize(overlay_bgr, (x2 - x1, y2 - y1))
                    frame[y1:y2, x1:x2] = overlay_bgr
            label = f"{class_names[pred_idx]} ({conf*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Emotion Recognition Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
