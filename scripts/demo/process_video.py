import argparse
import logging
import os
import sys
import time
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

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


def _build_preprocess(img_size: int, in_channels: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=in_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
        ]
    )


def _expand_box(
    box: Iterable[float], margin: float, width: int, height: int
) -> Tuple[int, int, int, int]:
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


def _rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


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
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]] | None]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    try:
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    except TypeError:
        boxes, probs = mtcnn.detect(pil_img)
        landmarks = None

    if boxes is None or len(boxes) == 0:
        return ([], None)
    width, height = pil_img.size
    out_boxes: List[Tuple[int, int, int, int]] = []
    out_landmarks: List[Tuple[int, int]] | None = None
    if landmarks is not None:
        out_landmarks = []
    for i, box in enumerate(boxes):
        score = probs[i] if probs is not None else None
        if score is not None and float(score) < threshold:
            continue
        out_boxes.append(_expand_box(box, margin, width, height))
        if landmarks is not None:
            pts = [(int(x), int(y)) for x, y in landmarks[i]]
            out_landmarks.append(pts)
    return (out_boxes, out_landmarks)


def _get_class_names(num_classes: int) -> List[str]:
    if num_classes == len(CANON_6):
        return [name.capitalize() for name in CANON_6]
    return [f"Class {i}" for i in range(num_classes)]


def _draw_landmarks(frame: np.ndarray, pts: List[Tuple[int, int]]) -> None:
    if len(pts) < 5:
        return
    left_eye, right_eye, nose, mouth_left, mouth_right = pts[:5]
    color = (0, 255, 0)
    for (x, y) in pts[:5]:
        cv2.circle(frame, (x, y), 2, color, -1)
    cv2.line(frame, left_eye, right_eye, color, 1)
    cv2.line(frame, left_eye, nose, color, 1)
    cv2.line(frame, right_eye, nose, color, 1)
    cv2.line(frame, nose, mouth_left, color, 1)
    cv2.line(frame, nose, mouth_right, color, 1)
    cv2.line(frame, mouth_left, mouth_right, color, 1)


def _draw_report(
    frame: np.ndarray,
    faces: int,
    reports: List[Tuple[int, List[str], List[float]]],
) -> None:
    x = 20
    y = 30
    color = (200, 0, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Number of Faces : {faces}", (x, y), font, 0.6, color, 2)
    y += 24
    cv2.putText(frame, "---------------------", (x, y), font, 0.6, color, 2)
    y += 24

    for idx, class_names, probs in reports:
        cv2.putText(frame, f"Emotional report : Face #{idx}", (x, y), font, 0.6, color, 2)
        y += 22
        for name, prob in zip(class_names, probs):
            cv2.putText(
                frame,
                f"{name} : {prob:.3f}",
                (x, y),
                font,
                0.5,
                color,
                1,
            )
            y += 20
        y += 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a video and save annotated output with emotion predictions."
    )
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="outputs/processed_video.mp4")
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
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Rotate input frames in degrees",
    )
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-threshold", type=float, default=0.9)
    parser.add_argument("--mtcnn-device", type=str, default=None)
    parser.add_argument("--no-report", action="store_true", help="Disable text report overlay")
    parser.add_argument("--log-every", type=int, default=30, help="Log progress every N frames")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument(
        "--heatmap",
        choices=["none", "gradcam"],
        default="none",
        help="Overlay saliency heatmap on face region",
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.45)
    args = parser.parse_args()

    logger = logging.getLogger("process_video")
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    handler = (
        logging.FileHandler(args.log_file)
        if args.log_file
        else logging.StreamHandler()
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    device = _pick_device(args.device)
    class_names = _get_class_names(args.num_classes)

    model = build_model(args.arch, num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preprocess = _build_preprocess(args.img_size, args.in_channels)
    cam = GradCAM(model) if args.heatmap == "gradcam" else None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    first_ret, first_frame = cap.read()
    if not first_ret:
        raise FileNotFoundError(f"Could not read first frame: {args.input}")

    rotation = args.rotate
    first_frame = _rotate_frame(first_frame, rotation)
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if args.no_mtcnn:
        mtcnn = None
        detect_faces = lambda frame: ([(0, 0, frame.shape[1], frame.shape[0])], None)
    else:
        mtcnn = _build_mtcnn(args.mtcnn_device)
        detect_faces = lambda frame: _detect_faces_mtcnn(
            frame, mtcnn, args.mtcnn_margin, args.mtcnn_threshold
        )

    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info(
        "Model: arch=%s weights=%s device=%s",
        args.arch,
        args.weights,
        device.type,
    )
    if args.no_mtcnn:
        logger.info("Detector: full-frame (no mtcnn)")
    else:
        logger.info(
            "Detector: mtcnn (mtcnn_device=%s, threshold=%.2f)",
            args.mtcnn_device or "cpu",
            args.mtcnn_threshold,
        )
    logger.info(
        "Video: %dx%d @ %.2f fps (frames=%s) rotation=%d",
        width,
        height,
        fps,
        total_frames,
        rotation,
    )

    frame_idx = 0
    face_total = 0
    start_time = time.time()

    frame = first_frame
    while True:
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _rotate_frame(frame, rotation)
        frame_idx += 1

        (faces, landmarks) = detect_faces(frame)
        reports: List[Tuple[int, List[str], List[float]]] = []
        face_total += len(faces)

        for idx, (x1, y1, x2, y2) in enumerate(faces, start=1):
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_roi)
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            label = class_names[pred_idx]
            if cam is not None:
                result = cam(input_tensor, class_idx=pred_idx)
                overlay = overlay_cam_on_image(pil_img, result.cam.numpy(), alpha=args.heatmap_alpha)
                overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                overlay_bgr = cv2.resize(overlay_bgr, (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = overlay_bgr

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face #{idx}",
                (x1, max(0, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (x2 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if landmarks is not None and idx - 1 < len(landmarks):
                _draw_landmarks(frame, landmarks[idx - 1])

            reports.append((idx, class_names, probs.tolist()))

        if not args.no_report:
            _draw_report(frame, len(faces), reports)

        out.write(frame)
        frame = None

        if args.log_every > 0 and frame_idx % args.log_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            proc_fps = frame_idx / elapsed
            avg_faces = face_total / frame_idx
            if total_frames > 0:
                remaining = total_frames - frame_idx
                eta = remaining / max(proc_fps, 1e-6)
                logger.info(
                    "Frame %d/%d | faces=%d | avg_faces=%.2f | fps=%.2f | eta=%.1fs",
                    frame_idx,
                    total_frames,
                    len(faces),
                    avg_faces,
                    proc_fps,
                    eta,
                )
            else:
                logger.info(
                    "Frame %d | faces=%d | avg_faces=%.2f | fps=%.2f",
                    frame_idx,
                    len(faces),
                    avg_faces,
                    proc_fps,
                )

    cap.release()
    out.release()
    if cam is not None:
        cam.close()
    elapsed_total = max(time.time() - start_time, 1e-6)
    logger.info(
        "Done. frames=%d avg_faces=%.2f time=%.1fs fps=%.2f",
        frame_idx,
        (face_total / frame_idx) if frame_idx else 0.0,
        elapsed_total,
        (frame_idx / elapsed_total) if frame_idx else 0.0,
    )
    print(f"Saved processed video to: {args.output}")


if __name__ == "__main__":
    main()
