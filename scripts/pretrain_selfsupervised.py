import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.pretrain_selfsupervised import pretrain_ssl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining (SimCLR) on CelebA."
    )
    parser.add_argument("--data-dir", default="data/celebA")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-path", default="outputs/pretrained_backbone.pth")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--no-face-crop", action="store_true")
    parser.add_argument("--face-padding", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    pretrain_ssl(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        num_workers=args.num_workers,
        seed=args.seed,
        output_path=args.output_path,
        image_size=args.image_size,
        split=args.split,
        crop_faces=not args.no_face_crop,
        face_padding=args.face_padding,
        log_interval=args.log_interval,
    )

    print(f"Saved backbone to: {args.output_path}")


if __name__ == "__main__":
    main()
