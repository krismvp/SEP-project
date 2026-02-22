import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.training.train_mixed_affectnet_ferplus_raf import (
    train_mixed_affectnet_ferplus_raf,
)


def _parse_domain_probs(value: str) -> list[float]:
    """Parse domain sampling probabilities from a comma-separated string."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--domain-probs must have 3 comma-separated values, e.g. 0.33,0.33,0.34"
        )
    try:
        # Convert string values to floats
        probs = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--domain-probs must contain numeric values."
        ) from exc
    if any(prob < 0 for prob in probs):
        raise argparse.ArgumentTypeError("--domain-probs values must be non-negative.")
    if sum(probs) <= 0:
        raise argparse.ArgumentTypeError("--domain-probs must sum to > 0.")
    return probs


def main() -> None:
    """Main function: Parses command-line arguments and starts training a mixed-domain emotion recognition model across multiple datasets."""
    parser = argparse.ArgumentParser(
        description="Train a mixed-domain AffectNet + FER+ + RAF model with balanced sampling."
    )
    parser.add_argument("--affectnet-dir", default="data/AffectNet")
    parser.add_argument("--fer-data-dir", default="data/ferplus")
    parser.add_argument("--raf-data-dir", default="data/RAF-DB")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weighted-sampler", dest="weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.set_defaults(weighted_sampler=True)
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--use-mtcnn", action="store_true")
    parser.add_argument("--no-mtcnn", action="store_true")
    parser.add_argument("--mtcnn-margin", type=float, default=0.25)
    parser.add_argument("--mtcnn-device", type=str, default="cpu")
    parser.add_argument(
        "--domain-probs",
        type=_parse_domain_probs,
        default=_parse_domain_probs("0.33,0.33,0.34"),
        help="Comma-separated dataset sampling probs: AffectNet,FER,RAF",
    )
    parser.add_argument("--selection-metric", choices=["avg", "min"], default="avg")
    parser.add_argument(
        "--output-dir",
        default="outputs/mixed/affectnet_ferplus_raf_best_generalization",
    )
    args = parser.parse_args()
    if args.no_mtcnn:
        args.use_mtcnn = False

    history = train_mixed_affectnet_ferplus_raf(
        affectnet_data_dir=args.affectnet_dir,
        fer_data_dir=args.fer_data_dir,
        raf_data_dir=args.raf_data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=args.augmentation,
        lr=args.lr,
        pretrained_path=args.pretrained_path,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=args.weighted_sampler,
        use_weighted_loss=not args.no_weighted_loss,
        class_weight_power=args.class_weight_power,
        patience=args.patience,
        seed=args.seed,
        val_split=args.val_split,
        use_mtcnn=args.use_mtcnn,
        mtcnn_margin=args.mtcnn_margin,
        mtcnn_device=args.mtcnn_device,
        domain_probs=args.domain_probs,
        selection_metric=args.selection_metric,
        output_dir=args.output_dir,
    )
    if history["epochs"]:
        print(
            f"Best epoch: {history['best_epoch']} | "
            f"Best score: {history['best_score']:.2f}"
        )

if __name__ == "__main__":
    main()
