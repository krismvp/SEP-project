import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def load_metrics_from_json(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    required_keys = ["train_losses", "val_losses", "train_accs", "val_accs"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing key in JSON: {key}")
    return data

def compare_models(model_configs: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    models_data = []
    for config in model_configs:
        name = config["name"]
        json_path = config["json_path"]
        print(f"Loading {name} from: {json_path}")
        metrics = load_metrics_from_json(json_path)
        models_data.append({"name": name, "metrics": metrics})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison: Training Curves", fontsize=14, weight="bold")
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for idx, model in enumerate(models_data):
        name = model["name"]
        metrics = model["metrics"]
        epochs = range(1, len(metrics["train_accs"]) + 1)
        color = colors[idx % len(colors)]
        
        axes[0].plot(
            epochs, metrics["train_accs"],
            label=f"{name} (Train)", linestyle="-", color=color, linewidth=2
        )
        axes[0].plot(
            epochs, metrics["val_accs"],
            label=f"{name} (Val)", linestyle="--", color=color, linewidth=2
        )
    
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title("Accuracy Comparison", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    for idx, model in enumerate(models_data):
        name = model["name"]
        metrics = model["metrics"]
        epochs = range(1, len(metrics["train_losses"]) + 1)
        color = colors[idx % len(colors)]
        
        axes[1].plot(
            epochs, metrics["train_losses"],
            label=f"{name} (Train)", linestyle="-", color=color, linewidth=2
        )
        axes[1].plot(
            epochs, metrics["val_losses"],
            label=f"{name} (Val)", linestyle="--", color=color, linewidth=2
        )
    
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Loss Comparison", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Comparison plots saved to: {plot_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare training curves from multiple models."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model specs in format 'name:path/to/metrics.json'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save comparison plots"
    )
    
    args = parser.parse_args()
    
    model_configs = []
    for model_spec in args.models:
        if ":" not in model_spec:
            print(f"Error: Invalid format '{model_spec}'")
            print("Expected format: 'ModelName:/path/to/metrics.json'")
            sys.exit(1)
        
        name, json_path = model_spec.split(":", 1)
        model_configs.append({"name": name, "json_path": json_path})
    
    compare_models(model_configs, args.output_dir)

if __name__ == "__main__":
    main()

