import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

from train import train_model


VARIANTS = [
    {
        "variant": "A_pos_1head_1layer",
        "no_positional_encoding": False,
        "num_heads": 1,
        "num_layers": 1,
    },
    {
        "variant": "B_pos_4head_1layer",
        "no_positional_encoding": False,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "variant": "C_no_pos_4head_1layer",
        "no_positional_encoding": True,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "variant": "D_pos_4head_2layer",
        "no_positional_encoding": False,
        "num_heads": 4,
        "num_layers": 2,
    },
]


def namespace_from_args(args, overrides):
    values = vars(args).copy()
    values.update(overrides)
    values.pop("variant", None)
    return SimpleNamespace(**values)


def write_results_csv(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_results.csv"
    fieldnames = [
        "variant",
        "positional_encoding",
        "heads",
        "layers",
        "parameters",
        "train_time_seconds",
        "val_accuracy",
        "val_f1",
        "test_accuracy",
        "test_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})
    return path


def plot_curves(histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_curves.png"

    plt.figure(figsize=(10, 5))
    for variant, history in histories.items():
        epochs = [row["epoch"] for row in history]
        val_acc = [row["val_accuracy"] for row in history]
        plt.plot(epochs, val_acc, marker="o", label=variant)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Mini Transformer validation accuracy by epoch")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_log_loss_curves(histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_loss_log_curves.png"

    plt.figure(figsize=(10, 5))
    for variant, history in histories.items():
        epochs = [row["epoch"] for row in history]
        train_loss = [max(row["train_loss"], 1e-8) for row in history]
        plt.plot(epochs, train_loss, marker="o", label=variant)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (log scale)")
    plt.yscale("log")
    plt.title("Mini Transformer training loss by epoch")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def run_benchmark(args):
    output_dir = Path(args.output_dir)
    results = []
    histories = {}

    for variant_config in VARIANTS:
        config = deepcopy(variant_config)
        variant_name = config.pop("variant")
        variant_args = namespace_from_args(args, config)
        result, history = train_model(variant_args, variant_name=variant_name)
        results.append(result)
        histories[variant_name] = history

    results_path = write_results_csv(results, output_dir)
    curve_path = plot_curves(histories, output_dir)
    log_loss_curve_path = plot_log_loss_curves(histories, output_dir)

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": results,
                "results_csv": str(results_path),
                "curve_png": str(curve_path),
                "log_loss_curve_png": str(log_loss_curve_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the required mini Transformer benchmark.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--validation-csv", default="data/validation.csv")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["first", "mean"], default="first")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_benchmark(args)
    for result in results:
        print(
            f"{result['variant']}: val_acc={result['val_accuracy']:.4f}, "
            f"test_acc={result['test_accuracy']:.4f}, test_f1={result['test_f1']:.4f}, "
            f"time={result['train_time_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
