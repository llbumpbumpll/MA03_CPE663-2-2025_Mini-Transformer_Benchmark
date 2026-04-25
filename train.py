import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import MAX_LEN, VOCAB, create_dataloaders
from model import MiniTransformerClassifier
from utils import AverageMeter, binary_metrics, count_params, set_seed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(args):
    return MiniTransformerClassifier(
        vocab_size=len(VOCAB),
        max_len=MAX_LEN,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_positional_encoding=not args.no_positional_encoding,
        pooling=args.pooling,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), tokens.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    y_true = []
    y_pred = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        loss_meter.update(loss.item(), tokens.size(0))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = loss_meter.avg
    return metrics


def make_class_weight(train_loader, device):
    labels = train_loader.dataset.df["label"]
    counts = labels.value_counts().sort_index()
    total = counts.sum()
    weights = [total / (2 * counts.get(i, 1)) for i in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_history(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return history_path


def train_model(args, variant_name="single"):
    set_seed(args.seed)
    device = get_device()
    train_loader, validation_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args).to(device)
    class_weight = make_class_weight(train_loader, device) if args.class_weight else None
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_precision": validation_metrics["precision"],
            "val_recall": validation_metrics["recall"],
            "val_f1": validation_metrics["f1"],
        }
        history.append(row)

        if validation_metrics["f1"] > best_val_f1:
            best_val_f1 = validation_metrics["f1"]
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        if not args.quiet:
            print(
                f"{variant_name} epoch {epoch:02d}/{args.epochs} "
                f"loss={train_loss:.4f} val_acc={validation_metrics['accuracy']:.4f} "
                f"val_f1={validation_metrics['f1']:.4f}"
            )

    train_time = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = evaluate(model, validation_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    output_dir = Path(args.output_dir) / variant_name
    history_path = save_history(history, output_dir)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    result = {
        "variant": variant_name,
        "positional_encoding": not args.no_positional_encoding,
        "heads": args.num_heads,
        "layers": args.num_layers,
        "embed_dim": args.embed_dim,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "class_weight": args.class_weight,
        "parameters": count_params(model),
        "train_time_seconds": train_time,
        "val_accuracy": validation_metrics["accuracy"],
        "val_precision": validation_metrics["precision"],
        "val_recall": validation_metrics["recall"],
        "val_f1": validation_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a from-scratch mini Transformer classifier.")
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
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["first", "mean"], default="first")
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result, _ = train_model(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
