"""Generic feed-forward regression network — the training loop that actually
lets the model learn.

Every "learning" step here comes from one line: ``loss.backward()``. PyTorch
walks the computation graph built during the forward pass back to front and
applies the chain rule at each layer (dL/dW2 needs dL/dy first; dL/dW1 needs
dL/dh, which itself needs dL/dy passed back through ReLU and W2) to fill in
every ``parameter.grad``. ``optimizer.step()`` then moves each parameter
against its gradient. No manual derivative was hand-coded — that is the
point of reverse-mode autodiff.

Point this at a CSV with numeric feature columns + one numeric target column
once real regression data is available; test_neural_network.py validates the
pipeline against a tiny synthetic dataset (no accuracy claims).

Usage:
    python neural_network.py --data path/to/data.csv --target y --epochs 200
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint


def build_model(in_features: int = 10) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def load_dataset(csv_path: Path, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    feature_names = [name for name in rows[0] if name != target]
    x = torch.tensor([[float(row[name]) for name in feature_names] for row in rows])
    y = torch.tensor([[float(row[target])] for row in rows])
    return x, y


def split_dataset(x: torch.Tensor, y: torch.Tensor, val_ratio: float, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=generator)
    n_val = max(1, round(len(x) * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def standardize(x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):
    # Gradient descent converges reliably when every input/target sits near
    # unit scale; CSV columns can otherwise be wildly different (e.g. cm vs. kg).
    if mean is None:
        mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std, mean, std


def grad_norm(model: nn.Module) -> float:
    total = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
    return total.sqrt().item()


def run_epoch(model, x, y, criterion, optimizer, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        if train:
            optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        gnorm = 0.0
        if train:
            loss.backward()  # <- chain rule runs here, filling every .grad
            gnorm = grad_norm(model)
            optimizer.step()
    return loss.item(), gnorm


def train(data: Path, target: str, out_dir: Path, epochs: int, lr: float,
          val_ratio: float, patience: int, weight_decay: float, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_dataset(data, target)
    x_train, y_train, x_val, y_val = split_dataset(x, y, val_ratio)

    x_train, x_mean, x_std = standardize(x_train)
    y_train, y_mean, y_std = standardize(y_train)
    x_val, _, _ = standardize(x_val, x_mean, x_std)
    y_val, _, _ = standardize(y_val, y_mean, y_std)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model = build_model(in_features=x.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)
    stopper = EarlyStopper(patience)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
              "y_mean": y_mean.tolist(), "y_std": y_std.tolist()}
    (out_dir / "stats.json").write_text(json.dumps(stats))

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, gnorm = run_epoch(model, x_train, y_train, criterion, optimizer, train=True)
        val_loss, _ = run_epoch(model, x_val, y_val, criterion, optimizer, train=False)
        scheduler.step(val_loss)

        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"grad_norm={gnorm:.4f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "grad_norm": gnorm})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train a feed-forward regression network")
    parser.add_argument("--data", required=True, type=Path, help="CSV with numeric feature columns + target column")
    parser.add_argument("--target", required=True, help="name of the target column in --data")
    parser.add_argument("--out-dir", type=Path, default=Path("neural_network_runs"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.data, args.target, args.out_dir, args.epochs, args.lr,
          args.val_ratio, args.patience, args.weight_decay)


if __name__ == "__main__":
    main()
