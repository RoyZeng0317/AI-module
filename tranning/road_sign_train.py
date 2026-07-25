"""Taiwan road-sign / driving-test sign recognition — training scaffold.

Point this at the output of data_split.py (a folder containing train/,
val/, test/ subfolders, each in ImageFolder layout: one subdirectory per
sign class) once real road-sign images are available. Right now, with no
data provided yet, this only defines the architecture and training loop;
it is smoke-tested against a tiny synthetic dataset in
test_road_sign_train.py (no accuracy claims — just "the pipeline runs").

Design choices made specifically to avoid overfitting AND underfitting on
what will likely be a small, class-imbalanced sign dataset:

  Underfitting:
    - Transfer learning (MobileNetV2 pretrained on ImageNet) instead of a
      from-scratch CNN — the backbone already has useful low/mid-level
      features, so the small classifier head can converge even with few
      images per class.
    - ReduceLROnPlateau backs off the learning rate only when validation
      loss stalls, instead of a fixed schedule that might decay too early.
    - --unfreeze-backbone flag: if train accuracy stays low, the frozen
      backbone is the usual suspect — unfreeze it to add capacity.

  Overfitting:
    - Augmentation deliberately excludes horizontal/vertical flips: signs
      are directional (arrows, keep-left/keep-right, one-way), so flipping
      would teach the model the wrong class. Rotation/color-jitter/affine
      only.
    - Dropout in the classifier head + weight decay (AdamW) + label
      smoothing.
    - Early stopping on validation loss (patience-based), restoring the
      best checkpoint rather than the last one.
    - Per-epoch train/val accuracy gap is printed with a warning so a human
      can catch overfitting even if early stopping hasn't kicked in yet.

Usage:
    python road_sign_train.py --data-dir path/to/split_output --epochs 30
    python road_sign_train.py --data-dir path/to/split_output --unfreeze-backbone --lr 1e-4
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError("train/ and val/ do not have the same set of class subfolders")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int, freeze_backbone: bool, dropout: float = 0.3):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train(data_dir: Path, out_dir: Path, epochs: int, batch_size: int, lr: float,
          patience: int, image_size: int, freeze_backbone: bool, dropout: float,
          weight_decay: float, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = build_dataloaders(data_dir, image_size, batch_size)
    model = build_model(len(classes), freeze_backbone, dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        gap = train_acc - val_acc
        warning = ""
        if gap > 0.15:
            warning = "  [warning: train/val accuracy gap suggests overfitting]"
        elif train_acc < 0.4 and epoch >= max(3, epochs // 3):
            warning = "  [warning: low train accuracy this far in — possible underfitting; try --unfreeze-backbone]"

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, classes, history


def main():
    parser = argparse.ArgumentParser(description="Train a road-sign classifier")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="folder with train/ and val/ subfolders (output of data_split.py)")
    parser.add_argument("--out-dir", type=Path, default=Path("road_sign_runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()

    train(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.lr,
          args.patience, args.image_size, args.freeze_backbone, args.dropout, args.weight_decay)


if __name__ == "__main__":
    main()
