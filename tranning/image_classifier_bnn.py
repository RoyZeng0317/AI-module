"""Generic Bayesian image classifier — custom CNN trained from scratch +
Monte Carlo Dropout, for any "one subfolder per class" image dataset (not
tied to road signs specifically; see road_sign_train.py for that dedicated
pipeline, which this deliberately does not replace or modify).

Point this at the output of data_split.py (a folder containing train/,
val/, test/ subfolders, each in ImageFolder layout) once real images are
available. Right now, with no data provided yet, this only defines the
architecture and training loop; it is smoke-tested against a tiny synthetic
dataset in test_image_classifier_bnn.py (no accuracy claims — just "the
pipeline runs").

Architecture and anti-overfitting/underfitting choices mirror
road_sign_train.py (transfer learning so a small classifier head can
converge with few images per class, ReduceLROnPlateau, --unfreeze-backbone,
dropout + weight decay via AdamW, label smoothing, early stopping restoring
the best checkpoint, per-epoch train/val accuracy gap warning) — see that
module's docstring for the full rationale. Two differences, because this
script is meant for generic photos rather than direction-sensitive signs:

  - Random horizontal flip is ON by default (most everyday object classes
    are left/right-symmetric in meaning). Pass --no-hflip for datasets where
    orientation IS the class (arrows, text, handedness, road signs, etc.).

  - Bayesian inference via Monte Carlo Dropout (see bayesian_utils.py):
    classify_image() runs the trained Dropout layer stochastically
    (mc_dropout_mode) across `mc_samples` forward passes instead of one
    deterministic pass, and reports a confidence (mean top-class
    probability) and predictive entropy alongside the predicted label —
    an honest "how sure is the model" signal rather than a single greedy
    guess that looks equally confident whether it's right or not. This is
    the same trick used by chats.py's mc_chat_reply(); see
    bayesian_utils.py's module docstring for why MC Dropout rather than a
    full variational (Bayes by Backprop) network was chosen — same
    RTX 4060 8GB budget reasoning applies here.

Usage:
    python image_classifier_bnn.py --data-dir path/to/split_output --grayscale
    python image_classifier_bnn.py --data-dir path/to/split_output --epochs 30
    python image_classifier_bnn.py --data-dir path/to/split_output --no-hflip --epochs 30
    python image_classifier_bnn.py --classify path/to/image.jpg --out-dir image_classifier_runs
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from bayesian_utils import low_confidence_warning, mc_predict_classification, predictive_entropy
from train_utils import EarlyStopper, accuracy_gap_warning, plateau_scheduler, save_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "image_classifier_runs"
DEFAULT_MC_SAMPLES = 20


def build_transforms(image_size: int = 224, hflip: bool = True, grayscale: bool = False):
    train_ops = [transforms.Resize((image_size, image_size))]
    if hflip:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops += [
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    # 保留三通道架構；灰階時在 Normalize 前將亮度複製到三通道。
    if grayscale:
        train_ops.insert(-2, transforms.Grayscale(num_output_channels=3))
    train_tf = transforms.Compose(train_ops)
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if grayscale:
        eval_tf.transforms.insert(1, transforms.Grayscale(num_output_channels=3))
    return train_tf, eval_tf


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int, hflip: bool, grayscale: bool = False):
    train_tf, eval_tf = build_transforms(image_size, hflip, grayscale)
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError("train/ and val/ do not have the same set of class subfolders")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


class FeatureCNN(nn.Module):
    """自建卷積特徵提取器；所有權重從隨機初始化開始學習。"""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        blocks, channels = [], 3
        for width in (16, 32, 64):
            blocks.append(nn.Sequential(nn.Conv2d(channels, width, 3, padding=1),
                          nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)))
            channels = width
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(num_classes: int, freeze_backbone: bool = False, dropout: float = 0.3,
                architecture: str = "feature_cnn"):
    if architecture == "feature_cnn":
        # 舊呼叫即使傳 freeze_backbone=True，自建模型仍必須訓練特徵層。
        return FeatureCNN(num_classes, dropout)
    if architecture != "mobilenet_v2":
        raise ValueError(f"Unknown architecture: {architecture}")
    # 僅供舊 checkpoint 載入；不下載或載入外部預訓練權重。
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
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



def plot_history(out_dir: Path) -> Path:
    """Render recorded training/validation metrics without a GUI or retraining."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import PercentFormatter

    out_dir = Path(out_dir)
    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError("history.json contains no training epochs")
    fig = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    for ax, metric, title in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        for split in ("train", "val"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                    label="Training" if split == "train" else "Validation", marker=".")
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=160)
    return path


def train(data_dir: Path, out_dir: Path, epochs: int, batch_size: int, lr: float,
          patience: int, image_size: int, freeze_backbone: bool, dropout: float,
          weight_decay: float, hflip: bool = True, device: str | None = None, grayscale: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = build_dataloaders(data_dir, image_size, batch_size, hflip, grayscale)
    model = build_model(len(classes), freeze_backbone, dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    (out_dir / "config.json").write_text(json.dumps({"image_size": image_size, "dropout": dropout, "grayscale": grayscale, "architecture": "feature_cnn"}, indent=2))

    stopper = EarlyStopper(patience)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        warning = accuracy_gap_warning(train_acc, val_acc, epoch, epochs, underfit_hint="--unfreeze-backbone")

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    print(f"Training curves: {plot_history(out_dir)}")
    return model, classes, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    classes_path, config_path, model_path = out_dir / "classes.json", out_dir / "config.json", out_dir / "best_model.pt"
    if not (classes_path.exists() and config_path.exists() and model_path.exists()):
        return None

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model = build_model(len(classes), freeze_backbone=False, dropout=config["dropout"],
                        architecture=config.get("architecture", "mobilenet_v2"))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, classes, config["image_size"], config.get("grayscale", False))
    _loaded_models[key] = loaded
    return loaded


def classify_image(image_path: Path, out_dir: Path = DEFAULT_OUT_DIR,
                    mc_samples: int = DEFAULT_MC_SAMPLES) -> tuple[str, float, float]:
    """Bayesian (MC Dropout) classification of a single image.

    Returns (predicted_class, confidence, entropy):
      - confidence: mean softmax probability of the winning class across
        `mc_samples` stochastic forward passes (not a single deterministic
        pass) — see bayesian_utils.mc_predict_classification().
      - entropy: predictive entropy of the mean distribution; near 0 when
        the model is sure, ln(num_classes) when it's split evenly across
        every class (maximally unsure).

    Returns ("模型尚未訓練...", 0.0, 0.0) rather than crashing if no
    checkpoint has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return (
            "模型尚未訓練，請先準備好 train/val 資料夾並執行 "
            "`python image_classifier_bnn.py --data-dir <dir>` 進行訓練。",
            0.0, 0.0,
        )

    model, classes, image_size, grayscale = loaded
    _, eval_tf = build_transforms(image_size, grayscale=grayscale)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_tf(image).unsqueeze(0)

    mean_probs, _std_probs = mc_predict_classification(model, tensor, mc_samples)
    entropy = float(predictive_entropy(mean_probs).item())
    top_idx = int(mean_probs.argmax(dim=-1).item())
    confidence = float(mean_probs[0, top_idx].item())
    return classes[top_idx], confidence, entropy


def export_features(image_path: Path, out_dir: Path) -> Path:
    """輸出各卷積階段前四個通道的反應，不將特徵圖當作辨識正確證明。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到圖片：{image_path.resolve()}。請將範例圖片路徑換成實際存在的檔案。")
    loaded = _load(Path(out_dir))
    if loaded is None:
        raise FileNotFoundError("Train a checkpoint before exporting features")
    model, _, image_size, grayscale = loaded
    if not isinstance(model, FeatureCNN):
        raise ValueError("Feature maps require a feature_cnn checkpoint")
    _, transform = build_transforms(image_size, grayscale=grayscale)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    fig = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 4)
    with torch.no_grad():
        for stage, block in enumerate(model.features):
            tensor = block(tensor)
            for channel in range(4):
                axes[stage, channel].imshow(tensor[0, channel].cpu(), cmap="gray")
                axes[stage, channel].set_title(f"Stage {stage + 1} / Channel {channel + 1}")
                axes[stage, channel].axis("off")
    path = Path(out_dir) / "feature_maps.png"
    fig.savefig(path, dpi=140)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train or run a generic Bayesian (MC Dropout) image classifier")
    parser.add_argument("--features", type=Path, help="export learned feature maps for an image")
    parser.add_argument("--plot-history", action="store_true", help="plot existing history.json without training")
    parser.add_argument("--data-dir", type=Path,
                        help="folder with train/ and val/ subfolders (ImageFolder layout, output of data_split.py)")
    parser.add_argument("--grayscale", action="store_true", help="將訓練與驗證圖片轉成灰階，推論沿用 checkpoint 設定")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="also the MC Dropout probability used at --classify time")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=False)
    parser.add_argument("--no-hflip", dest="hflip", action="store_false",
                        help="disable random horizontal flip augmentation — turn off for direction-sensitive "
                             "classes (arrows, text, left/right-specific objects)")
    parser.set_defaults(hflip=True)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES,
                        help="number of MC Dropout forward passes for --classify's confidence/entropy")
    parser.add_argument("--classify", type=Path,
                        help="skip training; classify a single image against --out-dir's checkpoint")
    args = parser.parse_args()

    if args.features:
        try:
            print(export_features(args.features, args.out_dir))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.plot_history:
        print(plot_history(args.out_dir))
        return

    if args.classify:
        label, confidence, entropy = classify_image(args.classify, out_dir=args.out_dir, mc_samples=args.mc_samples)
        warning = low_confidence_warning(confidence)
        print(f"predicted: {label}  confidence={confidence:.3f}  entropy={entropy:.3f}{warning}")
        return

    if not args.data_dir:
        parser.error("--data-dir is required unless --classify is given")

    train(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.lr,
          args.patience, args.image_size, args.freeze_backbone, args.dropout,
          args.weight_decay, args.hflip, grayscale=args.grayscale)


if __name__ == "__main__":
    main()
