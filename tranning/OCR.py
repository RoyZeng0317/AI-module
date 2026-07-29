"""Text recognition (OCR) — from-scratch CRNN + CTC scaffold (no Tesseract,
no cloud OCR API — Rule 06).

Point this at a manifest of (image, transcript) pairs once a real dataset is
available. Right now, with no data provided yet, this only defines the
architecture and training loop; it is smoke-tested against a handful of
synthetic character-image pairs in test_OCR.py (no recognition-accuracy
claim — just "the pipeline runs").

Manifest format — a JSON list, image paths relative to --images-root
(defaults to the manifest's own folder):

    [{"image": "0001.png", "text": "台北市中正區"}, {"image": "0002.png", "text": "STOP"}]

Architecture: CNN feature extractor (VGG-style, 7 conv layers) that reduces
each input image to a single row of feature "columns" left-to-right, fed
into a 2-layer bidirectional LSTM, then a linear layer producing
per-column character logits. Trained with CTC loss (Connectionist Temporal
Classification), which is the reason the manifest only needs a full
transcript per image and not per-character bounding boxes: CTC marginalizes
over every possible column-to-character alignment itself. Character-level
vocabulary built from the training manifest (same approach as chats.py) so
Chinese and English/digit signage can share one model.

Design choices made to avoid overfitting AND underfitting on what will
likely be a small, mixed-script text-recognition dataset (same reasoning as
road_sign_train.py / circuit_diagram_train.py, adapted to sequence output):

  Underfitting:
    - 2-layer bidirectional LSTM on top of the CNN gives the model left/right
      context for each column instead of classifying columns independently.
    - ReduceLROnPlateau backs off the learning rate only once val loss
      stalls, instead of a fixed decay schedule that might cut it too early.

  Overfitting:
    - Augmentation deliberately excludes horizontal/vertical flips: text is
      directional (mirrored characters/digits are unreadable or become a
      different glyph), same reasoning road_sign_train.py uses for signs.
      Only small rotation, minor translation, and brightness/contrast jitter
      are applied.
    - Dropout between LSTM layers + weight decay (AdamW).
    - Early stopping on validation loss (patience-based), restoring the
      best checkpoint rather than the last one.
    - Per-epoch validation CER (character error rate, via edit distance) is
      printed alongside the loss gap so a human can see real recognition
      quality, not just the loss number, and a warning fires on a wide
      train/val loss gap.

Usage:
    python OCR.py --train-manifest path/to/train.json --val-manifest path/to/val.json --epochs 50
    python OCR.py --recognize path/to/some_image.png   # run the last checkpoint on one image
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint

BLANK = 0  # reserved CTC blank id; real characters start at index 1
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "ocr_runs"


def build_charset(records: list[dict]) -> dict[str, int]:
    chars = sorted({c for record in records for c in record["text"]})
    return {c: i + 1 for i, c in enumerate(chars)}


def load_image(path: Path, img_height: int, img_width: int) -> Image.Image:
    """Resize preserving aspect ratio to img_height, then right-pad (white
    background) up to img_width — avoids the text-distorting stretch a
    plain resize-to-fixed-size would cause, at the cost of cropping text
    images wider than img_width (raise --img-width for a wider dataset).
    """
    image = Image.open(path).convert("L")
    w, h = image.size
    new_w = max(1, min(img_width, round(w * img_height / h)))
    image = image.resize((new_w, img_height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (img_width, img_height), color=255)
    canvas.paste(image, (0, 0))
    return canvas


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(2),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


class OCRDataset(Dataset):
    def __init__(self, records: list[dict], images_root: Path, charset: dict[str, int],
                 img_height: int, img_width: int, train: bool):
        self.records = records
        self.images_root = Path(images_root)
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width
        self.transform = build_transforms(train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = load_image(self.images_root / record["image"], self.img_height, self.img_width)
        tensor = self.transform(image)
        text = record["text"]
        # characters not seen in the training charset (e.g. a val-only glyph)
        # are dropped rather than mapped to an UNK id — CTC has no natural
        # "unknown glyph" symbol to predict, unlike chats.py's token UNK.
        label = torch.tensor([self.charset[c] for c in text if c in self.charset], dtype=torch.long)
        return tensor, label, text


def collate(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, target_lengths, list(texts)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 32, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        if img_height % 16 != 0:
            raise ValueError("img_height must be a multiple of 16 (the CNN halves height 4 times)")
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (img_height // 16, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)              # (N, 512, 1, T)
        conv = conv.squeeze(2)          # (N, 512, T)
        conv = conv.permute(0, 2, 1)    # (N, T, 512)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)       # (N, T, num_classes)
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)  # (T, N, num_classes) — nn.CTCLoss layout


def greedy_decode(log_probs: torch.Tensor, idx_to_char: dict) -> list[str]:
    """Collapse repeated symbols and drop blanks (standard CTC greedy
    decoding) — good enough to sanity-check training; swap for beam search
    with a language model later if accuracy needs it.
    """
    preds = log_probs.argmax(2).transpose(0, 1)  # (N, T)
    results = []
    for seq in preds.tolist():
        chars, prev = [], None
        for idx in seq:
            if idx != BLANK and idx != prev:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def char_error_rate(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, target) / len(target)


def train(train_manifest: Path, val_manifest: Path, images_root: Path, out_dir: Path,
          epochs: int, batch_size: int, lr: float, patience: int, img_height: int,
          img_width: int, hidden_size: int, dropout: float, weight_decay: float,
          device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_records = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_records = json.loads(Path(val_manifest).read_text(encoding="utf-8"))
    images_root = Path(images_root) if images_root else Path(train_manifest).resolve().parent

    charset = build_charset(train_records)
    idx_to_char = {i: c for c, i in charset.items()}

    train_loader = DataLoader(
        OCRDataset(train_records, images_root, charset, img_height, img_width, train=True),
        batch_size=batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        OCRDataset(val_records, images_root, charset, img_height, img_width, train=False),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    model = CRNN(len(charset) + 1, img_height, hidden_size, dropout).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charset.json").write_text(json.dumps(charset, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"img_height": img_height, "img_width": img_width, "hidden_size": hidden_size}, indent=2)
    )

    stopper = EarlyStopper(patience)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        for images, targets, target_lengths, _ in train_loader:
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_samples += images.size(0)
        train_loss = train_loss_sum / train_samples

        model.eval()
        val_loss_sum, val_samples, cer_sum = 0.0, 0, 0.0
        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
                log_probs = model(images)
                input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss_sum += loss.item() * images.size(0)
                val_samples += images.size(0)
                preds = greedy_decode(log_probs.cpu(), idx_to_char)
                cer_sum += sum(char_error_rate(pred, target) for pred, target in zip(preds, texts))
        val_loss = val_loss_sum / val_samples
        val_cer = cer_sum / val_samples
        scheduler.step(val_loss)

        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_cer={val_cer:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_cer": val_cer})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, charset, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    charset_path, config_path, model_path = out_dir / "charset.json", out_dir / "config.json", out_dir / "best_model.pt"
    if not (charset_path.exists() and config_path.exists() and model_path.exists()):
        return None

    charset = json.loads(charset_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    idx_to_char = {i: c for c, i in charset.items()}

    model = CRNN(len(charset) + 1, config["img_height"], config["hidden_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, config, idx_to_char)
    _loaded_models[key] = loaded
    return loaded


def recognize(image_path: Path, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Run the self-trained CRNN+CTC model on one image.

    Returns a placeholder message (rather than crashing) if no checkpoint
    has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return "模型尚未訓練，請先提供文字辨識資料集並執行 `python OCR.py --train-manifest ... --val-manifest ...` 進行訓練。"

    model, config, idx_to_char = loaded
    image = load_image(Path(image_path), config["img_height"], config["img_width"])
    tensor = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image)).unsqueeze(0)

    with torch.no_grad():
        log_probs = model(tensor)
    return greedy_decode(log_probs, idx_to_char)[0]


def main():
    parser = argparse.ArgumentParser(description="Train or run a from-scratch CRNN+CTC OCR model")
    parser.add_argument("--train-manifest", type=Path, help='JSON file of [{"image": ..., "text": ...}, ...]')
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--images-root", type=Path, default=None,
                        help="base dir image paths are relative to (default: --train-manifest's own folder)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--recognize", type=Path, default=None,
                        help="skip training; run OCR on this image using --out-dir's checkpoint")
    args = parser.parse_args()

    if args.recognize:
        print(recognize(args.recognize, out_dir=args.out_dir))
        return

    if not (args.train_manifest and args.val_manifest):
        parser.error("--train-manifest and --val-manifest are required unless --recognize is given")

    train(args.train_manifest, args.val_manifest, args.images_root, args.out_dir,
          args.epochs, args.batch_size, args.lr, args.patience, args.img_height,
          args.img_width, args.hidden_size, args.dropout, args.weight_decay)


if __name__ == "__main__":
    main()
