"""Character/persona model — turns a text description of a person into a
reusable "character card" (temperament trait scores + a persona embedding
vector), from scratch, no external AI API (Rule 06).

This is the first piece of a two-stage plan: (1) this file — text
description -> temperament profile, done here; (2) voice cloning — giving
that character an actual voice — is NOT built in this file. Voice/timbre
cloning needs real reference audio for speaker identity, which a text
description alone cannot provide; the character card below only carries a
`voice_ref` placeholder (null) for that future module to fill in with a path
to reference audio once it exists. At most, the trait scores here could
later inform *speaking style* (energetic vs. calm delivery, formality, ...)
once a TTS/voice-cloning component is built — that wiring is future work,
not implemented here.

Manifest format — a JSON list, trait scores in [0, 1] (0 = not present,
1 = strongly present; omitted tags default to 0). The set of trait tags is
NOT hard-coded — it is whatever tags appear in --train-manifest, same as how
chats.py builds its vocabulary from the training data instead of a fixed list:

    [{"name": "阿凱", "description": "個性活潑開朗，講話很直接，偶爾有點衝動。",
      "traits": {"活潑": 0.9, "直接": 0.8, "沉穩": 0.1}},
     {"name": "小雨", "description": "說話輕聲細語，做事謹慎，很少主動開口。",
      "traits": {"溫柔": 0.8, "沉穩": 0.7, "活潑": 0.1}}]

Architecture: character-level tokenizer + vocab built from the training
descriptions (character-level rather than whitespace-split so Chinese and
English both work, same reasoning as chats.py), fed into a bidirectional GRU;
the encoder outputs are mean-pooled over non-PAD positions into a fixed-size
persona embedding, then a linear head predicts a score per trait tag
(multi-label, sigmoid + BCE loss — a character can hold several traits at
once with different intensities, not just one class).

Design choices made to avoid overfitting AND underfitting on what will
likely be a small, hand-written set of character descriptions:

  Underfitting:
    - Bidirectional GRU gives the encoder both-direction context over what
      are usually short free-text descriptions, instead of only the
      left-to-right context a plain GRU would have.
    - ReduceLROnPlateau backs off the learning rate only once validation
      loss stalls, instead of a fixed schedule that might decay too early.

  Overfitting:
    - Dropout before the trait classifier head + weight decay (AdamW).
    - Early stopping on validation loss (patience-based), restoring the
      best checkpoint rather than the last one.
    - Per-epoch validation MAE (mean absolute error between predicted and
      target trait scores, in the same 0-1 scale as the labels) is printed
      alongside the loss gap so a human can see real profile accuracy, not
      just the loss number, and a warning fires on a wide train/val gap.

Usage:
    python character_model.py --train-manifest train.json --val-manifest val.json --epochs 50
    python character_model.py --build --name "阿凱" --description "個性活潑開朗，講話很直接。"
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint

PAD, UNK = 0, 1
SPECIAL_TOKENS = {"<pad>": PAD, "<unk>": UNK}

DEFAULT_CHARACTERS_DIR = Path(__file__).resolve().parent / "characters"
DEFAULT_OUT_DIR = DEFAULT_CHARACTERS_DIR / "character_runs"
MAX_LEN = 80  # character descriptions tend to run longer than a short chat reply


def tokenize(text: str) -> list[str]:
    return list(text.strip())


def build_vocab(records: list[dict]) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    for record in records:
        for token in tokenize(record["description"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_trait_vocab(records: list[dict]) -> list[str]:
    tags = {tag for record in records for tag in record.get("traits", {})}
    return sorted(tags)


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids


def encode_traits(traits: dict, trait_vocab: list[str]) -> list[float]:
    # tags not seen during training are dropped rather than erroring — same
    # "unseen label at val/inference time is silently ignored" choice OCR.py
    # makes for characters missing from its training charset.
    return [max(0.0, min(1.0, traits.get(tag, 0.0))) for tag in trait_vocab]


class CharacterDataset(Dataset):
    def __init__(self, records: list[dict], vocab: dict, trait_vocab: list[str], max_len: int):
        self.records = records
        self.vocab = vocab
        self.trait_vocab = trait_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        input_ids = torch.tensor(encode(record["description"], self.vocab, self.max_len))
        target = torch.tensor(encode_traits(record.get("traits", {}), self.trait_vocab), dtype=torch.float32)
        return input_ids, target, record.get("name", "")


class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_traits: int, embed_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_traits)

    def forward(self, input_ids):
        mask = (input_ids != PAD).unsqueeze(-1)  # (batch, max_len, 1)
        outputs, _ = self.gru(self.embedding(input_ids))  # (batch, max_len, hidden*2)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean over real tokens only
        pooled = self.dropout(pooled)
        return pooled, self.classifier(pooled)  # persona embedding, trait logits


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_abs_err, total_samples, total_targets = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, targets, _ in loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            if train:
                optimizer.zero_grad()
            _, logits = model(input_ids)
            loss = criterion(logits, targets)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_abs_err += (torch.sigmoid(logits) - targets).abs().sum().item()
            total_samples += input_ids.size(0)
            total_targets += targets.numel()
    return total_loss / total_samples, total_abs_err / max(total_targets, 1)


def train(train_manifest: Path, val_manifest: Path, out_dir: Path, epochs: int, batch_size: int,
          embed_size: int, hidden_size: int, lr: float, max_len: int, dropout: float,
          weight_decay: float, patience: int, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_records = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_records = json.loads(Path(val_manifest).read_text(encoding="utf-8"))

    vocab = build_vocab(train_records)
    trait_vocab = build_trait_vocab(train_records)
    if not trait_vocab:
        raise ValueError("--train-manifest has no \"traits\" tags at all — nothing to learn to predict")

    train_loader = DataLoader(
        CharacterDataset(train_records, vocab, trait_vocab, max_len), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        CharacterDataset(val_records, vocab, trait_vocab, max_len), batch_size=batch_size, shuffle=False
    )

    model = CharacterEncoder(len(vocab), len(trait_vocab), embed_size, hidden_size, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "traits.json").write_text(json.dumps(trait_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"embed_size": embed_size, "hidden_size": hidden_size, "max_len": max_len,
                     "dropout": dropout}, indent=2)
    )

    stopper = EarlyStopper(patience)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_mae={train_mae:.3f}  "
              f"val_loss={val_loss:.4f} val_mae={val_mae:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
                         "val_loss": val_loss, "val_mae": val_mae})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, vocab, trait_vocab, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    vocab_path = out_dir / "vocab.json"
    traits_path = out_dir / "traits.json"
    config_path = out_dir / "config.json"
    model_path = out_dir / "best_model.pt"
    if not (vocab_path.exists() and traits_path.exists() and config_path.exists() and model_path.exists()):
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    trait_vocab = json.loads(traits_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model = CharacterEncoder(len(vocab), len(trait_vocab), config["embed_size"],
                              config["hidden_size"], config["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, vocab, trait_vocab, config)
    _loaded_models[key] = loaded
    return loaded


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-一-鿿]", "_", name).strip("_")
    return cleaned or "character"


def build_character(name: str, description: str, out_dir: Path = DEFAULT_OUT_DIR,
                     characters_dir: Path | None = None) -> dict:
    """Run the self-trained encoder on one character description, returning
    (and saving to characters_dir) a "character card": trait scores + the
    raw persona embedding, both reusable later (e.g. to condition chats.py's
    replies on this persona, or as a style hint for a future voice-cloning
    module — neither is wired up yet).

    Returns a card with a non-null "status" message (rather than crashing)
    if no checkpoint has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return {
            "name": name, "description": description, "traits": {}, "embedding": None, "voice_ref": None,
            "status": "模型尚未訓練，請先提供人物描述資料集並執行 "
                      "`python character_model.py --train-manifest ... --val-manifest ...` 進行訓練。",
        }

    model, vocab, trait_vocab, config = loaded
    input_ids = torch.tensor([encode(description, vocab, config["max_len"])])
    with torch.no_grad():
        embedding, logits = model(input_ids)
        scores = torch.sigmoid(logits).squeeze(0).tolist()

    card = {
        "name": name,
        "description": description,
        "traits": {tag: round(score, 4) for tag, score in zip(trait_vocab, scores)},
        "embedding": [round(v, 6) for v in embedding.squeeze(0).tolist()],
        "voice_ref": None,  # future voice-cloning module fills this in with a reference-audio path
        "status": None,
    }

    characters_dir = Path(characters_dir) if characters_dir else DEFAULT_CHARACTERS_DIR
    characters_dir.mkdir(parents=True, exist_ok=True)
    (characters_dir / f"{_safe_filename(name)}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return card


def main():
    parser = argparse.ArgumentParser(description="Train or run a from-scratch character/persona model")
    parser.add_argument("--train-manifest", type=Path, help='JSON file of [{"description": ..., "traits": ...}, ...]')
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--characters-dir", type=Path, default=DEFAULT_CHARACTERS_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--build", action="store_true",
                         help="skip training; build one character card from --name/--description using --out-dir's checkpoint")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    args = parser.parse_args()

    if args.build:
        if not (args.name and args.description):
            parser.error("--name and --description are required with --build")
        card = build_character(args.name, args.description, out_dir=args.out_dir, characters_dir=args.characters_dir)
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return

    if not (args.train_manifest and args.val_manifest):
        parser.error("--train-manifest and --val-manifest are required unless --build is given")

    train(args.train_manifest, args.val_manifest, args.out_dir, args.epochs, args.batch_size,
          args.embed_size, args.hidden_size, args.lr, args.max_len, args.dropout,
          args.weight_decay, args.patience)


if __name__ == "__main__":
    main()
