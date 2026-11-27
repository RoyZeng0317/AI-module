"""本地多輪SFT；獨立train/val、共用對話格式、動態padding與新輸出目錄。"""
import argparse, hashlib, json, random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from bpe_tokenizer import BPETokenizer, EOS, PAD
from context_format import FORMAT_VERSION, RESPONSE_BUDGET, build_prompt

ROOT = Path(__file__).resolve().parents[1]


class ContextDataset(Dataset):
    def __init__(self, records, tokenizer, block_size, response_budget=RESPONSE_BUDGET, skip_long=False):
        if not isinstance(response_budget, int) or isinstance(response_budget, bool) or not 0 < response_budget < block_size:
            raise ValueError("response_budget必須大於0且小於block_size。")
        self.examples, self.skipped = [], 0
        for record in records:
            if not isinstance(record.get("reply"), str) or not record["reply"].strip():
                raise ValueError("每筆多輪資料必須有非空回答。")
            reply = tokenizer.encode(record["reply"]) + [EOS]
            try:
                if len(reply) > response_budget:
                    raise ValueError("回答超過response_budget。")
                prefix = build_prompt(tokenizer, record["prompt"], record.get("history"), block_size-response_budget)
            except ValueError as exc:
                if not skip_long or ("預算" not in str(exc) and "response_budget" not in str(exc)):
                    raise
                self.skipped += 1
                continue
            seq = prefix + reply
            x, y = torch.tensor(seq[:-1]), torch.tensor(seq[1:])
            y[:len(prefix)-1] = PAD  # SEP位置預測第一個回答token；前文及當輪問題不計loss
            if not torch.any(y != PAD):
                raise ValueError("多輪資料沒有可訓練的回答token。")
            self.examples.append((x, y))
        if not self.examples:
            raise ValueError("沒有有效的多輪訓練資料。")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_context(batch):
    return tuple(pad_sequence(items, batch_first=True, padding_value=PAD) for items in zip(*batch))


def _read(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("train與val必須是各自非空的JSON陣列。")
    return records


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finetune_context(train_path, val_path, source_dir, out_dir, epochs=5, batch_size=8, lr=1e-4,
                     freeze_layers=4, freeze_embeddings=True, seed=42, response_budget=RESPONSE_BUDGET,
                     replay_path=None, dropout=.1, weight_decay=.01, label_smoothing=.05,
                     patience=5, device=None, gpu_mem_fraction=.85, checkpoint_every=1):
    from transformer_chat import GPT, _cap_gpu_memory, _freeze_for_finetune, _resolve_device, _run_epochs
    out_dir, source_dir = Path(out_dir).resolve(), Path(source_dir).resolve()
    if not out_dir.is_relative_to(ROOT) or out_dir == ROOT or out_dir.exists():
        raise ValueError("多輪模型必須輸出到專案內尚不存在的新目錄，不可覆寫正式權重。")
    if min(epochs, batch_size, lr) <= 0:
        raise ValueError("epochs、batch_size、lr必須大於零。")
    train_records, val_records = _read(train_path), _read(val_path)
    train_ids = {r["dialogue_id"] for r in train_records}
    if train_ids & {r["dialogue_id"] for r in val_records}:
        raise ValueError("同一段對話不可跨train與val。")
    fingerprints = lambda records: {json.dumps([r.get("history", []), r["prompt"], r["reply"]], ensure_ascii=False) for r in records}
    if fingerprints(train_records) & fingerprints(val_records):
        raise ValueError("train與val有重複的完整樣本。")
    random.seed(seed); torch.manual_seed(seed)
    device = _resolve_device(device); _cap_gpu_memory(device, gpu_mem_fraction)
    tokenizer = BPETokenizer.load(source_dir)
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    train_data = ContextDataset(train_records, tokenizer, config["block_size"], response_budget)
    val_data = ContextDataset(val_records, tokenizer, config["block_size"], response_budget)
    replay_skipped = 0
    if replay_path is not None:
        replay_records = _read(replay_path)
        if fingerprints(replay_records) & fingerprints(val_records):
            raise ValueError("replay與val有重複的完整樣本。")
        replay = ContextDataset(replay_records, tokenizer, config["block_size"], response_budget, skip_long=True)
        replay_skipped = replay.skipped
        train_data.examples.extend(replay.examples)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_context)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_context)
    model = GPT(tokenizer.vocab_size, config["block_size"], config["n_layer"], config["n_embd"], config["n_head"], dropout).to(device)
    model.load_state_dict(torch.load(source_dir / "model.pt", map_location=device))
    model.label_smoothing = label_smoothing
    _freeze_for_finetune(model, freeze_embeddings, freeze_layers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    config.update(context_format=FORMAT_VERSION, response_budget=response_budget, dropout=dropout,
                  context_training={"source_dir": str(source_dir), "source_model_sha256": _hash(source_dir / "model.pt"),
                                    "tokenizer_sha256": _hash(source_dir / "bpe_vocab.json"), "merges_sha256": _hash(source_dir / "bpe_merges.json"),
                                    "train_sha256": _hash(train_path), "val_sha256": _hash(val_path),
                                    "replay_sha256": _hash(replay_path) if replay_path else None,
                                    "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
                                    "freeze_layers": freeze_layers, "freeze_embeddings": freeze_embeddings,
                                    "weight_decay": weight_decay, "label_smoothing": label_smoothing,
                                    "train_examples": len(train_data), "val_examples": len(val_data), "replay_skipped": replay_skipped})
    out_dir.mkdir(parents=True)
    def save(folder, history):
        folder.mkdir(parents=True, exist_ok=True)
        tokenizer.save(folder)
        torch.save(model.state_dict(), folder / "model.pt")
        (folder / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    history = _run_epochs(model, train_loader, val_loader, optimizer, scheduler, epochs, patience, device,
                          tag="context", baseline_loss=__import__("math").log(tokenizer.vocab_size),
                          checkpoint_every=checkpoint_every,
                          save_checkpoint=lambda current, records: save(out_dir / "checkpoint", records))
    config["context_training"]["best_epoch"] = min(history, key=lambda row: row["val_loss"])["epoch"]
    save(out_dir, history)
    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "val", "source-dir", "out-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-layers", type=int, default=4)
    args = parser.parse_args()
    finetune_context(args.train, args.val, args.source_dir, args.out_dir, args.epochs, args.batch_size,
                     args.lr, freeze_layers=args.freeze_layers, replay_path=args.replay)


if __name__ == "__main__":
    main()
