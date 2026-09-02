"""tranning/embedding_model.py -- self-trained sentence embedding model (bidirectional Transformer encoder + unsupervised SimCSE contrastive loss), used by lib/RAG.py for semantic retrieval."""

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bpe_tokenizer import BPETokenizer, PAD

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "embed_runs"
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_GPU_MEM_FRACTION = 0.85  # same RTX 4060 8GB headroom cap as transformer_chat.py


def _resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _cap_gpu_memory(device: str, fraction: float | None) -> None:
    if device == "cuda" and fraction is not None:
        torch.cuda.set_per_process_memory_fraction(fraction)


# --- model ---------------------------------------------------------------

class BiSelfAttention(nn.Module):
    """Multi-head self-attention with only a key-padding mask -- no causal
    mask, since a sentence encoder needs every position to see the whole
    sequence, not just what came before it."""

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = BiSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SentenceEncoder(nn.Module):
    """Bidirectional Transformer encoder -> masked mean-pool over non-pad
    positions -> L2-normalized sentence vector."""

    def __init__(self, vocab_size: int, max_len: int, n_layer: int, n_embd: int,
                 n_head: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        pad_mask = idx == PAD
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)

        valid = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)


# --- unsupervised SimCSE training data ------------------------------------

class SentenceDataset(Dataset):
    def __init__(self, sentences: list[str], tokenizer: BPETokenizer, max_len: int):
        self.ids = [tokenizer.encode(s, max_len=max_len) for s in sentences]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.ids[idx], dtype=torch.long)


_SENT_SPLIT_RE = re.compile(r"[^\n。！？!?.]+[。！？!?.]?")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.findall(text) if s.strip()]


def _collect_default_sentences(max_sentences: int) -> list[str]:
    """Pulls training sentences from data/pairs.json + data/corpus_*.txt --
    no separate labeled dataset needed, since SimCSE's contrastive loss is
    unsupervised (see module docstring)."""
    sentences: list[str] = []

    pairs_path = _DATA_DIR / "pairs.json"
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
        for pair in pairs:
            sentences.append(pair["prompt"])
            sentences.append(pair["reply"])

    for corpus_name in ("corpus_zh_starter.txt", "corpus_code_fullstack.txt", "corpus_code_html_css.txt"):
        corpus_path = _DATA_DIR / corpus_name
        if not corpus_path.exists():
            continue
        text = corpus_path.read_text(encoding="utf-8", errors="ignore")
        sentences.extend(_split_sentences(text))
        if len(sentences) >= max_sentences:
            break

    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen and 2 <= len(s) <= 300:
            seen.add(s)
            unique.append(s)

    random.shuffle(unique)
    return unique[:max_sentences]


def _simcse_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """z1/z2 are the same sentences forwarded twice through the model with
    independent dropout masks (the SimCSE trick) -- each sentence's z2 is
    its own positive, every other sentence in the batch is a negative."""
    sim = z1 @ z2.t() / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def train(sentences: list[str] | None = None, out_dir: Path = DEFAULT_OUT_DIR, epochs: int = 10,
          batch_size: int = 64, max_len: int = 64, n_layer: int = 4, n_embd: int = 256,
          n_head: int = 4, dropout: float = 0.1, lr: float = 3e-4, weight_decay: float = 0.01,
          vocab_size: int = 8000, temperature: float = 0.05, max_sentences: int = 20000,
          device: str | None = None, gpu_mem_fraction: float | None = DEFAULT_GPU_MEM_FRACTION):
    device = _resolve_device(device)
    _cap_gpu_memory(device, gpu_mem_fraction)

    if sentences is None:
        sentences = _collect_default_sentences(max_sentences)
    if len(sentences) < 2:
        raise ValueError("need at least 2 sentences to train a contrastive embedding model")

    tokenizer = BPETokenizer.train(sentences, vocab_size=vocab_size)
    dataset = SentenceDataset(sentences, tokenizer, max_len)
    batch_size = min(batch_size, len(sentences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SentenceEncoder(tokenizer.vocab_size, max_len, n_layer, n_embd, n_head, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            z1 = model(x)
            z2 = model(x)
            loss = _simcse_loss(z1, z2, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        epoch_loss = loss_sum / max(batches, 1)
        scheduler.step(epoch_loss)
        print(f"[embed] epoch {epoch:3d}  loss={epoch_loss:.4f}", flush=True)
        history.append({"epoch": epoch, "loss": epoch_loss})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # tokenizer/config/weights written together after training finishes, same
    # ordering fix as transformer_chat.py -- avoids a half-written out_dir
    # (new vocab, stale weights) being visible to a concurrent reader.
    tokenizer.save(out_dir)
    config = {"max_len": max_len, "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "dropout": dropout}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, tokenizer, history


# --- inference -------------------------------------------------------------

_loaded_encoder: dict = {}


def load_encoder(out_dir: Path = DEFAULT_OUT_DIR, device: str | None = None):
    """Returns (SentenceEncoder, BPETokenizer, max_len, device), cached per
    (out_dir, device) so repeated calls don't reload the checkpoint. Returns
    None if no checkpoint has been trained yet at out_dir."""
    device = _resolve_device(device)
    key = (str(out_dir), device)
    if key in _loaded_encoder:
        return _loaded_encoder[key]

    out_dir = Path(out_dir)
    model_path, config_path = out_dir / "model.pt", out_dir / "config.json"
    if not (model_path.exists() and config_path.exists() and (out_dir / "bpe_vocab.json").exists()):
        return None

    tokenizer = BPETokenizer.load(out_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = SentenceEncoder(tokenizer.vocab_size, config["max_len"], config["n_layer"],
                             config["n_embd"], config["n_head"], config.get("dropout", 0.0))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    loaded = (model, tokenizer, config["max_len"], device)
    _loaded_encoder[key] = loaded
    return loaded


def embed_texts(texts: list[str], encoder: SentenceEncoder, tokenizer: BPETokenizer,
                 batch_size: int = 32) -> list[list[float]]:
    device = next(encoder.parameters()).device
    encoder.eval()
    vectors: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ids = torch.tensor([tokenizer.encode(t, max_len=encoder.max_len) for t in batch],
                                dtype=torch.long, device=device)
            vectors.extend(encoder(ids).cpu().tolist())
    return vectors


def encode_text(text: str, out_dir: Path = DEFAULT_OUT_DIR) -> list[float]:
    loaded = load_encoder(out_dir)
    if loaded is None:
        return []
    encoder, tokenizer, _max_len, _device = loaded
    return embed_texts([text], encoder, tokenizer)[0]


# --- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-trained sentence embedding model (unsupervised SimCSE-style contrastive learning)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train on data/pairs.json + data/corpus_*.txt sentences")
    p_train.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--max-len", type=int, default=64)
    p_train.add_argument("--n-layer", type=int, default=4)
    p_train.add_argument("--n-embd", type=int, default=256)
    p_train.add_argument("--n-head", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=0.01)
    p_train.add_argument("--vocab-size", type=int, default=8000)
    p_train.add_argument("--temperature", type=float, default=0.05)
    p_train.add_argument("--max-sentences", type=int, default=20000)
    p_train.add_argument("--gpu-mem-fraction", type=float, default=DEFAULT_GPU_MEM_FRACTION,
                          help="cap this process to this fraction of total VRAM on CUDA devices; "
                               "pass 0/negative to disable the cap")

    p_enc = sub.add_parser("encode", help="quick manual check: print a trained checkpoint's vector for one sentence")
    p_enc.add_argument("text")
    p_enc.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    args = parser.parse_args()
    if args.command == "train":
        gpu_mem_fraction = args.gpu_mem_fraction if args.gpu_mem_fraction > 0 else None
        train(out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
              max_len=args.max_len, n_layer=args.n_layer, n_embd=args.n_embd,
              n_head=args.n_head, dropout=args.dropout, lr=args.lr,
              weight_decay=args.weight_decay, vocab_size=args.vocab_size,
              temperature=args.temperature, max_sentences=args.max_sentences,
              gpu_mem_fraction=gpu_mem_fraction)
    elif args.command == "encode":
        vector = encode_text(args.text, out_dir=args.out_dir)
        if not vector:
            print(f"embedding 模型尚未訓練，請先執行 python embedding_model.py train"
                  f"（checkpoint 應位於 {args.out_dir}）")
            return
        print(f"vector dim={len(vector)}  first 8 values={[round(v, 4) for v in vector[:8]]}")


if __name__ == "__main__":
    main()
