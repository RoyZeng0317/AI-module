"""transformer_chat.py — decoder-only Transformer (GPT-style) chat model.

Why this file exists next to chats.py's GRU model: chats.py's checkpoint
converged to train loss 0.0001 on 28 hand-written pairs (see CLAUDE.md
修正日誌) — that is essentially perfect memorization of those 28 exact
sentences, not language understanding. Feed it anything even slightly off
from a memorized prompt and it has no real grammar/meaning model to fall
back on, so it stitches together fragments of whatever it *does* remember
and produces garbled, truncated-looking output (e.g. "自建的聊天模型,目前
還在學習中" coming out as "自續學習中") — this is the direct cause of the
"斷斷續續" behaviour reported against chats.py. That is not a bug fixable
by tuning chats.py's hyperparameters; it is the ceiling of "tiny
character-level GRU + 28 examples of straight memorization".

This file is the honest next step toward "more LLM-like": every modern LLM
(GPT/Claude/etc.) is a decoder-only Transformer that learns language in two
stages, and that is what's implemented here —

  1. pretrain() — unsupervised next-token prediction over a plain text
     corpus (no prompt/reply structure at all, just "predict the next
     token"). This is the stage that actually teaches the model grammar,
     common word/phrase shapes, and some world knowledge if the corpus
     contains it. chats.py's GRU never had this stage.
  2. finetune() — supervised fine-tuning on data/pairs.json's existing
     {"prompt", "reply"} format, continuing from the pretrained weights,
     with the loss masked over the prompt tokens (see ChatSFTDataset) so
     the model is only scored on producing the reply, not on reproducing
     the user's own words back.

Architecture: token embedding (tied to the output projection, standard GPT
practice) + learned positional embedding + N pre-norm Transformer blocks
(causal multi-head self-attention + GELU feed-forward, both with residual
connections) + final LayerNorm. No pretrained weights are loaded from
anywhere — trained from scratch on whatever corpus/pairs you point it at,
consistent with Rule 06 (self-built only).

Honest scope (told to the user before writing a line of this): on a single
RTX 4060 8GB, from scratch, this project can realistically train a Transformer
in the tens-of-millions-of-parameters range (the defaults below land around
that). That is nowhere near GPT/Claude-class model scale (hundreds of
billions of parameters, trillions of training tokens) — real step-by-step
logical reasoning is not something this will have. What IS a realistic goal:
once there is real pretraining-corpus text to point --corpus at (not yet
provided — see CLAUDE.md to-do), replies should stop being pure
memorization and start generalizing to phrasings the model never saw
verbatim, the way word/character-level statistics generalize in any
n-gram/neural language model, just far better with self-attention over the
full context than chats.py's GRU could ever manage.

Anti-overfitting/underfitting design mirrors the rest of this project's
training scripts (road_sign_train.py / OCR.py / speech_to_text.py):
AdamW + weight decay, ReduceLROnPlateau, early stopping on validation loss
with a restored best checkpoint, and a printed warning when the train/val
gap suggests overfitting or when train loss stays high too long (see
_run_epochs()). Augmentation-by-flipping doesn't apply to text the way it
does to images/signs, so there is no equivalent knob here.

Deliberately NOT wired into tools.py/smart_reply()/home_screen.py yet:
chats.py's GRU checkpoint keeps serving the live chat UI until a Transformer
checkpoint trained on a real corpus is actually validated to reply better
than the (currently working, if narrow) GRU baseline. Swapping the live
model over is a separate, deliberate step once that's true.

Usage:
    python transformer_chat.py pretrain --corpus path/to/corpus.txt --epochs 20
    python transformer_chat.py finetune --data ../data/pairs.json --epochs 50
    python transformer_chat.py chat                      # REPL against the fine-tuned checkpoint
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bpe_tokenizer import BPETokenizer, EOS, PAD, SEP

DEFAULT_PRETRAIN_DIR = Path(__file__).resolve().parent / "gpt_pretrain_runs"
DEFAULT_FINETUNE_DIR = Path(__file__).resolve().parent / "gpt_chat_runs"


# --- model -------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (position t can only
    attend to positions <= t) plus a key-padding mask (never attend to
    <pad> positions) — the mechanism that lets every output position see
    the *whole* preceding context directly, instead of chats.py's GRU
    squeezing everything through one recurrent hidden state.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_SCALE_INIT = True  # see GPT._init_weights
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_head, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_dim)
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
        self.net[2].RESIDUAL_SCALE_INIT = True  # see GPT._init_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block (LayerNorm before, not after, each
    sub-layer) — the standard GPT-2-style arrangement, more stable to train
    than the original post-norm Transformer, which matters here since this
    is trained from scratch with no pretrained initialization to lean on.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_embd: int,
                 n_head: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=PAD)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying (standard GPT practice)

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-branch output projections
        # (CausalSelfAttention.proj, the second Linear in each MLP): every
        # block ADDS its attention/MLP output onto the residual stream, so
        # with ordinary init the stream's variance compounds across
        # n_layer blocks and the untrained model's logits come out wildly
        # overscaled. Confirmed empirically before this fix existed: a
        # first real pretrain run (8000-token vocab, where an untrained
        # model should start near the random-guess loss ln(8000)=9.0)
        # started at loss 94.9 and was still at 14.5 after 15 epochs --
        # never even reaching the random-guess baseline. Scaling these
        # specific weights by 1/sqrt(2*n_layer) keeps residual-stream
        # variance roughly constant regardless of depth (Radford et al.,
        # "Language Models are Unsupervised Multitask Learners", 2019).
        for module in self.modules():
            if getattr(module, "RESIDUAL_SCALE_INIT", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        key_padding_mask = idx == PAD
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # single flattened cross_entropy over the whole (B*T,) target
            # tensor, NOT a per-timestep loop — chats.py hit a 0/0 -> NaN
            # loss (see CLAUDE.md Error log #4) from computing loss
            # per-timestep, where a timestep that happened to be all-<pad>
            # across the batch divided by zero valid tokens. Flattening
            # first means the mean divides by the total valid-token count
            # across the whole batch, which is only zero if an entire batch
            # were nothing but padding — never true here.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=PAD)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.9) -> torch.Tensor:
        """Autoregressive sampling: temperature + top-k + top-p (nucleus),
        not chat_reply()'s pure greedy argmax. Greedy decoding from a
        language model collapses into repetition loops far more readily
        than a seq2seq memorization model does; sampling from a restricted
        candidate set is what lets replies vary and read more naturally,
        which is the whole point of moving off the GRU model.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == EOS:
                break
        return idx


# --- pretraining data: plain text -> next-token chunks -----------------

class LMChunkDataset(Dataset):
    """Chops one long token-id stream into non-overlapping
    (block_size -> block_size, shifted by one) windows for next-token
    prediction. The last chunk is right-padded with <pad> if the stream
    doesn't divide evenly — ignore_index=PAD in the loss already means
    those positions contribute nothing.
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, math.ceil(max(len(self.ids) - 1, 1) / self.block_size))

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        chunk = self.ids[start:start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [PAD] * (self.block_size + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# --- fine-tuning data: data/pairs.json -> prompt-masked chunks ---------

class ChatSFTDataset(Dataset):
    """Builds `prompt_tokens + <sep> + reply_tokens + <eos>` sequences from
    data/pairs.json's existing {"prompt", "reply"} format, then masks the
    loss over every position that predicts a prompt/<sep> token — only
    "given this prompt, produce this reply" should be scored, not "predict
    the user's own words back" (same masking idea chats.py already used for
    <pad>, extended here to also blank out the prompt span).
    """

    def __init__(self, pairs: list[dict], tokenizer: BPETokenizer, block_size: int):
        self.block_size = block_size
        self.examples: list[tuple[list[int], int]] = []
        for pair in pairs:
            prompt_ids = tokenizer.encode(pair["prompt"])
            reply_ids = tokenizer.encode(pair["reply"])
            seq = (prompt_ids + [SEP] + reply_ids + [EOS])[: block_size + 1]
            prompt_len = min(len(prompt_ids) + 1, len(seq))  # +1 accounts for <sep>
            self.examples.append((seq, prompt_len))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        seq, prompt_len = self.examples[idx]
        seq = seq + [PAD] * (self.block_size + 1 - len(seq))
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        y[: max(prompt_len - 1, 0)] = PAD
        return x, y


# --- shared training loop -----------------------------------------------

def _run_epochs(model: GPT, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, scheduler, epochs: int, patience: int,
                 device: str, tag: str, baseline_loss: float) -> list[dict]:
    """`baseline_loss` is ln(vocab_size) -- the loss an untrained model
    with a uniform random output distribution would score. Unlike OCR.py's
    CTC loss (where the warning thresholds this loop's structure was copied
    from are fixed absolute numbers), a language model's cross-entropy
    scale is directly a function of vocab_size, so "train loss is still
    high" has to be judged relative to that baseline, not a hardcoded
    constant -- a run with vocab_size=20000 has a ~30% higher baseline than
    one with vocab_size=8000 for reasons that have nothing to do with
    under/overfitting.
    """
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(val_loss)

        warning = ""
        if val_loss > train_loss * 1.3 and train_loss < baseline_loss:
            warning = "  [warning: val loss well above train loss -- possible overfitting]"
        elif train_loss > baseline_loss * 0.7 and epoch >= max(3, epochs // 3):
            warning = "  [warning: train loss still high this far in -- possible underfitting]"

        print(f"[{tag}] epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

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
    return history


# --- stage 1: pretraining ------------------------------------------------

def pretrain(corpus_path: Path, out_dir: Path = DEFAULT_PRETRAIN_DIR, epochs: int = 20,
             batch_size: int = 32, block_size: int = 512, n_layer: int = 8, n_embd: int = 384,
             n_head: int = 6, dropout: float = 0.1, lr: float = 3e-4, weight_decay: float = 0.01,
             vocab_size: int = 8000, val_split: float = 0.1, patience: int = 5,
             device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    text = Path(corpus_path).read_text(encoding="utf-8")

    tokenizer = BPETokenizer.train([text], vocab_size=vocab_size)
    ids = tokenizer.encode(text)

    split = int(len(ids) * (1 - val_split))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(val_ids) < 2:
        # tiny smoke-test corpus: reuse the training stream for validation
        # rather than crash on an empty split -- there is no real held-out
        # signal at this scale, this is only for exercising the pipeline.
        val_ids = train_ids

    train_loader = DataLoader(LMChunkDataset(train_ids, block_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LMChunkDataset(val_ids, block_size), batch_size=batch_size, shuffle=False)

    model = GPT(tokenizer.vocab_size, block_size, n_layer, n_embd, n_head, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = _run_epochs(model, train_loader, val_loader, optimizer, scheduler,
                           epochs, patience, device, tag="pretrain",
                           baseline_loss=math.log(tokenizer.vocab_size))

    # tokenizer/config/weights all written together *after* training finishes
    # (see chats.py train()'s identical fix) — writing the tokenizer first and
    # model.pt last would leave out_dir in a mismatched state (new vocab, old
    # or missing weights) for the entire training run, which crashes any
    # concurrent reader (e.g. a chat REPL pointed at this checkpoint) with a
    # tensor-shape mismatch instead of just seeing the old, still-good model.
    tokenizer.save(out_dir)
    config = {
        "block_size": block_size, "n_layer": n_layer, "n_embd": n_embd,
        "n_head": n_head, "dropout": dropout,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, tokenizer, history


# --- stage 2: fine-tuning -------------------------------------------------

def finetune(data_path: Path, pretrain_dir: Path = DEFAULT_PRETRAIN_DIR,
             out_dir: Path = DEFAULT_FINETUNE_DIR, epochs: int = 50, batch_size: int = 8,
             lr: float = 1e-4, weight_decay: float = 0.01, dropout: float = 0.1,
             val_split: float = 0.1, patience: int = 8, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = json.loads(Path(data_path).read_text(encoding="utf-8"))
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_split)))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    if not val_pairs:
        val_pairs = train_pairs

    pretrain_dir = Path(pretrain_dir)
    tokenizer = BPETokenizer.load(pretrain_dir)
    config = json.loads((pretrain_dir / "config.json").read_text(encoding="utf-8"))
    block_size = config["block_size"]

    model = GPT(tokenizer.vocab_size, block_size, config["n_layer"], config["n_embd"],
                config["n_head"], dropout).to(device)
    model.load_state_dict(torch.load(pretrain_dir / "model.pt", map_location=device))

    train_loader = DataLoader(ChatSFTDataset(train_pairs, tokenizer, block_size),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ChatSFTDataset(val_pairs, tokenizer, block_size),
                             batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = _run_epochs(model, train_loader, val_loader, optimizer, scheduler,
                           epochs, patience, device, tag="finetune",
                           baseline_loss=math.log(tokenizer.vocab_size))

    # see pretrain()'s identical comment: tokenizer/config/weights written
    # together after training finishes, not before, so out_dir never sits in
    # a mismatched half-written state for the whole training run.
    tokenizer.save(out_dir)
    (out_dir / "config.json").write_text(json.dumps(dict(config, dropout=dropout), indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, tokenizer, history


# --- inference -------------------------------------------------------------

_loaded_gpt: dict = {}


def _load_gpt(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_gpt:
        return _loaded_gpt[key]

    out_dir = Path(out_dir)
    model_path, config_path = out_dir / "model.pt", out_dir / "config.json"
    if not (model_path.exists() and config_path.exists() and (out_dir / "bpe_vocab.json").exists()):
        return None

    tokenizer = BPETokenizer.load(out_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = GPT(tokenizer.vocab_size, config["block_size"], config["n_layer"],
                config["n_embd"], config["n_head"], config.get("dropout", 0.0))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, tokenizer, config["block_size"])
    _loaded_gpt[key] = loaded
    return loaded


def reply(message: str, out_dir: Path = DEFAULT_FINETUNE_DIR, max_new_tokens: int = 60,
          temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> str:
    """Generate a reply from the fine-tuned GPT checkpoint at out_dir.

    Returns a placeholder message (rather than crashing) if no checkpoint
    has been trained yet -- same contract as chats.py's chat_reply().
    """
    loaded = _load_gpt(Path(out_dir))
    if loaded is None:
        return "Transformer 模型尚未訓練，請先執行 `python transformer_chat.py pretrain --corpus ...`，再執行 `finetune`。"

    model, tokenizer, block_size = loaded
    prompt_ids = (tokenizer.encode(message) + [SEP])[-block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, top_p=top_p)
    generated = out[0, len(prompt_ids):].tolist()
    return tokenizer.decode(generated) or "..."


# --- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="From-scratch decoder-only Transformer chat model (pretrain + fine-tune)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("pretrain", help="unsupervised next-token pretraining on a plain text corpus")
    p_pre.add_argument("--corpus", type=Path, required=True, help="plain UTF-8 .txt file of training text")
    p_pre.add_argument("--out-dir", type=Path, default=DEFAULT_PRETRAIN_DIR)
    p_pre.add_argument("--epochs", type=int, default=20)
    p_pre.add_argument("--batch-size", type=int, default=32)
    p_pre.add_argument("--block-size", type=int, default=512)
    p_pre.add_argument("--n-layer", type=int, default=8)
    p_pre.add_argument("--n-embd", type=int, default=384)
    p_pre.add_argument("--n-head", type=int, default=6)
    p_pre.add_argument("--dropout", type=float, default=0.1)
    p_pre.add_argument("--lr", type=float, default=3e-4)
    p_pre.add_argument("--weight-decay", type=float, default=0.01)
    p_pre.add_argument("--vocab-size", type=int, default=8000)
    p_pre.add_argument("--val-split", type=float, default=0.1)
    p_pre.add_argument("--patience", type=int, default=5)

    p_fin = sub.add_parser("finetune", help="supervised fine-tuning on a pairs.json-format dataset")
    p_fin.add_argument("--data", type=Path, required=True)
    p_fin.add_argument("--pretrain-dir", type=Path, default=DEFAULT_PRETRAIN_DIR)
    p_fin.add_argument("--out-dir", type=Path, default=DEFAULT_FINETUNE_DIR)
    p_fin.add_argument("--epochs", type=int, default=50)
    p_fin.add_argument("--batch-size", type=int, default=8)
    p_fin.add_argument("--dropout", type=float, default=0.1)
    p_fin.add_argument("--lr", type=float, default=1e-4)
    p_fin.add_argument("--weight-decay", type=float, default=0.01)
    p_fin.add_argument("--val-split", type=float, default=0.1)
    p_fin.add_argument("--patience", type=int, default=8)

    p_chat = sub.add_parser("chat", help="REPL against a fine-tuned checkpoint")
    p_chat.add_argument("--out-dir", type=Path, default=DEFAULT_FINETUNE_DIR)
    p_chat.add_argument("--max-new-tokens", type=int, default=60)
    p_chat.add_argument("--temperature", type=float, default=0.8)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    if args.command == "pretrain":
        pretrain(args.corpus, args.out_dir, args.epochs, args.batch_size, args.block_size,
                  args.n_layer, args.n_embd, args.n_head, args.dropout, args.lr,
                  args.weight_decay, args.vocab_size, args.val_split, args.patience)
    elif args.command == "finetune":
        finetune(args.data, args.pretrain_dir, args.out_dir, args.epochs, args.batch_size,
                  args.lr, args.weight_decay, args.dropout, args.val_split, args.patience)
    elif args.command == "chat":
        print("Chat with the fine-tuned Transformer model (type 'exit' to quit)")
        while True:
            text = input("You: ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            print(f"Model: {reply(text, out_dir=args.out_dir, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)}")


if __name__ == "__main__":
    main()
