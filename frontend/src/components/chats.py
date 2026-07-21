"""Chat model — from-scratch sequence-to-sequence scaffold (no external AI API).

Point this at a JSON file of [{"prompt": ..., "reply": ...}, ...] pairs once a
real conversation dataset is available. Right now, with no data provided yet,
this only defines the architecture (GRU encoder-decoder) and training loop;
it is smoke-tested against a handful of synthetic pairs in test_chats.py (no
reply-quality claims — just "the pipeline runs").

Architecture: a character-level tokenizer + vocab built from the training
pairs (character-level rather than whitespace-split so it works for Chinese,
which has no spaces between words, as well as English), a GRU encoder that
reads the prompt into a single hidden state, and a GRU decoder that generates
the reply character-by-character (teacher forcing during training, greedy
decoding at inference). This mirrors the classic "seq2seq chatbot from
scratch" design — no pretrained weights, no calls to any cloud LLM API. Swap
the tokenizer for something smarter (jieba for Chinese, BPE, ...) once real
data is in and the language mix is known.

Usage:
    python chats.py --data path/to/pairs.json --epochs 50
    python chats.py --chat                      # REPL against the last checkpoint
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK}

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "chat_runs"
MAX_LEN = 40


def tokenize(text: str) -> list[str]:
    return list(text.strip())


def build_vocab(pairs: list[dict]) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    for pair in pairs:
        for text in (pair["prompt"], pair["reply"]):
            for token in tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK) for t in tokenize(text)][: max_len - 1]
    ids.append(EOS)
    ids += [PAD] * (max_len - len(ids))
    return ids


class ChatPairsDataset(Dataset):
    def __init__(self, pairs: list[dict], vocab: dict, max_len: int):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src = encode(pair["prompt"], self.vocab, self.max_len)
        tgt = encode(pair["reply"], self.vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, src):
        _, hidden = self.gru(self.embedding(src))
        return hidden  # (1, batch, hidden_size)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, hidden):
        # input_step: (batch, 1)
        output, hidden = self.gru(self.embedding(input_step), hidden)
        return self.out(output.squeeze(1)), hidden


def train(data_path: Path, out_dir: Path, epochs: int, batch_size: int,
          embed_size: int, hidden_size: int, lr: float, max_len: int,
          teacher_forcing_ratio: float, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = json.loads(Path(data_path).read_text(encoding="utf-8"))
    vocab = build_vocab(pairs)

    dataset = ChatPairsDataset(pairs, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(len(vocab), embed_size, hidden_size).to(device)
    decoder = Decoder(len(vocab), embed_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    # sum reduction + manual normalization by valid-token count: with mean
    # reduction, a batch whose targets are all PAD at some timestep (short
    # replies fully padded past their length) divides 0/0 -> NaN loss.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"embed_size": embed_size, "hidden_size": hidden_size, "max_len": max_len}, indent=2)
    )

    history = []
    for epoch in range(1, epochs + 1):
        total_loss, total_tokens = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            hidden = encoder(src)
            decoder_input = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=device)

            step_loss = torch.tensor(0.0, device=device)
            teacher_forcing = random.random() < teacher_forcing_ratio
            for t in range(max_len):
                logits, hidden = decoder(decoder_input, hidden)
                step_loss = step_loss + criterion(logits, tgt[:, t])
                decoder_input = tgt[:, t].unsqueeze(1) if teacher_forcing else logits.argmax(1, keepdim=True)

            valid_tokens = int((tgt != PAD).sum().item())
            loss = step_loss / max(valid_tokens, 1)
            loss.backward()
            optimizer.step()
            total_loss += step_loss.item()
            total_tokens += valid_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        print(f"epoch {epoch:3d}  loss={avg_loss:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss})

    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    torch.save(decoder.state_dict(), out_dir / "decoder.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return encoder, decoder, vocab, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    vocab_path, config_path = out_dir / "vocab.json", out_dir / "config.json"
    encoder_path, decoder_path = out_dir / "encoder.pt", out_dir / "decoder.pt"
    if not (vocab_path.exists() and config_path.exists() and encoder_path.exists() and decoder_path.exists()):
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    idx2word = {i: w for w, i in vocab.items()}

    encoder = Encoder(len(vocab), config["embed_size"], config["hidden_size"])
    decoder = Decoder(len(vocab), config["embed_size"], config["hidden_size"])
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    encoder.eval()
    decoder.eval()

    loaded = (encoder, decoder, vocab, idx2word, config["max_len"])
    _loaded_models[key] = loaded
    return loaded


def chat_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Generate a reply from the self-trained seq2seq model.

    Returns a placeholder message (rather than crashing) if no checkpoint has
    been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return "模型尚未訓練，請先提供對話資料集並執行 `python chats.py --data <pairs.json>` 進行訓練。"

    encoder, decoder, vocab, idx2word, max_len = loaded
    src = torch.tensor([encode(message, vocab, max_len)])

    output_ids = []
    with torch.no_grad():
        hidden = encoder(src)
        decoder_input = torch.tensor([[SOS]])
        for _ in range(max_len):
            logits, hidden = decoder(decoder_input, hidden)
            next_id = logits.argmax(1).item()
            if next_id == EOS:
                break
            output_ids.append(next_id)
            decoder_input = torch.tensor([[next_id]])

    return "".join(idx2word.get(i, "<unk>") for i in output_ids) or "..."


def main():
    parser = argparse.ArgumentParser(description="Train or chat with a from-scratch seq2seq chat model")
    parser.add_argument("--data", type=Path, help='JSON file of [{"prompt": ..., "reply": ...}, ...] pairs')
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--chat", action="store_true",
                         help="skip training; start an interactive REPL against --out-dir's checkpoint")
    args = parser.parse_args()

    if args.chat:
        print("Chat with your self-trained model (type 'exit' to quit)")
        while True:
            text = input("You: ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            print(f"Model: {chat_reply(text, out_dir=args.out_dir)}")
        return

    if not args.data:
        parser.error("--data is required unless --chat is given")

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.embed_size,
          args.hidden_size, args.lr, args.max_len, args.teacher_forcing_ratio)


if __name__ == "__main__":
    main()
