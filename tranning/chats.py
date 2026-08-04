"""Chat model — from-scratch sequence-to-sequence scaffold (no external AI API).

Point this at a JSON file of [{"prompt": ..., "reply": ...}, ...] pairs once a
real conversation dataset is available. Right now, with no data provided yet,
this only defines the architecture (GRU encoder-decoder) and training loop;
it is smoke-tested against a handful of synthetic pairs in test_chats.py (no
reply-quality claims — just "the pipeline runs").

Architecture: a character-level tokenizer + vocab built from the training
pairs (character-level rather than whitespace-split so it works for Chinese,
which has no spaces between words, as well as English), a GRU encoder that
reads the prompt into a per-position sequence of hidden states, and a GRU
decoder with Luong-style attention over those encoder states that generates
the reply character-by-character (teacher forcing during training, greedy
decoding at inference). This mirrors the classic "seq2seq chatbot from
scratch" design — no pretrained weights, no calls to any cloud LLM API.
Attention was added after a plain (no-attention) GRU decoder reliably
memorized short chat replies but collapsed to a single answer for every
prompt once targets got long (100-200+ character code snippets) — squeezing
a whole reply through one fixed hidden vector doesn't scale, attention lets
the decoder look back at specific source positions each step instead. Swap
the tokenizer for something smarter (jieba for Chinese, BPE, ...) once real
data is in and the language mix is known.

2026-07-28 Bayesian upgrade (MC Dropout, Gal & Ghahramani 2016): the encoder
and decoder each gained a real nn.Dropout layer (--dropout), and training
moved from plain Adam to AdamW (--weight-decay) — under a Bayesian reading,
weight decay is a Gaussian prior on the weights, so "dropout + weight decay"
together are the standard cheap stand-in for a fully variational network
(which would need every weight replaced by its own mean+variance and cost
far more compute than an RTX 4060 8GB budget wants to spend on a chat
model). The training loop itself did NOT change — still ordinary gradient
descent on the same cross-entropy loss, no ELBO, no extra loss term. What
changed is inference: chat_reply()/mc_chat_reply() now run mc_samples
independent greedy decodes with dropout deliberately left ON (see
bayesian_utils.mc_dropout_mode) instead of a single deterministic pass, and
report the majority-vote reply plus a confidence score (how many of those
samples agreed). Honest caveat: this is not a way to need less training
data — the data requirement is unchanged — it's a way to get an honest "how
sure is the model" number instead of a single greedy decode that looks
equally confident whether the model actually learned the answer or is
guessing; with very little training data the confidence number can still
look artificially high if the model simply memorized that exact prompt.

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

from bayesian_utils import low_confidence_warning, majority_vote, mc_dropout_mode
from tools import route_reply

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK}

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "chat_runs"
# Code snippets ("寫一個氣泡排序法" -> a real function body) run 100-200+
# characters, ~5x longer than a casual chat reply. Mixing both lengths into
# one checkpoint made every reply collapse to the same garbage output (the
# few long code examples dominated the loss and overwhelmed the short chat
# examples in the same 128-dim hidden state) — so code gets its own
# checkpoint, trained with a longer --max-len, and smart_reply() below picks
# which checkpoint to use per message.
CODE_OUT_DIR = Path(__file__).resolve().parent / "code_runs"
MAX_LEN = 40

_CODE_PREFIXES = ("寫一個", "寫一段", "寫個")


def is_code_request(message: str) -> bool:
    """Heuristic: does this look like "write me a snippet that does X"
    (routes to CODE_OUT_DIR) rather than a general chat/capability question
    like "你能幫我寫程式嗎" (routes to the normal chat checkpoint)?
    """
    text = message.strip()
    if text.startswith(_CODE_PREFIXES):
        return True
    return text.lower().startswith("python") and "寫" in text


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
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        # dropout defaults to 0.0 so existing callers that construct Encoder
        # positionally without it (voice_clone.py reuses this class) are
        # unaffected; chats.py's own train()/load path passes a real value.
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        outputs = self.dropout(outputs)
        return outputs, hidden  # outputs: (batch, src_len, hidden_size); hidden: (1, batch, hidden_size)


class Attention(nn.Module):
    """Luong-style ("general") attention: score each encoder position against
    the current decoder hidden state via a learned bilinear map, softmax over
    non-PAD positions, then take the weighted sum as the context vector.

    Plain GRU seq2seq squeezes the *whole* source sentence through one fixed
    hidden_size vector, which falls apart once targets run past ~50-100
    characters (exactly what broke the code-snippet checkpoint — it kept
    collapsing to the same memorized answer no matter the prompt). Attention
    lets the decoder look back at specific source positions at every step
    instead of relying on that single bottleneck, which is the standard fix
    for long-sequence seq2seq — still trained from scratch, no external model.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        # decoder_hidden: (batch, hidden_size); encoder_outputs: (batch, src_len, hidden_size)
        # src_mask: (batch, src_len) bool, True where the source position is PAD
        energy = torch.bmm(self.attn(encoder_outputs), decoder_hidden.unsqueeze(2)).squeeze(2)
        energy = energy.masked_fill(src_mask, float("-inf"))
        weights = torch.softmax(energy, dim=1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_size)
        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_step, hidden, encoder_outputs, src_mask):
        # input_step: (batch, 1)
        embedded = self.dropout(self.embedding(input_step))  # (batch, 1, embed_size)
        context = self.attention(hidden.squeeze(0), encoder_outputs, src_mask)  # (batch, hidden_size)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        # dropout right before the output projection is where MC Dropout
        # inference (bayesian_utils.mc_dropout_mode) does the most good: it
        # directly perturbs the logits each stochastic pass, which is what
        # mc_chat_reply()'s majority vote needs to actually see disagreement.
        combined = self.dropout(torch.cat([output.squeeze(1), context], dim=1))
        return self.out(combined), hidden


def train(data_path: Path, out_dir: Path, epochs: int, batch_size: int,
          embed_size: int, hidden_size: int, lr: float, max_len: int,
          teacher_forcing_ratio: float, dropout: float = 0.3, weight_decay: float = 1e-4,
          device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = json.loads(Path(data_path).read_text(encoding="utf-8"))
    vocab = build_vocab(pairs)

    dataset = ChatPairsDataset(pairs, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(len(vocab), embed_size, hidden_size, dropout).to(device)
    decoder = Decoder(len(vocab), embed_size, hidden_size, dropout).to(device)
    # AdamW rather than plain Adam: its decoupled weight decay is a Gaussian
    # prior on the weights, which paired with the dropout above is this
    # model's (cheap, MC-Dropout-based) approximation to a Bayesian network
    # — see the module docstring's 2026-07-28 note.
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=weight_decay
    )
    # sum reduction + manual normalization by valid-token count: with mean
    # reduction, a batch whose targets are all PAD at some timestep (short
    # replies fully padded past their length) divides 0/0 -> NaN loss.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, epochs + 1):
        total_loss, total_tokens = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            encoder_outputs, hidden = encoder(src)
            src_mask = src == PAD
            decoder_input = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=device)

            step_loss = torch.tensor(0.0, device=device)
            teacher_forcing = random.random() < teacher_forcing_ratio
            for t in range(max_len):
                logits, hidden = decoder(decoder_input, hidden, encoder_outputs, src_mask)
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

    # vocab.json/config.json 跟權重檔一起在訓練「結束」後才寫入（而非開頭）：
    # 舊版在迴圈開始前就先寫 vocab.json，如果重新訓練時字元集有變動（新增了
    # 訓練資料），out_dir 會有一段長達整個訓練時間的空窗期，vocab.json 已經是
    # 新的、但 encoder.pt/decoder.pt 還是舊的——這段時間任何人（CLI、GUI）
    # 讀取這個 checkpoint 都會因為 embedding 維度對不上而直接 Error(s) in
    # loading state_dict。全部搬到訓練迴圈之後一起寫，讓 out_dir 隨時只會是
    #「完整的舊 checkpoint」或「完整的新 checkpoint」兩種狀態之一。
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"embed_size": embed_size, "hidden_size": hidden_size, "max_len": max_len}, indent=2)
    )
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

    # dropout has no learned parameters, so it doesn't affect
    # load_state_dict — reading it back from config.json is just for
    # architecture bookkeeping and so mc_dropout_mode() has a real dropout
    # probability (not the 0.0 default) to make stochastic at MC time.
    dropout = config.get("dropout", 0.0)
    encoder = Encoder(len(vocab), config["embed_size"], config["hidden_size"], dropout)
    decoder = Decoder(len(vocab), config["embed_size"], config["hidden_size"], dropout)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    encoder.eval()
    decoder.eval()

    loaded = (encoder, decoder, vocab, idx2word, config["max_len"])
    _loaded_models[key] = loaded
    return loaded


_NOT_TRAINED_MESSAGE = "模型尚未訓練，請先提供對話資料集並執行 `python chats.py --data <pairs.json>` 進行訓練。"
DEFAULT_MC_SAMPLES = 20


def _greedy_decode(encoder: Encoder, decoder: Decoder, src: torch.Tensor,
                    max_len: int, idx2word: dict) -> str:
    """One deterministic-shaped decode pass. Not actually deterministic when
    dropout is left stochastic by the caller (mc_dropout_mode) — that's the
    point: mc_chat_reply() calls this DEFAULT_MC_SAMPLES times and the
    dropout masks make each call a different sample from the approximate
    posterior.
    """
    output_ids = []
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src)
        src_mask = src == PAD
        decoder_input = torch.tensor([[SOS]])
        for _ in range(max_len):
            logits, hidden = decoder(decoder_input, hidden, encoder_outputs, src_mask)
            next_id = logits.argmax(1).item()
            if next_id == EOS:
                break
            output_ids.append(next_id)
            decoder_input = torch.tensor([[next_id]])
    return "".join(idx2word.get(i, "<unk>") for i in output_ids)


def mc_chat_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR,
                   mc_samples: int = DEFAULT_MC_SAMPLES) -> tuple[str, float]:
    """Bayesian (MC Dropout) reply generation: greedy-decodes `mc_samples`
    times with dropout deliberately left on (bayesian_utils.mc_dropout_mode),
    then majority-votes across the resulting replies. Returns
    (reply, confidence), where confidence is the fraction of samples that
    landed on that exact reply — 1.0 means every sample agreed, low values
    mean the model produced several different guesses for this prompt.

    Returns (placeholder_message, 0.0) rather than crashing if no checkpoint
    has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return _NOT_TRAINED_MESSAGE, 0.0

    encoder, decoder, vocab, idx2word, max_len = loaded
    src = torch.tensor([encode(message, vocab, max_len)])

    with mc_dropout_mode(encoder), mc_dropout_mode(decoder):
        samples = [_greedy_decode(encoder, decoder, src, max_len, idx2word) for _ in range(mc_samples)]

    reply, confidence = majority_vote(samples)
    return (reply or "..."), confidence


def chat_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR,
               mc_samples: int = DEFAULT_MC_SAMPLES) -> str:
    """Generate a reply from the self-trained seq2seq model.

    Kept as a plain str-returning function (smart_reply(), web/backend/app.py
    and app/components/conversation.py all depend on that shape) — internally
    this is now mc_chat_reply()'s majority-vote reply; use mc_chat_reply()
    directly when the confidence score is also needed.
    """
    reply, _confidence = mc_chat_reply(message, out_dir=out_dir, mc_samples=mc_samples)
    return reply


def smart_reply_traced(message: str, out_dir: Path = DEFAULT_OUT_DIR,
                        force_mode: str = "auto") -> tuple[str, str]:
    """Like smart_reply(), but also returns *why* that path answered the
    message — the actual rule/pattern that fired, not a decorative label and
    not a fabricated reasoning chain (sinco is a small memorization model,
    it has no real step-by-step reasoning to show; this reports the real
    routing decision instead, which is the honest version of "thinking").

    force_mode overrides the automatic chat/code routing (CLAUDE.md 需求
    #01 的 /model 指令): "auto" (default, existing behaviour, unchanged) =
    decide via is_code_request(); "sinco" = always answer with the general
    chat checkpoint at out_dir even if the message looks code-shaped;
    "code" = always answer with the code checkpoint even if it doesn't.
    route_reply()（天氣/搜尋等即時查詢）一律優先，不受 force_mode 影響——
    那是誠實資料查詢，跟「要用哪個聊天 checkpoint 回答」是兩件事。
    """
    routed = route_reply(message)
    if routed is not None:
        return routed
    use_code = force_mode == "code" or (force_mode == "auto" and is_code_request(message))
    if use_code:
        reply, confidence = mc_chat_reply(message, out_dir=CODE_OUT_DIR)
        warning = low_confidence_warning(confidence)
        reason = ("已手動切換為程式碼模式" if force_mode == "code" else
                  '訊息以「寫一個／寫一段」或「Python 怎麼寫」開頭，判斷為程式碼請求')
        return (
            f'{reason} → 使用 sinco-code 模型'
            f'（貝氏 MC Dropout 信心度 {confidence:.0%}）{warning}',
            reply,
        )
    reply, confidence = mc_chat_reply(message, out_dir=out_dir)
    warning = low_confidence_warning(confidence)
    reason = ("已手動切換為一般聊天模式" if force_mode == "sinco" else
              "沒有比對到即時查詢或程式碼請求的句型")
    return (
        f"{reason} → 交給 sinco 一般聊天模型"
        f"（貝氏 MC Dropout 信心度 {confidence:.0%}）{warning}",
        reply,
    )


def smart_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Reply to `message`, checking a live data lookup (tools.route_reply)
    before falling back to the trained seq2seq model.

    Weather/search-shaped messages ("台北天氣", "搜尋 X", "你認識 X 嗎", ...)
    get answered with real data pulled at call time — sinco's own model
    never has to memorize (and can't be wrong about) live facts. Everything
    else goes to chat_reply() as before.
    """
    _, reply = smart_reply_traced(message, out_dir=out_dir)
    return reply


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
    parser.add_argument("--dropout", type=float, default=0.3,
                         help="also the MC Dropout probability used at chat-time inference")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chat", action="store_true",
                         help="skip training; start an interactive REPL against --out-dir's checkpoint")
    args = parser.parse_args()

    if args.chat:
        print("Chat with your self-trained model (type 'exit' to quit)")
        while True:
            text = input("You: ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            trace, reply = smart_reply_traced(text, out_dir=args.out_dir)
            print(f"[{trace}]")
            print(f"Model: {reply}")
        return

    if not args.data:
        parser.error("--data is required unless --chat is given")

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.embed_size,
          args.hidden_size, args.lr, args.max_len, args.teacher_forcing_ratio,
          args.dropout, args.weight_decay)


if __name__ == "__main__":
    main()
