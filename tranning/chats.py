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

2026-09-08 typo-robustness augmentation (--typo-noise-prob, default 0.4): each
training example, on every epoch, has a chance of having its *prompt only*
(never the reply target) run through inject_typos() — random char-level
substitute/delete/insert/adjacent-swap noise drawn from the training vocab
itself. Reason: this is still a character-level exact-match memorizer (see
above), so a single mistyped character used to turn a known prompt into an
unseen one and produce a wrong/garbled reply. Applied inside
ChatPairsDataset.__getitem__ rather than by writing extra rows into
pairs.json, so the same 297 pairs get a different noisy variant each epoch
for free — no bigger dataset file, no extra epochs, one training run.

Usage:
    python chats.py --data path/to/pairs.json --epochs 50
    python chats.py --chat                      # REPL against the last checkpoint
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import code_retrieval
import transformer_chat
from bayesian_utils import low_confidence_warning, majority_vote, mc_dropout_mode
from tools import route_reply

if getattr(sys, "frozen", False):
    # PyInstaller onefile (nova.exe, built to live at the project root):
    # __file__ resolves inside the temp extraction dir at runtime, which has
    # none of the actual checkpoints — use the running exe's own directory
    # instead, same fix as lib/components/cli.py.
    _TRANNING_DIR = Path(sys.executable).resolve().parent / "tranning"
    _LIB_DIR = str(Path(sys.executable).resolve().parent / "lib")
else:
    _TRANNING_DIR = Path(__file__).resolve().parent
    # lib/NVIDIA.py 所在資料夾：跟 conversation.py 把 tranning/ 加進 sys.path
    # 是同一種寫法，只是這裡指向專案根目錄底下的 lib/。只有 force_mode="nvidia"
    # 真的被呼叫到時才會 import（見 _nvidia_reply()），sinco/sinco-code 這兩個
    # 從零訓練的預設模型完全不依賴這個資料夾或 openai 套件。
    _LIB_DIR = str(Path(__file__).resolve().parent.parent / "lib")

# tranning/MentalHealth/action.py 的情緒傾向分數模型（0~5 有界迴歸，見該檔案
# docstring）——跟這裡的 sinco 聊天模型是兩個獨立訓練、獨立 checkpoint 的
# 東西，這裡只是借用它的 predict_emotion() 在 trace 裡多報一行「使用者這句話
# 的情緒判斷」，不影響 sinco 的回覆內容。沒有 __init__.py，用跟 action.py
# 自己 import train_utils 同一種手動加 sys.path 的寫法。
_MENTAL_HEALTH_DIR = str(_TRANNING_DIR / "MentalHealth")
if _MENTAL_HEALTH_DIR not in sys.path:
    sys.path.insert(0, _MENTAL_HEALTH_DIR)
import action as emotion_action  # noqa: E402

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK}

DEFAULT_OUT_DIR = _TRANNING_DIR / "chat_runs"
# Code snippets ("寫一個氣泡排序法" -> a real function body) run 100-200+
# characters, ~5x longer than a casual chat reply. Mixing both lengths into
# one checkpoint made every reply collapse to the same garbage output (the
# few long code examples dominated the loss and overwhelmed the short chat
# examples in the same 128-dim hidden state) — so code gets its own
# checkpoint, trained with a longer --max-len, and smart_reply() below picks
# which checkpoint to use per message.
CODE_OUT_DIR = _TRANNING_DIR / "code_runs"
MAX_LEN = 40
# Rule 06 的算力上限（RTX 4060 8G）：訓練用量不得超過半張卡，用
# torch.cuda.set_per_process_memory_fraction 在程式碼裡直接鎖死，超用時
# CUDA 會直接丟 OutOfMemoryError，而不是只能靠訓練前後跑 nvidia-smi 人工
# 觀察（to-do #24 那次的做法）才發現有沒有超量。
DEFAULT_GPU_MEM_FRACTION = 0.5

_CODE_PREFIXES = ("寫一個", "寫一段", "寫個")


def is_code_request(message: str) -> bool:
    """Heuristic: does this look like "write me a snippet that does X"
    (routes to CODE_OUT_DIR) rather than a general chat/capability question
    like "你能幫我寫程式嗎" (routes to the normal chat checkpoint)?
    """
    text = message.strip()
    if text.startswith(_CODE_PREFIXES):
        return True
    if text.lower().startswith("python") and "寫" in text:
        return True
    # code_snippet_import.py 的 build_prompt() 產生的多語言 prompt 模板是
    # 「用 <語言> 寫一個/寫一段/寫個 ...」（例如「用 C++ 寫一個氣泡排序法」），
    # 開頭是語言字樣而不是「寫一個」，原本單純 startswith(_CODE_PREFIXES)
    # 比對不到，會讓這批多語言程式碼訓練資料在實際聊天路由時被誤判成一般
    # 聊天請求——這裡額外比對「用...寫」這個形狀。
    if text.startswith("用") and any(prefix in text for prefix in _CODE_PREFIXES):
        return True
    return False


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


# 2026-09-08 錯字容忍度：使用者實際打字會漏字/多字/選錯同位鍵/選錯注音候選字，
# 但這個字元級模型本來就是逐字比對，訓練資料只有「乾淨」的句子，稍微打錯字
# 就等於變成沒看過的全新輸入（跟 to-do #12/#14 記錄的「沒背過就答錯」是同一個
# 根因）。修法不是換架構，是資料增強：訓練時把 prompt（只有 prompt，reply
# 維持原文）隨機做字元級增刪換位，讓模型看過同一句話的多種錯字版本仍要對應同一
# 個正確回覆。char_pool 直接從這批訓練資料自己的 vocab 取字（排除特殊
# token），不是任意 unicode 字元，錯字換出來的字才會落在中英文混雜語料本來
# 就會出現的字元範圍內，也不會讓 vocab 變大、不吃額外 GPU 顯存。
_TYPO_OPS = ("substitute", "delete", "insert", "swap")
_TYPO_CHAR_RATE = 0.15  # 一句話被選中做增強時，每個字元的變動機率


def inject_typos(text: str, char_rate: float, char_pool: list[str]) -> str:
    """Return `text` with random per-character typos (substitute/delete/
    insert/adjacent-swap), each character mutated independently with
    probability `char_rate`. `char_pool` supplies the replacement characters
    for substitute/insert — pass characters drawn from the training vocab so
    injected noise stays inside the language mix the model already knows.
    """
    if char_rate <= 0 or not char_pool:
        return text
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        if random.random() >= char_rate:
            out.append(chars[i])
            i += 1
            continue
        op = random.choice(_TYPO_OPS)
        if op == "delete":
            i += 1
        elif op == "substitute":
            out.append(random.choice(char_pool))
            i += 1
        elif op == "insert":
            out.append(chars[i])
            out.append(random.choice(char_pool))
            i += 1
        elif op == "swap" and i + 1 < len(chars):
            out.append(chars[i + 1])
            out.append(chars[i])
            i += 2
        else:  # 選到 swap 但已經是最後一個字，沒有下一個字可換
            out.append(chars[i])
            i += 1
    return "".join(out)


class ChatPairsDataset(Dataset):
    def __init__(self, pairs: list[dict], vocab: dict, max_len: int,
                 typo_prob: float = 0.0, typo_char_pool: list[str] | None = None):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len
        # typo_prob：整句話（而非每個字）被選中做錯字增強的機率。刻意兩層
        # 機率（先整句抽籤，抽中才逐字用 _TYPO_CHAR_RATE 變動）而不是單一
        # 逐字機率，是為了讓「乾淨、一字不差」的原始句子仍佔多數訓練樣本，
        # 保留既有精準重現能力（to-do #14 驗證過的行為），不會因為每個 epoch
        # 都在跟雜訊版本學習而反過來讓精準比對變差。__getitem__ 每次被
        # DataLoader 呼叫都重新抽籤/重新產生雜訊，同一筆資料在不同 epoch
        # 會看到不同的錯字版本，等於免費做資料擴增，不用把 pairs.json 實際
        # 膨脹成好幾倍、也不用因此多訓練好幾輪。
        self.typo_prob = typo_prob
        self.typo_char_pool = typo_char_pool or []

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        if self.typo_prob > 0 and random.random() < self.typo_prob:
            prompt = inject_typos(prompt, _TYPO_CHAR_RATE, self.typo_char_pool)
        src = encode(prompt, self.vocab, self.max_len)
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
          device: str | None = None, gpu_mem_fraction: float = DEFAULT_GPU_MEM_FRACTION,
          typo_noise_prob: float = 0.4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, torch.cuda.current_device())
    pairs = json.loads(Path(data_path).read_text(encoding="utf-8"))
    vocab = build_vocab(pairs)
    typo_char_pool = [tok for tok in vocab if tok not in SPECIAL_TOKENS]

    dataset = ChatPairsDataset(pairs, vocab, max_len, typo_prob=typo_noise_prob, typo_char_pool=typo_char_pool)
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

    # vocab.json/config.json 跟權重檔永遠一起寫（而非分開）：如果只先寫
    # vocab.json、權重檔晚點才存，中途任何人（CLI、GUI）讀取這個 checkpoint
    # 都會因為 embedding 維度對不上而直接 Error(s) in loading state_dict。
    # 四個檔案綁在同一個函式裡一次寫完，讓 out_dir 隨時只會是「完整的舊
    # checkpoint」或「完整的新 checkpoint」兩種狀態之一。
    def save_checkpoint():
        (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "config.json").write_text(
            json.dumps({"embed_size": embed_size, "hidden_size": hidden_size, "max_len": max_len}, indent=2)
        )
        torch.save(encoder.state_dict(), out_dir / "encoder.pt")
        torch.save(decoder.state_dict(), out_dir / "decoder.pt")

    # progress.json 每個 epoch 覆寫一次（跟 history.json 不同，這份只放
    # "現在跑到哪" 的單一快照，不用整份讀完再自己找最後一筆），訓練跑到一半
    # 也能隨時打開看目前 epoch/loss/預估剩餘時間，不用等 epochs 全部跑完
    # （原本 history.json 只在迴圈結束後才寫一次，訓練中途完全看不到進度）。
    def save_progress(epoch: int, avg_loss: float | None, elapsed: float, eta: float, done: bool):
        (out_dir / "progress.json").write_text(json.dumps({
            "epoch": epoch,
            "epochs": epochs,
            "percent": round(epoch / epochs * 100, 1),
            "loss": avg_loss,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "done": done,
        }, indent=2))

    history = []
    start_time = time.time()
    best_loss = float("inf")
    params = list(encoder.parameters()) + list(decoder.parameters())
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
            # 沒有裁剪時，長時間訓練偶爾會在某個 epoch 梯度暴衝，把已經收斂
            # 好的權重瞬間打壞，且後面的 epoch 回不去（3000 epochs 那次從
            # loss 0.65 一路發散到 4.1 就是這樣）——clip 到 max_norm=5 是
            # RNN 訓練的標準做法，擋掉單步暴衝但不影響正常梯度下降。
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            total_loss += step_loss.item()
            total_tokens += valid_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (epochs - epoch)
        # flush=True: 沒有這個，被導向檔案/pipe 時 print 會整批緩衝，
        # 中途讀檔看到的永遠是空的，要等訓練完全跑完才會一次冒出來
        # （這次重訓 chat_runs 就是這樣才看不到進度）。
        is_best = avg_loss < best_loss
        print(f"epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
              f"{'  (best)' if is_best else ''}", flush=True)
        history.append({"epoch": epoch, "loss": avg_loss})
        if is_best:
            # 每個 epoch 都存最佳狀態，而不是只存訓練跑完當下那個 epoch：
            # 上面的梯度裁剪能降低暴衝機率，但沒辦法保證整個訓練過程都不會
            # 有比較差的尾段——與其事後才發現最後一個 epoch 比中間差，不如
            # 全程都保留看過的最佳結果。
            best_loss = avg_loss
            save_checkpoint()
        # history.json/progress.json 每個 epoch 都寫（跟 vocab/config/權重的
        # 「訓練跑完才一次寫」不同）：這兩份只是進度紀錄，不會被拿去
        # load_state_dict，中途寫壞或被強制中止時最多只是進度數字停在中途，
        # 不會像權重檔那樣造成 size mismatch，所以可以放心即時更新，讓外部
        # 隨時打開 progress.json 就能看到目前 epoch/loss/預估剩餘時間。
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_progress(epoch, avg_loss, elapsed, eta, done=False)

    save_progress(epochs, history[-1]["loss"] if history else None, time.time() - start_time, 0.0, done=True)
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


_REPEAT_MAX_PERIOD, _REPEAT_MIN_CYCLES = 6, 3


def _is_repeating(ids: list[int]) -> bool:
    """True once the tail of `ids` is some short cycle (period 1..6, e.g. a
    single repeated character or a repeated multi-char unit like ", AI")
    repeated _REPEAT_MIN_CYCLES times in a row. An undertrained checkpoint's
    greedy decode has no built-in reason to ever stop repeating itself once
    it locks onto such a cycle (this is exactly how "我是 AI, AI, AI, AI..."
    happens) — this is an inference-time safety net, not a fix for the
    underlying undertraining, so it only fires on genuine loops, not on a
    legitimately short repeated word.
    """
    for period in range(1, _REPEAT_MAX_PERIOD + 1):
        needed = period * _REPEAT_MIN_CYCLES
        if len(ids) < needed:
            continue
        tail = ids[-needed:]
        if all(tail[i] == tail[i % period] for i in range(needed)):
            return True
    return False


def _greedy_decode(encoder: Encoder, decoder: Decoder, src: torch.Tensor,
                    max_len: int, idx2word: dict, temperature: float = 0.0) -> str:
    """One deterministic-shaped decode pass. Not actually deterministic when
    dropout is left stochastic by the caller (mc_dropout_mode) — that's the
    point: mc_chat_reply() calls this DEFAULT_MC_SAMPLES times and the
    dropout masks make each call a different sample from the approximate
    posterior.

    temperature=0.0 (default) keeps this a true greedy decode (argmax, same
    as before). temperature>0 instead scales logits by 1/temperature, softmax,
    and samples (torch.multinomial) — low values (<1) stay close to greedy,
    high values (>1) pick lower-probability characters more often. This is a
    second, independent source of variety on top of MC Dropout: dropout
    perturbs *which* posterior sample is decoded, temperature perturbs *how
    committed* that decode is to its own top choice at each character.
    """
    output_ids = []
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src)
        src_mask = src == PAD
        decoder_input = torch.tensor([[SOS]])
        for _ in range(max_len):
            logits, hidden = decoder(decoder_input, hidden, encoder_outputs, src_mask)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = logits.argmax(1).item()
            if next_id == EOS:
                break
            output_ids.append(next_id)
            if _is_repeating(output_ids):
                break
            decoder_input = torch.tensor([[next_id]])
    return "".join(idx2word.get(i, "<unk>") for i in output_ids)


def mc_chat_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR,
                   mc_samples: int = DEFAULT_MC_SAMPLES, temperature: float = 0.0) -> tuple[str, float]:
    """Bayesian (MC Dropout) reply generation: greedy-decodes `mc_samples`
    times with dropout deliberately left on (bayesian_utils.mc_dropout_mode),
    then majority-votes across the resulting replies. Returns
    (reply, confidence), where confidence is the fraction of samples that
    landed on that exact reply — 1.0 means every sample agreed, low values
    mean the model produced several different guesses for this prompt.

    temperature is forwarded to _greedy_decode() unchanged (0.0 = greedy,
    same behaviour as before this parameter existed); raising it makes the
    per-sample decodes more varied, so confidence should be expected to drop
    as temperature rises even for prompts the model knows well.

    Returns (placeholder_message, 0.0) rather than crashing if no checkpoint
    has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return _NOT_TRAINED_MESSAGE, 0.0

    encoder, decoder, vocab, idx2word, max_len = loaded
    src = torch.tensor([encode(message, vocab, max_len)])

    with mc_dropout_mode(encoder), mc_dropout_mode(decoder):
        samples = [_greedy_decode(encoder, decoder, src, max_len, idx2word, temperature)
                   for _ in range(mc_samples)]

    reply, confidence = majority_vote(samples)
    return (reply or "..."), confidence


def chat_reply(message: str, out_dir: Path = DEFAULT_OUT_DIR,
               mc_samples: int = DEFAULT_MC_SAMPLES, temperature: float = 0.0) -> str:
    """Generate a reply from the self-trained seq2seq model.

    Kept as a plain str-returning function (smart_reply(), web/backend/app.py
    and app/components/conversation.py all depend on that shape) — internally
    this is now mc_chat_reply()'s majority-vote reply; use mc_chat_reply()
    directly when the confidence score is also needed.
    """
    reply, _confidence = mc_chat_reply(message, out_dir=out_dir, mc_samples=mc_samples, temperature=temperature)
    return reply


def _nvidia_reply(message: str, history: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """/model nvidia 專用：呼叫 lib/NVIDIA.py 的 NVIDIA 雲端 API（外部模型，
    非本專案自訓練——CLAUDE.md 規則 #06 預設仍是 sinco/sinco-code，這個模式
    要使用者透過 /model nvidia 明確選用才會走到這裡）。openai 套件與對外
    網路請求都只在這個分支才會發生，不影響其餘模式。

    `history` 原樣轉交給 nvidia_reply() 組成多輪 messages（見該函式說明）——
    這裡不額外處理，只是單純傳遞，避免呼叫端跟實際組 messages 的邏輯分散
    在兩個檔案。

    回傳 (思考過程, 正式回覆)——nvidia_reply() 已經把 NVIDIA API 串流回應
    裡的 reasoning_content（思考過程）跟 content（正式回覆）分開收集。
    """
    if _LIB_DIR not in sys.path:
        sys.path.insert(0, _LIB_DIR)
    from NVIDIA import nvidia_reply

    try:
        return nvidia_reply(message, history=history)
    except Exception as exc:
        return "", f"NVIDIA API 呼叫失敗：{exc}"


def _emotion_trace_suffix(message: str) -> str:
    """Detection-only add-on for the trace string: reports what
    tranning/MentalHealth/action.py's predict_emotion() thinks `message`'s
    emotional valence is (0~5 score + label) plus its MC Dropout confidence,
    or "" if that checkpoint hasn't been trained yet — never raises, never
    changes the actual sinco reply. mc_confidence is agreement across
    stochastic dropout passes, not correctness — a wrong prediction can
    still land at 100% if the model is *consistently* wrong about it.
    """
    result = emotion_action.predict_emotion(message)
    if result["status"] is not None:
        return ""
    return (f"\n情緒判斷：{result['label_name']}（分數 {result['score']:.1f}/5，"
            f"MC Dropout 信心度 {result['mc_confidence']:.0%}）")


def smart_reply_traced(message: str, out_dir: Path = DEFAULT_OUT_DIR,
                        force_mode: str = "auto", temperature: float = 0.0,
                        history: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """Like smart_reply(), but also returns *why* that path answered the
    message — the actual rule/pattern that fired, not a decorative label and
    not a fabricated reasoning chain (sinco is a small memorization model,
    it has no real step-by-step reasoning to show; this reports the real
    routing decision instead, which is the honest version of "thinking").

    force_mode overrides the automatic chat/code routing (CLAUDE.md 需求
    #01 的 /model 指令): "auto" (default, existing behaviour, unchanged) =
    decide via is_code_request(); "sinco" = always answer with the general
    chat checkpoint at out_dir even if the message looks code-shaped;
    "code" = always answer with the code checkpoint even if it doesn't;
    "nvidia" = 明確選用外部 NVIDIA 雲端模型（見 _nvidia_reply()），是唯一
    會離開本機、呼叫外部 API 的模式，其餘模式維持 Rule 06「全部自建」。
    route_reply()（天氣/搜尋等即時查詢）一律優先，不受 force_mode 影響——
    那是誠實資料查詢，跟「要用哪個聊天 checkpoint 回答」是兩件事。

    temperature 轉呼叫 mc_chat_reply()（見該函式說明），0.0（預設）等同原本
    行為不變；只影響 sinco/sinco-code 兩個生成分支，不影響 route_reply() 或
    code_retrieval 命中既有範例原文那兩條路徑（沒有「生成」這回事可調）。

    history 只有在 force_mode="nvidia" 時才會被用到（轉交給 _nvidia_reply()
    組成多輪 messages）。sinco/sinco-code 這兩個字元級模型故意不吃歷史（見
    lib/components/conversation.py 開頭的說明——塞歷史字串反而會把回覆拉走），
    所以其餘分支完全忽略這個參數，呼叫端可以無條件傳，不用依模式判斷要不要帶。
    """
    routed = route_reply(message)
    if routed is not None:
        return routed
    if force_mode == "nvidia":
        reasoning, reply = _nvidia_reply(message, history=history)
        trace = "已手動切換為 NVIDIA 雲端模型（nemotron-3-ultra，外部 API，非本專案自訓練）"
        if reasoning:
            trace += f"\n思考過程：\n{reasoning}"
        return trace, reply
    use_code = force_mode == "code" or (force_mode == "auto" and is_code_request(message))
    if use_code:
        reason_prefix = "已手動切換為程式碼模式" if force_mode == "code" else \
            '訊息以「寫一個／寫一段」或「用<語言>寫」開頭，判斷為程式碼請求'
        # 先試檢索式：data/code_pairs.json 裡有夠相似的既有範例就直接回傳
        # 原文，不經過模型生成——比 sinco-code 生成可靠得多（見
        # code_retrieval.py 開頭說明），資料庫沒覆蓋到才退回模型生成。
        retrieved = code_retrieval.retrieve(message)
        if retrieved is not None:
            reply, score, matched_prompt = retrieved
            return (
                f'{reason_prefix} → 檢索式命中既有訓練範例「{matched_prompt}」'
                f'（字元相似度 {score:.0%}，直接回傳訓練資料原文，非模型生成）',
                reply,
            )
        reply, confidence = mc_chat_reply(message, out_dir=CODE_OUT_DIR, temperature=temperature)
        warning = low_confidence_warning(confidence)
        return (
            f'{reason_prefix} → 資料庫沒有夠相似的既有範例，改用 sinco-code 模型生成'
            f'（貝氏 MC Dropout 信心度 {confidence:.0%}）{warning}' + _emotion_trace_suffix(message),
            reply,
        )
    # 2026-09-10：一般聊天分支正式從 chats.py 自己的字元級 GRU（chat_runs）
    # 切換成 transformer_chat.py 的 Transformer checkpoint（gpt_pretrain_runs
    # + gpt_chat_runs）——這是 transformer_chat.py 模組docstring 一開始就說
    # 「等實際驗證比 GRU 好才切換」的那一步，今天用同一批情緒陪伴測試句反覆
    # 比較過（repetition penalty 修生成迴圈、調低 lr/加大 weight decay 修過
    # 擬合、原始問答 oversample 3 倍找回身份/能力問答）確認 Transformer 版本
    # 明顯更好才換。GRU 的 mc_chat_reply()/chat_reply() 函式本身保留不刪，
    # 只是不再是 smart_reply_traced() 預設路徑，仍可用 chats.py --chat 直接
    # 呼叫。沒有 MC Dropout 信心度這個機制可以沿用（Transformer 走取樣式
    # 生成，不是 GRU 的多次 dropout forward 投票），trace 改成誠實地只說明
    # 是哪個模型在回覆，不硬套一個不適用的信心度數字。
    reply = transformer_chat.reply(message)
    reason = ("已手動切換為一般聊天模式" if force_mode == "sinco" else
              "沒有比對到即時查詢或程式碼請求的句型")
    return (
        f"{reason} → 交給 sinco 一般聊天模型（Transformer，2026-09-10 上線）" + _emotion_trace_suffix(message),
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
    parser.add_argument("--gpu-mem-fraction", type=float, default=DEFAULT_GPU_MEM_FRACTION,
                         help="cap CUDA memory to this fraction of the GPU (Rule 06: RTX 4060 8G, must not exceed half)")
    parser.add_argument("--typo-noise-prob", type=float, default=0.4,
                         help="probability a training prompt gets random char-level typo noise injected "
                              "each epoch (0 disables); reply targets are never touched")
    parser.add_argument("--chat", action="store_true",
                         help="skip training; start an interactive REPL against --out-dir's checkpoint")
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="--chat only: 0.0 (default) = greedy decode, same as before this flag existed; "
                              ">0 scales logits by 1/temperature and samples instead of always taking the top "
                              "character (<1 mostly-greedy, >1 more varied/riskier wording)")
    args = parser.parse_args()

    if args.chat:
        print(f"Chat with your self-trained model (type 'exit' to quit, temperature={args.temperature})")
        while True:
            text = input("You: ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            trace, reply = smart_reply_traced(text, out_dir=args.out_dir, temperature=args.temperature)
            print(f"[{trace}]")
            print(f"Model: {reply}")
        return

    if not args.data:
        parser.error("--data is required unless --chat is given")

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.embed_size,
          args.hidden_size, args.lr, args.max_len, args.teacher_forcing_ratio,
          args.dropout, args.weight_decay, gpu_mem_fraction=args.gpu_mem_fraction,
          typo_noise_prob=args.typo_noise_prob)


if __name__ == "__main__":
    main()
