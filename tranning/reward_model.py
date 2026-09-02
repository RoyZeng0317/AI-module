"""reward_model.py — 學習型 Reward Model（Bradley-Terry pairwise 訓練），
取代 tranning/train_grpo.py 原本的規則式 reward（json_format_reward()／
accuracy_reward()，靠 regex／字串比對，只能覆蓋寫得出規則的窄任務）。

三個設計決定，記錄一下為什麼：

(a) 為什麼用「成對偏好」而不是直接回歸一個絕對分數：真正的偏好資料天生
就是比較性的（「A 比 B 好」），不是絕對分數——人（或這裡示範用的合成
資料）很難一致地給「這段回覆值 7.3 分」這種絕對評分，但「A 比 B 好」
容易判斷得多。這正是 RLHF 論文（Christiano et al., 2017；InstructGPT,
2022）採用的 Bradley-Terry 配對比較模型：loss = -log(sigmoid(r_chosen -
r_rejected))，用 `F.softplus(-(r_chosen - r_rejected))` 算（數學上等價，
數值上更穩定的寫法）。

(b) 為什麼取「最後一個真實 token」的 hidden state，而不是像
character_model.py 的 CharacterEncoder 那樣 mean-pool：這裡接的是
transformer_chat.py 的 GPT trunk，是單向 causal attention（見
CausalSelfAttention 的 causal_mask + key_padding_mask），每個位置只看得到
它之前的 token；只要 padding 都在右邊，最後一個真實 token 的 hidden state
就已經是「整段 prompt+completion」的因果摘要。mean-pool 會把還沒看到
completion 的早期 prompt hidden state也平均進去，稀釋掉 completion 本身
帶來的訊號。character_model.py 用 mean-pool 是因為它是雙向 GRU，每個位置
本來就看得到整句話，跟這裡的情況不一樣。

(c) 為什麼 checkpoint 要自帶完整的 GPT trunk 權重，而不是只存 reward head
再依賴 base_dir 之後還在：reward model 訓練完之後可能會被拿去評分很久
（train_grpo.py 每次 GRPO rollout 都要呼叫），但 base_dir 指向的 SFT
checkpoint 之後可能又被重新 finetune、覆蓋掉。把 trunk+head 一起存進
out_dir，reward model 就能獨立載入評分，不用擔心 base_dir 還在不在。

Usage:
    python reward_model.py --data data/reward_pairs.jsonl --epochs 30
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bpe_tokenizer import BPETokenizer, EOS, PAD, SEP
from train_utils import EarlyStopper, plateau_scheduler
from transformer_chat import GPT, DEFAULT_FINETUNE_DIR

DEFAULT_REWARD_DIR = Path(__file__).resolve().parent / "reward_runs"


# --- 01. RewardModel：GPT trunk（可從 SFT checkpoint 載入權重）+ 純量 head ---

class RewardModel(nn.Module):
    def __init__(self, gpt: GPT, n_embd: int):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(n_embd, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embd) -- 每個位置在 causal trunk 底下算出來的 hidden state
        _, _, x = self.gpt(idx, return_hidden=True)
        last_pos = (idx != PAD).sum(dim=1).clamp(min=1) - 1  # 每列最後一個真實 token 的位置
        pooled = x[torch.arange(idx.size(0), device=idx.device), last_pos]  # (B, n_embd)
        return self.head(pooled).squeeze(-1)  # (B,)


# --- 02. 偏好資料：{"prompt","chosen","rejected"} 一行一筆的 JSONL ----------

def load_pairs(jsonl_path: Path) -> list[dict]:
    pairs = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


class PreferencePairDataset(Dataset):
    """每筆組成 prompt + <sep> + chosen/rejected + <eos>，跟 transformer_chat.py
    的 ChatSFTDataset 同一套序列組法，讓 reward model 看到的輸入格式跟
    SFT/GRPO 一致。回傳未 padding 的 id list，交給 collate_pairs 在各自 batch
    內動態 padding（batch 之間長度可能不同，不必固定成 block_size）。
    """

    def __init__(self, pairs: list[dict], tokenizer: BPETokenizer, block_size: int):
        self.examples: list[tuple[list[int], list[int]]] = []
        for pair in pairs:
            prompt_ids = tokenizer.encode(pair["prompt"])
            chosen_ids = (prompt_ids + [SEP] + tokenizer.encode(pair["chosen"]) + [EOS])[:block_size]
            rejected_ids = (prompt_ids + [SEP] + tokenizer.encode(pair["rejected"]) + [EOS])[:block_size]
            self.examples.append((chosen_ids, rejected_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_pairs(batch: list[tuple[list[int], list[int]]]):
    chosen_seqs, rejected_seqs = zip(*batch)
    max_len = max(max(len(s) for s in chosen_seqs), max(len(s) for s in rejected_seqs))

    def pad_batch(seqs):
        return torch.tensor([s + [PAD] * (max_len - len(s)) for s in seqs], dtype=torch.long)

    return pad_batch(chosen_seqs), pad_batch(rejected_seqs)


# --- 03. 一個 epoch：pairwise loss + pairwise accuracy -----------------------

def run_epoch(model: RewardModel, loader: DataLoader, optimizer, device: str, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for chosen, rejected in loader:
            chosen, rejected = chosen.to(device), rejected.to(device)
            if train:
                optimizer.zero_grad()
            r_chosen = model(chosen)
            r_rejected = model(rejected)
            # Bradley-Terry：-log(sigmoid(r_chosen - r_rejected))，用
            # softplus(-x) 這個數值穩定的等價寫法。
            loss = F.softplus(-(r_chosen - r_rejected)).mean()
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * chosen.size(0)
            correct += (r_chosen > r_rejected).sum().item()
            total += chosen.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# --- 04. 載入 SFT checkpoint 當作 trunk 起點（沿用 train_grpo._load_policy 同款寫法）---

def _load_base_gpt(base_dir: Path, device: str):
    base_dir = Path(base_dir)
    model_path, config_path = base_dir / "model.pt", base_dir / "config.json"
    if not (model_path.exists() and config_path.exists() and (base_dir / "bpe_vocab.json").exists()):
        return None
    tokenizer = BPETokenizer.load(base_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gpt = GPT(tokenizer.vocab_size, config["block_size"], config["n_layer"],
              config["n_embd"], config["n_head"], config.get("dropout", 0.0))
    gpt.load_state_dict(torch.load(model_path, map_location=device))
    return gpt, tokenizer, config


# --- 05. 主訓練迴圈 ----------------------------------------------------------

def train_reward_model(data_path: Path, base_dir: Path = DEFAULT_FINETUNE_DIR,
                        out_dir: Path = DEFAULT_REWARD_DIR, epochs: int = 30,
                        batch_size: int = 8, lr: float = 1e-4, weight_decay: float = 0.01,
                        val_split: float = 0.1, patience: int = 6, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loaded = _load_base_gpt(base_dir, device)
    if loaded is None:
        print(f"尚未找到可用的基礎 checkpoint：{base_dir}\n"
              "請先執行 `python transformer_chat.py pretrain --corpus ...` "
              "再執行 `finetune`，reward model 是接在 SFT checkpoint 上繼續訓練。")
        return None

    gpt, tokenizer, config = loaded
    block_size = config["block_size"]
    model = RewardModel(gpt, config["n_embd"]).to(device)

    pairs = load_pairs(data_path)
    if not pairs:
        print(f"{data_path} 沒有可用的偏好資料（需要至少一筆 {{prompt,chosen,rejected}}）。")
        return None
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_split)))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    if not val_pairs:
        val_pairs = train_pairs

    train_loader = DataLoader(PreferencePairDataset(train_pairs, tokenizer, block_size),
                               batch_size=batch_size, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(PreferencePairDataset(val_pairs, tokenizer, block_size),
                             batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)
    stopper = EarlyStopper(patience)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"[reward] epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if stopper.step(val_loss, model, epoch):
            break
    stopper.restore_best(model)

    # 跟 pretrain()/finetune()/grpo_train() 同一個理由：全部檔案訓練完才一起
    # 寫，不要先寫 tokenizer/config 再寫權重，避免 out_dir 在訓練過程中
    # 停留在「字典是新的、權重是舊的/還沒寫完」這種被其他人讀到會崩潰的
    # 中間狀態（ErrorLog #9 的競態條件教訓）。這裡刻意不用
    # train_utils.save_checkpoint()：它只寫 best_model.pt，沒有 config/
    # tokenizer，會讓這份 checkpoint 之後只能靠 base_dir 還在才讀得回來，
    # 跟本檔案模組 docstring (c) 的自足性設計相違背。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), out_dir / "reward_model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return model, tokenizer, history


# --- 06. 推論：載入 checkpoint + 對 (prompt, completion) 評分 ----------------

_loaded_reward_models: dict = {}


def load_reward_model(out_dir: Path, device: str = "cpu"):
    out_dir = Path(out_dir)
    key = str(out_dir)
    if key in _loaded_reward_models:
        return _loaded_reward_models[key]

    required = ("reward_model.pt", "config.json", "bpe_vocab.json", "bpe_merges.json")
    if not all((out_dir / f).exists() for f in required):
        return None

    tokenizer = BPETokenizer.load(out_dir)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    gpt = GPT(tokenizer.vocab_size, config["block_size"], config["n_layer"],
              config["n_embd"], config["n_head"], config.get("dropout", 0.0))
    model = RewardModel(gpt, config["n_embd"])
    model.load_state_dict(torch.load(out_dir / "reward_model.pt", map_location=device))
    model.to(device).eval()

    result = (model, tokenizer, config["block_size"])
    _loaded_reward_models[key] = result
    return result


@torch.no_grad()
def score(model: RewardModel, tokenizer: BPETokenizer, prompt: str, completion: str,
          block_size: int, device: str) -> float:
    model.eval()
    ids = (tokenizer.encode(prompt) + [SEP] + tokenizer.encode(completion) + [EOS])[:block_size]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model(x).item()


# --- CLI ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="用成對偏好資料訓練 reward model")
    parser.add_argument("--data", type=Path,
                         default=Path(__file__).resolve().parent / "data" / "reward_pairs.jsonl")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_FINETUNE_DIR,
                         help="做為 trunk 起點的 SFT checkpoint 目錄（transformer_chat.py finetune 的輸出）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    train_reward_model(args.data, args.base_dir, args.out_dir, args.epochs, args.batch_size,
                        args.lr, args.weight_decay, args.val_split, args.patience)


if __name__ == "__main__":
    main()
