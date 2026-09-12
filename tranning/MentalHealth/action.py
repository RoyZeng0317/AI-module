"""情緒傾向分數模型(從零訓練，Rule 06：不叫外部 AI API)。

讀取 data/mental.csv(text, score 兩欄；score 是 0~5 的情緒傾向分數，
1=負向/壓力偏高、2=偏負向、3=中性/平淡、4=偏正向、5=正向；0 目前只有
1 筆資料，語意上接近中性)。架構比照 tranning/character_model.py：
character-level tokenizer 建 vocab(中英文都吃，不用斷詞器)，餵進雙向
GRU，對非 PAD 位置做 mean-pool 得到句子向量，接一個線性頭 + sigmoid
輸出一個 0~1 的正規化分數(乘回 SCORE_MAX 還原成 0~5)。

新版 mental.csv 從舊版的 0/1 二元擴成 0~5 六級，而且六級的筆數極不平均
(score 0 只有 1 筆、score 3 只有 6 筆、score 5 只有 7 筆，score 1/2/4
則各有 50/169/190 筆)。如果照舊做法把 score 當成離散類別分類，那極少數
的 0/3/5 類別模型只能死記硬背那幾筆的字面內容，換一句話術寫法就會判斷
錯——不是真的學會「這句話的情緒傾向大概落在哪裡」。

這次改成有界迴歸(bounded regression：sigmoid 輸出 * 5 夾在 [0,5])而不是
離散分類的原因：迴歸目標在數值上是連續、有順序的(1 到 5 是漸強的正向
傾向)，模型從樣本數多的 1/2/4 學到「用詞越負向分數越低、越正向分數越
高」這個連續趨勢後，對樣本數極少的 0/3/5 就能用內插(interpolate)推測
出接近的分數，而不需要每個分數都各自背下足夠多的例句。loss 用
SmoothL1(Huber)而不是 MSE，離群的少數類別分數不會把梯度拉爆，逼模型
硬記那幾筆。

另外新增一個「推論測試集」(test_split，預設 15%，train() 完全不會拿去
訓練或做 early stopping，只在訓練結束、還原最佳 checkpoint 後評估一次)：
這是專門用來檢查模型是不是真的在「推測」分數，而不是把訓練集背下來。
每筆測試資料的實際分數 vs. 模型預測分數會存成
`generalization_report.json`，方便人工檢查模型對沒看過的句子是否合理。

train/val/test 三路切分不能直接對原始 score 做 stratify(score=0 只有
1 筆，sklearn stratify 至少要 2 筆才能切)，改成先把 0~5 分數粗分成
「負向(<=1)/中性(2~3)/正向(>=4)」三個大桶做 stratify，確保三個切分
裡負向/中性/正向的比例都跟原始資料一致，細部分數的樣本則是各桶內隨機
分配。

設計取捨(維持過擬合／欠擬合的既有考量)：

  欠擬合(underfitting)：
    - 雙向 GRU 讓短句子(3~25 字)兩個方向的 context 都看得到。
    - ReduceLROnPlateau 只在驗證 loss 停滯時才降學習率。

  過擬合／死記硬背(overfitting)：
    - 分類頭前面加 dropout + AdamW weight decay(比舊版二元分類略高，
      因為新版分數更容易被少數類別的字面內容牽著走)。
    - Early stopping(看驗證 loss)，還原最佳 checkpoint。
    - vocab 只從訓練集切出來的文字建，不看驗證/測試集。
    - SmoothL1 迴歸 + 有界輸出(sigmoid*5)取代離散分類，讓模型學到的是
      連續的情緒傾向趨勢，而不是逐筆記憶的類別標籤。
    - 額外切出訓練/驗證都不會碰的推論測試集，訓練完成後才評估一次。

Usage:
    python action.py --epochs 60                        # 用 data/mental.csv 訓練
    python action.py --predict "今天心情很糟糕"           # 用已訓練好的 checkpoint 推測分數(0~5)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_MODULE_DIR = Path(__file__).resolve().parent          # tranning/MentalHealth
_TRANNING_DIR = _MODULE_DIR.parent                      # tranning
_PROJECT_ROOT = _TRANNING_DIR.parent                    # AI-module 專案根目錄
if str(_TRANNING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANNING_DIR))

from bayesian_utils import majority_vote, mc_dropout_mode  # noqa: E402
from train_utils import EarlyStopper, accuracy_gap_warning, plateau_scheduler, save_checkpoint  # noqa: E402

DEFAULT_MC_SAMPLES = 20

DEFAULT_CSV = _PROJECT_ROOT / "data" / "mental.csv"
DEFAULT_OUT_DIR = _MODULE_DIR / "emotion_runs"

PAD, UNK = 0, 1
SPECIAL_TOKENS = {"<pad>": PAD, "<unk>": UNK}
MAX_LEN = 40  # 資料集句子長度 3~25 字，留一點餘裕
SCORE_MIN, SCORE_MAX = 0, 5
LABEL_NAMES = {
    0: "中性／其他",
    1: "負向（低落／壓力）",
    2: "偏負向（焦慮／迷惘）",
    3: "中性（平靜／日常）",
    4: "偏正向（放鬆／愉快）",
    5: "正向（成就感／幸福）",
}


# --- 01. 讀取 CSV(沿用原本嘗試多種編碼的作法) --------------------------------

def load_dataset(csv_path: Path) -> tuple[list[str], list[int]]:
    df = None
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"使用編碼 '{enc}' 讀取檔案！")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise FileNotFoundError(f"無法讀取 CSV 檔案: {csv_path}(所有編碼皆嘗試失敗，或檔案不存在)")

    if not {"text", "score"}.issubset(df.columns):
        raise ValueError(f"{csv_path} 需要包含 'text' 與 'score' 兩個欄位，目前欄位: {list(df.columns)}")

    df = df.dropna(subset=["text", "score"])
    print(f"目前載入的資料總筆數: {len(df)}")
    if len(df) < 200:
        print("警告: 資料筆數少於 200 筆，建議擴充您的 CSV data set(資料集)")

    texts = df["text"].astype(str).tolist()
    labels = df["score"].astype(int).tolist()
    out_of_range = [s for s in labels if not (SCORE_MIN <= s <= SCORE_MAX)]
    if out_of_range:
        raise ValueError(f"{csv_path} 有分數超出 {SCORE_MIN}~{SCORE_MAX} 範圍: {sorted(set(out_of_range))}")
    return texts, labels


def _valence_bucket(score: int) -> int:
    """把 0~5 的細分數粗分成負向/中性/正向三桶，只用來做 stratify 切分。

    score=0 只有 1 筆、score=3 只有 6 筆，直接對原始 score stratify 會因為
    單一類別筆數不足而噴錯，所以改成用這三個桶(每桶都有 50 筆以上)。
    """
    if score <= 1:
        return 0  # 負向
    if score <= 3:
        return 1  # 中性
    return 2  # 正向


# --- 02. character-level vocab + padding ------------------------------------

def tokenize(text: str) -> list[str]:
    return list(str(text).strip())


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids


class EmotionDataset(Dataset):
    def __init__(self, texts: list[str], scores: list[int], vocab: dict, max_len: int):
        self.texts = texts
        self.scores = scores
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(encode(self.texts[idx], self.vocab, self.max_len))
        score = torch.tensor(float(self.scores[idx]))  # 原始 0~5 分數，正規化留給 run_epoch 做
        return input_ids, score


# --- 03. 模型：Embedding + BiGRU + mean-pool + 有界迴歸頭 ---------------------

class EmotionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != PAD).unsqueeze(-1)  # (batch, max_len, 1)
        outputs, _ = self.gru(self.embedding(input_ids))  # (batch, max_len, hidden*2)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean over real tokens only
        pooled = self.dropout(pooled)
        return torch.sigmoid(self.head(pooled).squeeze(-1))  # (batch,) 正規化分數 in [0, 1]


# --- 04. 一個 epoch：SmoothL1 迴歸 loss + MAE + 四捨五入命中率 ----------------

def run_epoch(model: EmotionEncoder, loader: DataLoader, criterion, optimizer, device: str, train: bool):
    model.train(mode=train)
    total_loss, total_abs_err, exact_correct, total = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, scores in loader:
            input_ids, scores = input_ids.to(device), scores.to(device)
            target_norm = scores / SCORE_MAX
            if train:
                optimizer.zero_grad()
            pred_norm = model(input_ids)
            loss = criterion(pred_norm, target_norm)
            if train:
                loss.backward()
                optimizer.step()
            pred_score = (pred_norm * SCORE_MAX).clamp(SCORE_MIN, SCORE_MAX)
            total_loss += loss.item() * input_ids.size(0)
            total_abs_err += (pred_score - scores).abs().sum().item()
            exact_correct += (pred_score.round() == scores.round()).sum().item()
            total += input_ids.size(0)
    return total_loss / total, total_abs_err / total, exact_correct / total  # loss, mae, rounded_acc


# --- 05. 主訓練迴圈 ----------------------------------------------------------

def train(csv_path: Path = DEFAULT_CSV, out_dir: Path = DEFAULT_OUT_DIR, epochs: int = 60,
          batch_size: int = 8, embed_size: int = 64, hidden_size: int = 128, lr: float = 1e-3,
          max_len: int = MAX_LEN, dropout: float = 0.35, weight_decay: float = 2e-4,
          val_split: float = 0.15, test_split: float = 0.15, patience: int = 8,
          device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    texts, scores = load_dataset(Path(csv_path))

    # 先切出「推論測試集」——訓練與驗證全程不會看到，訓練完才評估一次，
    # 用來檢查模型是不是真的在推測分數而不是背訓練集。
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        texts, scores, test_size=test_split, random_state=42,
        stratify=[_valence_bucket(s) for s in scores],
    )
    # 剩下的再切訓練/驗證(val_split 是相對於「原始全部資料」的比例)
    relative_val = val_split / (1 - test_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=relative_val, random_state=42,
        stratify=[_valence_bucket(s) for s in y_trainval],
    )
    print(f"訓練集樣本數: {len(x_train)}, 驗證集樣本數: {len(x_val)}, 推論測試集樣本數: {len(x_test)}")

    vocab = build_vocab(x_train)  # 只用訓練集建字元表，避免驗證/測試集洩漏

    train_loader = DataLoader(EmotionDataset(x_train, y_train, vocab, max_len),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmotionDataset(x_val, y_val, vocab, max_len),
                             batch_size=batch_size, shuffle=False)

    model = EmotionEncoder(len(vocab), embed_size, hidden_size, dropout).to(device)
    criterion = nn.SmoothL1Loss()  # Huber loss：對少數類別的離群分數比較不敏感，減少死記硬背的壓力
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"embed_size": embed_size, "hidden_size": hidden_size, "max_len": max_len,
                     "dropout": dropout, "score_min": SCORE_MIN, "score_max": SCORE_MAX}, indent=2)
    )

    stopper = EarlyStopper(patience)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_mae, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        warning = accuracy_gap_warning(train_acc, val_acc, epoch, epochs,
                                        underfit_hint="調大 --hidden-size 或拉長 --epochs")

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_mae={train_mae:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_mae={val_mae:.3f} val_acc={val_acc:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae, "train_acc": train_acc,
                         "val_loss": val_loss, "val_mae": val_mae, "val_acc": val_acc})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)

    _evaluate_generalization(model, x_test, y_test, vocab, max_len, device, out_dir)
    return model, vocab, history


def _evaluate_generalization(model: EmotionEncoder, x_test: list[str], y_test: list[int], vocab: dict,
                              max_len: int, device: str, out_dir: Path) -> list[dict]:
    """訓練/驗證都沒看過的推論測試集，跑一次前向推論並存成報告。

    重點不是拿來調參(調參要看 val_loss)，而是讓人可以肉眼檢查：模型對沒
    看過的句子推測出來的分數合不合理，尤其是原始資料筆數很少的 0/3/5 分。
    """
    model.eval()
    report: list[dict] = []
    with torch.no_grad():
        for text, actual_score in zip(x_test, y_test):
            input_ids = torch.tensor([encode(text, vocab, max_len)]).to(device)
            pred_norm = model(input_ids).item()
            predicted_score = pred_norm * SCORE_MAX
            report.append({
                "text": text,
                "actual_score": actual_score,
                "predicted_score": round(predicted_score, 2),
                "predicted_label": int(round(min(max(predicted_score, SCORE_MIN), SCORE_MAX))),
            })

    if report:
        mae = sum(abs(r["predicted_score"] - r["actual_score"]) for r in report) / len(report)
        exact_acc = sum(r["predicted_label"] == r["actual_score"] for r in report) / len(report)
        print(f"\n[推論測試集](未參與訓練/驗證，n={len(report)})  MAE={mae:.3f}  四捨五入命中率={exact_acc:.3f}")
        for r in report[:10]:
            print(f"  實際={r['actual_score']}  預測={r['predicted_score']:.2f}  {r['text']}")

    (out_dir / "generalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# --- 06. 推論：載入 checkpoint 對一句話推測分數 -------------------------------

_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    vocab_path, config_path, model_path = out_dir / "vocab.json", out_dir / "config.json", out_dir / "best_model.pt"
    if not (vocab_path.exists() and config_path.exists() and model_path.exists()):
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = EmotionEncoder(len(vocab), config["embed_size"], config["hidden_size"], config["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, vocab, config)
    _loaded_models[key] = loaded
    return loaded


def predict_emotion(text: str, out_dir: Path = DEFAULT_OUT_DIR, mc_samples: int = DEFAULT_MC_SAMPLES) -> dict:
    """`mc_confidence` (added 2026-09-10): same MC Dropout idea chats.py's
    mc_chat_reply() already uses for chat replies, adapted to this bounded-
    regression head — `mc_samples` stochastic forward passes (dropout left
    on, bayesian_utils.mc_dropout_mode) each rounded to the same 0-5 integer
    label the eval-mode prediction below uses, then majority_vote()'d. High
    agreement across samples means the model is consistently landing on the
    same label despite the random dropout masks; low agreement is an honest
    "not sure" signal, most likely for text unlike anything in mental.csv's
    423 rows or landing near a label boundary (e.g. predicted 2.4 vs 2.6).
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return {
            "text": text, "score": None, "label": None, "label_name": None, "mc_confidence": None,
            "status": "模型尚未訓練，請先執行 `python action.py --epochs ...` 進行訓練。",
        }

    model, vocab, config = loaded
    input_ids = torch.tensor([encode(text, vocab, config["max_len"])])
    score_max = config.get("score_max", SCORE_MAX)
    score_min = config.get("score_min", SCORE_MIN)

    with torch.no_grad():
        pred_norm = model(input_ids).item()
    score = pred_norm * score_max
    label = int(round(min(max(score, score_min), score_max)))

    with mc_dropout_mode(model), torch.no_grad():
        votes = [int(round(min(max(model(input_ids).item() * score_max, score_min), score_max)))
                 for _ in range(mc_samples)]
    _, mc_confidence = majority_vote(votes)

    return {
        "text": text, "score": round(score, 2), "label": label,
        "label_name": LABEL_NAMES.get(label, str(label)), "mc_confidence": mc_confidence, "status": None,
    }


# --- CLI ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="從零訓練情緒傾向分數模型(0~5，越高越正向)")
    parser.add_argument("--data", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15,
                         help="留給訓練/驗證都不會碰的推論測試集比例，用來檢查模型是否真的在推測分數")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--predict", type=str, default=None,
                         help="跳過訓練，直接用 --out-dir 的 checkpoint 對這句話推測分數(0~5)")
    args = parser.parse_args()

    if args.predict is not None:
        result = predict_emotion(args.predict, out_dir=args.out_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.embed_size, args.hidden_size,
          args.lr, args.max_len, args.dropout, args.weight_decay, args.val_split, args.test_split, args.patience)


if __name__ == "__main__":
    main()
