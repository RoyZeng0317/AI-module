"""自製「區域指稱」小模型 — 給一句話描述，從 YOLO 偵測到的多個候選框裡找出對應的那個。

概念上參考 om-ai-lab/VLX-Seek 公開說明的「用候選區域 token 做定位，而不是直接回歸
座標」這個想法，但完全乾淨室重新實作：不下載 VLX-Seek 的權重、不呼叫其 API、不照抄
其程式碼、不整合其依賴的 WeDetect 偵測器（AI-module 的 CLAUDE.md Rule 06：禁止使用
外部 AI 模型/雲端 API，所有模型自己從零設計、從零訓練，算力上限 RTX 4060 8GB）。

管線：
    YOLO 候選框（web/backend/detector.py 原有的 detect()，不開第二套推論）
    → 裁出每個候選框 → 本檔案從零設計的小型視覺區域編碼器 RegionEncoder
    → 跟既有 tranning/embedding_model.py 的自訓文字 embedding（凍結、不重新訓練）
      對比學習對齊到同一個向量空間
    → 文字查詢 embedding 跟每個候選框的區域 embedding 算 cosine 相似度，取最高分

目前沒有任何真實「框 + 文字描述」資料集，所以用合成資料（圓/方/三角形狀 + 模板化
顏色/方位描述句）先跑通整條管線，不代表真實語意理解能力，同 yolo_custom.py/
circuit_diagram_train.py 的「骨架先行、資料後補」慣例；test_region_refer.py 只驗證
管線跑得完。訓練前必須先有 embedding_model.py 的 checkpoint（python embedding_model.py
train），文字編碼器全程凍結、只訓練 RegionEncoder。

Usage:
    python region_refer.py --make-data region_refer_data --n 300
    python region_refer.py --train region_refer_data --epochs 30
    python region_refer.py --refer photo.jpg "左邊的紅色圓形" \
        --yolo-weights ../tranning/yolo_runs/custom/weights/best.pt
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "web" / "backend"), str(Path(__file__).resolve().parent)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bayesian_utils import low_confidence_warning  # noqa: E402
from embedding_model import DEFAULT_OUT_DIR as TEXT_ENCODER_DIR  # noqa: E402
from embedding_model import embed_texts, load_encoder  # noqa: E402
from train_utils import EarlyStopper, plateau_scheduler  # noqa: E402

RUNS = Path(__file__).resolve().parent / "region_refer_runs"
IMG_SIZE = 64
CROP_SIZE = 32  # RegionEncoder 的輸入邊長，跟合成資料的裁圖/真實偵測框裁圖都先 resize 成這個
COLORS = {"紅色": (0, 0, 255), "綠色": (0, 200, 0), "藍色": (255, 80, 0), "黃色": (0, 220, 220)}
SHAPES = ("圓形", "方形", "三角形")


# --- 合成資料：圓/方/三角 + 模板化顏色/方位描述句 ---------------------------

def _draw_shape(canvas, shape: str, color: tuple, cx: int, cy: int, r: int):
    if shape == "圓形":
        cv2.circle(canvas, (cx, cy), r, color, -1)
    elif shape == "方形":
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:
        pts = np.array([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])
        cv2.fillPoly(canvas, [pts], color)


def _position_word(cx: int, cy: int, size: int) -> str:
    half = size / 2
    horiz = "左邊" if cx < half * 0.8 else "右邊" if cx > half * 1.2 else "中間"
    vert = "上面" if cy < half * 0.8 else "下面" if cy > half * 1.2 else ""
    return (vert + horiz) if vert else horiz


def make_synthetic(out: Path, n: int = 300, size: int = IMG_SIZE, seed: int = 0) -> Path:
    """每張合成畫面畫 2~3 個不同顏色/形狀的物件，裁出每個物件當一筆 (crop, caption) 樣本，
    存成 <out>/pairs.jsonl + <out>/crops/*.jpg；回傳 pairs.jsonl 路徑。"""
    rng, out = random.Random(seed), Path(out)
    crop_dir = out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    pairs, idx = [], 0
    for _ in range(n):
        canvas = np.full((size, size, 3), 30, np.uint8)
        placed = []
        for _ in range(rng.randint(2, 3)):
            shape, (color_name, color) = rng.choice(SHAPES), rng.choice(list(COLORS.items()))
            r = rng.randint(6, 10)
            cx, cy = rng.randint(r + 2, size - r - 2), rng.randint(r + 2, size - r - 2)
            _draw_shape(canvas, shape, color, cx, cy, r)
            placed.append((shape, color_name, cx, cy, r))
        for shape, color_name, cx, cy, r in placed:
            caption = f"{_position_word(cx, cy, size)}的{color_name}{shape}"
            x1, y1, x2, y2 = max(cx - r - 2, 0), max(cy - r - 2, 0), min(cx + r + 2, size), min(cy + r + 2, size)
            crop_path = crop_dir / f"{idx}.jpg"
            cv2.imwrite(str(crop_path), canvas[y1:y2, x1:x2])
            pairs.append({"crop": str(crop_path.relative_to(out).as_posix()), "caption": caption})
            idx += 1

    pairs_path = out / "pairs.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
    return pairs_path


def _load_crop_tensor(path: Path, crop_size: int = CROP_SIZE) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"無法讀取裁圖：{path}")
    img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0  # BGR->RGB, HWC->CHW


# --- 視覺區域編碼器（從零設計，不是分類頭）----------------------------------

class RegionEncoder(nn.Module):
    """小型 CNN：裁圖 -> 捲積特徵 -> 投影到跟文字 embedding 同一個維度 -> L2 normalize，
    跟 embedding_model.SentenceEncoder 的輸出對齊到同一個向量空間，才能直接算 cosine
    相似度（結構參考 image_classifier_bnn.FeatureCNN 的捲積骨架，但輸出 embedding
    而不是分類 logits）。"""

    def __init__(self, embed_dim: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.features(x).flatten(1)), p=2, dim=-1)


def _info_nce(region_vecs: torch.Tensor, text_vecs: torch.Tensor, temperature: float) -> torch.Tensor:
    """跟 embedding_model._simcse_loss 同一套 InfoNCE 寫法，差別是這裡兩側是不同模態
    （圖/文字），文字側是凍結的參考向量，只有 region_vecs 這一側會收到梯度。"""
    sim = region_vecs @ text_vecs.t() / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def train(data_dir: Path, out_dir: Path = RUNS, epochs: int = 30, batch_size: int = 32,
          lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 5,
          temperature: float = 0.1, val_ratio: float = 0.2, crop_size: int = CROP_SIZE,
          text_encoder_dir: Path = TEXT_ENCODER_DIR, device: str | None = None):
    loaded = load_encoder(text_encoder_dir)
    if loaded is None:
        raise RuntimeError(
            f"文字 embedding 模型尚未訓練，請先執行 python embedding_model.py train"
            f"（checkpoint 應位於 {text_encoder_dir}）"
        )
    text_model, tokenizer, _max_len, _text_device = loaded
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    text_model.to(device)

    data_dir = Path(data_dir)
    pairs = [json.loads(line) for line in (data_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pairs) < 4:
        raise ValueError("需要至少 4 筆 (crop, caption) 樣本才能做對比學習（同批次內互為負例）")
    random.Random(0).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    with torch.no_grad():
        train_txt = torch.tensor(embed_texts([p["caption"] for p in train_pairs], text_model, tokenizer), device=device)
        val_txt = torch.tensor(embed_texts([p["caption"] for p in val_pairs], text_model, tokenizer), device=device)
    train_img = torch.stack([_load_crop_tensor(data_dir / p["crop"], crop_size) for p in train_pairs]).to(device)
    val_img = torch.stack([_load_crop_tensor(data_dir / p["crop"], crop_size) for p in val_pairs]).to(device)

    embed_dim = train_txt.shape[1]
    model = RegionEncoder(embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)
    stopper = EarlyStopper(patience)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_pairs), device=device)
        loss_sum, n_batches = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # InfoNCE 至少需要 2 筆才有負例
                continue
            optimizer.zero_grad()
            region_vecs = model(train_img[idx])
            loss = _info_nce(region_vecs, train_txt[idx], temperature)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = _info_nce(model(val_img), val_txt, temperature).item() if len(val_pairs) >= 2 else train_loss
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if stopper.step(val_loss, model, epoch):
            break
    stopper.restore_best(model)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "region_model.pt")
    (out_dir / "config.json").write_text(json.dumps({"embed_dim": embed_dim, "crop_size": crop_size}, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history


# --- 推論：候選框 -> 區域 embedding -> 跟文字查詢比對相似度 -------------------

_loaded_region_encoder: dict = {}


def load_region_encoder(out_dir: Path = RUNS, device: str | None = None):
    """回傳 (RegionEncoder, crop_size, device)；checkpoint 還沒訓練出來就回傳 None，
    跟 embedding_model.load_encoder() 同一套慣例。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(out_dir), device)
    if key in _loaded_region_encoder:
        return _loaded_region_encoder[key]

    out_dir = Path(out_dir)
    model_path, config_path = out_dir / "region_model.pt", out_dir / "config.json"
    if not (model_path.exists() and config_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = RegionEncoder(config["embed_dim"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loaded = (model, config["crop_size"], device)
    _loaded_region_encoder[key] = loaded
    return loaded


@dataclass
class ReferResult:
    label: str
    confidence: float
    box: list
    match_score: float
    warning: str


def refer(image: np.ndarray, query: str, yolo_weights: str, conf: float = 0.25,
          region_runs: Path = RUNS, text_encoder_dir: Path = TEXT_ENCODER_DIR) -> ReferResult | None:
    """給一張圖 + 一句話描述，回傳 YOLO 候選框裡最符合描述的那個；沒有任何候選框，
    或文字/區域編碼器還沒訓練好，回傳 None（不編造答案）。"""
    from detector import detect  # web/backend 共用推論入口，跟 yolo_custom.py 同一套做法，不開第二套

    detections = detect(image, conf=conf, weights=yolo_weights)
    if not detections:
        return None

    text_loaded = load_encoder(text_encoder_dir)
    region_loaded = load_region_encoder(region_runs)
    if text_loaded is None or region_loaded is None:
        return None
    text_model, tokenizer, _max_len, _device = text_loaded
    region_model, crop_size, device = region_loaded

    query_vec = torch.tensor(embed_texts([query], text_model, tokenizer)[0], device=device)
    crops = []
    for det in detections:
        x1, y1, x2, y2 = (max(int(v), 0) for v in det["box"])
        roi = image[y1:y2, x1:x2]
        crops.append(_load_crop_tensor_from_array(roi, crop_size))
    with torch.no_grad():
        region_vecs = region_model(torch.stack(crops).to(device))
        scores = (region_vecs @ query_vec).tolist()

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best, score = detections[best_idx], scores[best_idx]
    confidence = (score + 1) / 2  # cosine [-1, 1] -> [0, 1]
    return ReferResult(best["label"], best["confidence"], best["box"], round(confidence, 4),
                        low_confidence_warning(confidence))


def _load_crop_tensor_from_array(roi: np.ndarray, crop_size: int) -> torch.Tensor:
    if roi.size == 0:
        roi = np.zeros((crop_size, crop_size, 3), np.uint8)
    roi = cv2.resize(roi, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(roi[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0


def main():
    p = argparse.ArgumentParser(description="自製區域指稱模型（YOLO 偵測 + 自訓視覺/文字對比學習）")
    p.add_argument("--make-data", type=Path); p.add_argument("--n", type=int, default=300)
    p.add_argument("--train", type=Path); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=RUNS)
    p.add_argument("--refer", type=Path, help="要指稱的圖片")
    p.add_argument("--query", type=str, help="文字描述，跟 --refer 搭配使用")
    p.add_argument("--yolo-weights", type=str, help="YOLO 權重路徑，跟 --refer 搭配使用")
    a = p.parse_args()

    if a.make_data:
        print("資料集：", make_synthetic(a.make_data, a.n))
    if a.train:
        train(a.train, a.out_dir, epochs=a.epochs)
    if a.refer:
        if not (a.query and a.yolo_weights):
            p.error("--refer 需要搭配 --query 與 --yolo-weights")
        image = cv2.imread(str(a.refer))
        if image is None:
            raise ValueError(f"無法讀取圖片：{a.refer}")
        result = refer(image, a.query, a.yolo_weights, region_runs=a.out_dir)
        print(result if result else "找不到符合描述的候選框，或模型尚未訓練完成")


if __name__ == "__main__":
    main()
