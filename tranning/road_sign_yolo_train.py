"""Taiwan road-sign detector — training scaffold (YOLO 版本).

跟原本 `road_sign_train.py`（MobileNetV2 整張圖分類）並存、互不取代：那支假設
「一張照片只有一個路牌、已經裁好」；這支改用 YOLO 偵測，給一張畫面裡可能有
多個路牌、位置未知的真實場景（例如行車記錄器/路口攝影機畫面）用，輸出是每個
路牌的框 + 類別。

目前沒有任何標好框的路牌資料集，所以只定義訓練入口；用
test_road_sign_yolo_train.py 裡的小合成 YOLO 格式資料集煙煙測試管線本身能跑
完（矩形/圓形代表兩種路牌，不代表真實準確度）。

設計沿用 circuit_diagram_train.py 的思路（原因同該檔案/舊版 road_sign_train.py
docstring）：
  - COCO-pretrained yolo11n.pt 起點 + --freeze（預設凍結前 10 層）+ cos_lr，
    避免小資料集 underfitting。
  - fliplr=0 / flipud=0：路牌有方向性（箭頭、靠左/靠右不同標誌），鏡像會
    教錯類別，跟舊版分類器的理由一致，不開放覆寫。
  - summarize_overfitting() 讀 results.csv 印出 train/val loss 差距警告。
  - --detach（背景訓練）/ --status（查進度），跟 circuit_diagram_train.py
    /yolo_custom.py 同一套做法。

Usage:
    python road_sign_yolo_train.py --dataset-dir path/to/dataset --classes \
        stop yield no-entry keep-left keep-right speed-limit-50
    python road_sign_yolo_train.py --data path/to/data.yaml --epochs 50
    python road_sign_yolo_train.py --data path/to/data.yaml --detach
    python road_sign_yolo_train.py --status --out-dir road_sign_yolo_runs/train
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from circuit_diagram_train import build_data_yaml
from yolo_train_utils import summarize_overfitting

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "web" / "backend" / "yolo11n.pt"


def train(data: Path, out_dir: Path, epochs: int, batch: int, imgsz: int, patience: int,
          freeze: int, lr0: float, weight_decay: float, device, weights: Path):
    model = YOLO(str(weights))
    model.train(
        data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, patience=patience,
        freeze=freeze if freeze > 0 else None, lr0=lr0, weight_decay=weight_decay,
        cos_lr=True, fliplr=0.0, flipud=0.0, degrees=10.0,
        device=device, project=str(out_dir.parent), name=out_dir.name,
        exist_ok=True, verbose=False, plots=False,
    )
    save_dir = Path(model.trainer.save_dir)
    summary = summarize_overfitting(save_dir / "results.csv")
    if summary:
        print(summary)
    print(f"best weights: {model.trainer.best}")
    return model, save_dir


def detach_train(argv: list[str]) -> int:
    import subprocess
    root = Path(__file__).resolve().parent
    log_path = root / "road_sign_yolo_runs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0
    with open(log_path, "ab") as log:
        return subprocess.Popen([sys.executable, __file__, *[a for a in argv if a != "--detach"]],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 creationflags=flags, cwd=str(root.parent)).pid


def status(out_dir: Path) -> str:
    import csv

    import yaml
    csv_path, args_path = out_dir / "results.csv", out_dir / "args.yaml"
    if not csv_path.exists():
        return "尚無訓練紀錄（還沒開始，或第一個 epoch 還沒跑完）。"
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(csv_path, encoding="utf-8"))]
    total = yaml.safe_load(open(args_path, encoding="utf-8")).get("epochs", "?") if args_path.exists() else "?"
    last = rows[-1]
    return f"epoch {last['epoch']}/{total}  mAP50={float(last['metrics/mAP50(B)']):.3f}  mAP50-95={float(last['metrics/mAP50-95(B)']):.3f}"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a local YOLO road-sign detector")
    parser.add_argument("--data", type=Path, help="existing Ultralytics data.yaml")
    parser.add_argument("--dataset-dir", type=Path, help="dataset root with images/{train,val}, labels/{train,val}")
    parser.add_argument("--classes", nargs="+", help="class names, index = label id (used with --dataset-dir)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=Path("road_sign_yolo_runs") / "train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N backbone layers, 0 = unfreeze all")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(status(args.out_dir))
        return

    if args.data:
        data_path = args.data
    elif args.dataset_dir and args.classes:
        data_path = build_data_yaml(args.dataset_dir, args.classes)
    else:
        parser.error("pass either --data <data.yaml> or both --dataset-dir and --classes")

    if args.detach:
        print(f"已在背景啟動（PID {detach_train(sys.argv[1:])}），用 --status 查看進度")
        return

    train(data_path, args.out_dir, args.epochs, args.batch, args.imgsz, args.patience,
          args.freeze, args.lr0, args.weight_decay, args.device, args.weights)


if __name__ == "__main__":
    main()
