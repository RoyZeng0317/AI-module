"""PCB 缺陷偵測 — training scaffold.

Fine-tunes a local YOLO11 model (Ultralytics，同 detector.py / circuit_diagram_train.py
用的那套) 直接在完整 PCB 照片上框出缺陷位置（missing_hole/mouse_bite/open_circuit/
short/spur/spurious_copper，PKU-Market-PCB 的 6 類），不是先裁圖再分類——跟原本
`pcb_dataset_prep.py --out` + `image_classifier_bnn.py` 的分類路徑並存、互不影響。

資料準備：
    python pcb_dataset_prep.py --images <圖片資料夾> --annotations <XML資料夾> --yolo-out data/pcb_yolo
    python pcb_defect_train.py --data data/pcb_yolo/data.yaml

跟 circuit_diagram_train.py 不同的增強選擇：PCB 缺陷（刮痕/缺洞/短路等）本身沒有
方向性（鏡像後還是同一種缺陷），所以保留 Ultralytics 預設的 fliplr/flipud，
不像電路圖符號那樣關閉翻轉。其餘設計（COCO-pretrained 起點、--freeze、cos_lr、
summarize_overfitting()、--detach/--status）都跟 circuit_diagram_train.py 一致，
只是換了預設類別清單；沿用同一套 test_*.py 合成資料煙霧測試慣例（見
test_pcb_defect_train.py，不代表真實準確度）。

Usage:
    python pcb_defect_train.py --data data/pcb_yolo/data.yaml --epochs 50
    python pcb_defect_train.py --data data/pcb_yolo/data.yaml --detach
    python pcb_defect_train.py --status --out-dir pcb_defect_runs/train
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from circuit_diagram_train import build_data_yaml
from yolo_train_utils import summarize_overfitting

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "web" / "backend" / "yolo11n.pt"
DEFAULT_CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def train(data: Path, out_dir: Path, epochs: int, batch: int, imgsz: int, patience: int,
          freeze: int, lr0: float, weight_decay: float, device, weights: Path,
          mosaic: float = 1.0, fliplr: float = 0.5, flipud: float = 0.0):
    model = YOLO(str(weights))
    model.train(
        data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, patience=patience,
        freeze=freeze if freeze > 0 else None, lr0=lr0, weight_decay=weight_decay,
        cos_lr=True, mosaic=mosaic, fliplr=fliplr, flipud=flipud,
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
    """背景訓練，跟 circuit_diagram_train.py/yolo_custom.py 同一套做法。"""
    import subprocess
    root = Path(__file__).resolve().parent
    log_path = root / "pcb_defect_runs" / "train.log"
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
    parser = argparse.ArgumentParser(description="Fine-tune a local YOLO PCB defect detector")
    parser.add_argument("--data", type=Path, help="existing Ultralytics data.yaml")
    parser.add_argument("--dataset-dir", type=Path, help="dataset root with images/{train,val}, labels/{train,val}")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help=f"class names, index = label id (used with --dataset-dir；預設 {DEFAULT_CLASSES})")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=Path("pcb_defect_runs") / "train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N backbone layers, 0 = unfreeze all")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(status(args.out_dir))
        return

    if args.data:
        data_path = args.data
    elif args.dataset_dir:
        data_path = build_data_yaml(args.dataset_dir, args.classes)
    else:
        parser.error("pass either --data <data.yaml> or --dataset-dir (with --classes defaulting to PKU-Market-PCB's 6)")

    if args.detach:
        print(f"已在背景啟動（PID {detach_train(sys.argv[1:])}），用 --status 查看進度")
        return

    train(data_path, args.out_dir, args.epochs, args.batch, args.imgsz, args.patience,
          args.freeze, args.lr0, args.weight_decay, args.device, args.weights,
          args.mosaic, args.fliplr, args.flipud)


if __name__ == "__main__":
    main()
