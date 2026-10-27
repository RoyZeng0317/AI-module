"""Circuit-diagram component detector — training scaffold.

Fine-tunes a local YOLO11 model (Ultralytics, same library backend/detector.py
already uses) to detect circuit-schematic components (resistor, capacitor,
IC, diode, ...) so 視覺電路圖.py can invoke it directly on live camera frames
instead of on screenshots/recordings.

No circuit-component dataset exists yet, so this only defines the training
entry point; it is smoke-tested against a tiny synthetic YOLO-format dataset
in test_circuit_diagram_train.py (no accuracy claim — just "the training
pipeline runs end to end").

Design choices made to avoid overfitting AND underfitting on what will
likely be a small circuit-component dataset (same reasoning as
road_sign_train.py, adapted to detection):

  Underfitting:
    - Starts from COCO-pretrained yolo11n.pt (transfer learning) instead of
      random init — the backbone already has generic edge/corner features
      useful for line-drawn schematic symbols.
    - cos_lr cosine schedule instead of a fixed step decay.

  Overfitting:
    - --freeze N freezes the first N backbone layers (default 10) so a
      small dataset only has to fit the detection head; --freeze 0
      unfreezes everything if train accuracy stays low instead.
    - fliplr=0 / flipud=0: schematic symbols are direction-sensitive (a
      mirrored diode/transistor means a different component), same
      reasoning road_sign_train.py uses for road signs. Rotation/HSV
      jitter only.
    - `patience` early stopping (Ultralytics built-in) keeps the
      best-epoch weights once val loss stalls.
    - summarize_overfitting() reads Ultralytics' own results.csv after
      training and prints the final train/val loss gap so a human can
      catch overfitting even when patience hasn't tripped yet.

Recognizing a specific chip (e.g. "atmega328p") by sight, not just the
generic "ic" class, would need real labeled photos of that exact part --
none exist yet, same honest gap as the rest of this class list. Until then,
a specific-package class like "ic_dip28" (any DIP-28 chip, package shape
only, not the exact part) is the closest this detector can promise; pair it
with circuit_ic_generator.py/ic_library.py's known pinouts for whatever a
human confirms the detected chip actually is, rather than pretending the
detector itself can read a part number off a photo.

Usage:
    # dataset already laid out as <dir>/images/{train,val}, labels/{train,val}
    python circuit_diagram_train.py --dataset-dir path/to/dataset --classes \
        resistor capacitor inductor diode ic ic_dip28 transistor battery switch led wire_junction

    # or point straight at an existing Ultralytics data.yaml
    python circuit_diagram_train.py --data path/to/data.yaml --epochs 50 --freeze 0

    # 背景訓練（關掉終端機不中斷）+ 查進度，跟 yolo_custom.py 同一套做法
    python circuit_diagram_train.py --data path/to/data.yaml --detach
    python circuit_diagram_train.py --status --out-dir circuit_diagram_runs/train
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from yolo_train_utils import summarize_overfitting

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "web" / "backend" / "yolo11n.pt"


def build_data_yaml(dataset_dir: Path, classes: list) -> Path:
    """Write an Ultralytics data.yaml for a dataset already laid out as
    <dataset_dir>/images/{train,val} and <dataset_dir>/labels/{train,val}
    (Ultralytics finds labels by mirroring the images/ path into labels/).
    """
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"expected {img_dir} (YOLO layout: images/{split}, labels/{split})"
            )

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: name for i, name in enumerate(classes)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


def train(data: Path, out_dir: Path, epochs: int, batch: int, imgsz: int, patience: int,
          freeze: int, lr0: float, weight_decay: float, device, weights: Path,
          mosaic: float = 1.0, copy_paste: float = 0.0):
    model = YOLO(str(weights))
    model.train(
        data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, patience=patience,
        freeze=freeze if freeze > 0 else None, lr0=lr0, weight_decay=weight_decay,
        cos_lr=True, fliplr=0.0, flipud=0.0, degrees=10.0, mosaic=mosaic, copy_paste=copy_paste,
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
    """把同一條訓練指令丟到獨立子程序（關掉 VSCode／終端機也不會中斷），跟
    yolo_custom.py 的 detach_train() 同一套做法，輸出寫 <out-dir>/../train.log。"""
    import subprocess
    import sys
    root = Path(__file__).resolve().parent
    log_path = root / "circuit_diagram_runs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(log_path, "ab") as log:
        return subprocess.Popen([sys.executable, __file__, *[a for a in argv if a != "--detach"]],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 creationflags=flags, cwd=str(root.parent)).pid


def status(out_dir: Path) -> str:
    """讀 <out_dir>/results.csv + args.yaml 回報目前 epoch / 總 epoch / 最新 mAP50，
    跟 yolo_custom.py 的 status() 同一套格式，方便 train_gui.py 用同一種方式顯示進度。"""
    import csv

    import yaml as _yaml
    csv_path, args_path = out_dir / "results.csv", out_dir / "args.yaml"
    if not csv_path.exists():
        return "尚無訓練紀錄（還沒開始，或第一個 epoch 還沒跑完）。"
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(csv_path, encoding="utf-8"))]
    total = _yaml.safe_load(open(args_path, encoding="utf-8")).get("epochs", "?") if args_path.exists() else "?"
    last = rows[-1]
    return f"epoch {last['epoch']}/{total}  mAP50={float(last['metrics/mAP50(B)']):.3f}  mAP50-95={float(last['metrics/mAP50-95(B)']):.3f}"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a local YOLO circuit-component detector")
    parser.add_argument("--data", type=Path, help="existing Ultralytics data.yaml")
    parser.add_argument("--dataset-dir", type=Path,
                        help="dataset root with images/{train,val}, labels/{train,val}")
    parser.add_argument("--classes", nargs="+", help="class names, index = label id (used with --dataset-dir)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="starting weights (default: backend/yolo11n.pt, COCO-pretrained)")
    parser.add_argument("--out-dir", type=Path, default=Path("circuit_diagram_runs") / "train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N backbone layers, 0 = unfreeze all")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None, help="'0' for first CUDA GPU, 'cpu', or leave unset to auto-pick")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="copy-paste augmentation probability")
    parser.add_argument("--detach", action="store_true", help="背景執行訓練（log: <out-dir>/../train.log）")
    parser.add_argument("--status", action="store_true", help="查看訓練進度，不開始新訓練")
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
        import sys
        print(f"已在背景啟動（PID {detach_train(sys.argv[1:])}），用 --status 查看進度")
        return

    train(data_path, args.out_dir, args.epochs, args.batch, args.imgsz, args.patience,
          args.freeze, args.lr0, args.weight_decay, args.device, args.weights,
          args.mosaic, args.copy_paste)


if __name__ == "__main__":
    main()
