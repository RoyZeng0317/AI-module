"""Circuit-diagram component detector — training scaffold.

Fine-tunes a local YOLOv8 model (Ultralytics, same library backend/detector.py
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
    - Starts from COCO-pretrained yolov8n.pt (transfer learning) instead of
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

Usage:
    # dataset already laid out as <dir>/images/{train,val}, labels/{train,val}
    python circuit_diagram_train.py --dataset-dir path/to/dataset --classes \
        resistor capacitor inductor diode ic transistor battery switch led wire_junction

    # or point straight at an existing Ultralytics data.yaml
    python circuit_diagram_train.py --data path/to/data.yaml --epochs 50 --freeze 0
"""

import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "backend" / "yolov8n.pt"


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


def summarize_overfitting(results_csv: Path) -> None:
    """Print a final train/val loss-gap check from Ultralytics' own
    results.csv (always written after training) — no fragile mid-training
    callback needed since Ultralytics already logs every epoch itself.
    """
    if not results_csv.exists():
        return
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}

    def _loss_sum(prefix: str) -> float:
        return sum(float(last[k]) for k in last if k.startswith(prefix) and k.endswith("_loss"))

    train_loss, val_loss = _loss_sum("train/"), _loss_sum("val/")
    map_key = next((k for k in last if k.startswith("metrics/mAP50-95")), None)
    map50_95 = float(last[map_key]) if map_key else None

    summary = f"final epoch: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
    if map50_95 is not None:
        summary += f" mAP50-95={map50_95:.3f}"
    print(summary)

    if train_loss > 0 and val_loss > train_loss * 1.5:
        print("  [warning: val loss well above train loss — possible overfitting; "
              "try a higher --freeze or more augmentation]")
    if map50_95 is not None and map50_95 < 0.1:
        print("  [warning: mAP50-95 still very low — possible underfitting; "
              "try --freeze 0 (unfreeze backbone) or more epochs]")


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
    summarize_overfitting(save_dir / "results.csv")
    print(f"best weights: {model.trainer.best}")
    return model, save_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a local YOLO circuit-component detector")
    parser.add_argument("--data", type=Path, help="existing Ultralytics data.yaml")
    parser.add_argument("--dataset-dir", type=Path,
                        help="dataset root with images/{train,val}, labels/{train,val}")
    parser.add_argument("--classes", nargs="+", help="class names, index = label id (used with --dataset-dir)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="starting weights (default: backend/yolov8n.pt, COCO-pretrained)")
    parser.add_argument("--out-dir", type=Path, default=Path("circuit_diagram_runs") / "train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N backbone layers, 0 = unfreeze all")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None, help="'0' for first CUDA GPU, 'cpu', or leave unset to auto-pick")
    args = parser.parse_args()

    if args.data:
        data_path = args.data
    elif args.dataset_dir and args.classes:
        data_path = build_data_yaml(args.dataset_dir, args.classes)
    else:
        parser.error("pass either --data <data.yaml> or both --dataset-dir and --classes")

    train(data_path, args.out_dir, args.epochs, args.batch, args.imgsz, args.patience,
          args.freeze, args.lr0, args.weight_decay, args.device, args.weights)


if __name__ == "__main__":
    main()
