"""本地物品 YOLO 微調入口；預設 batch=4、workers=0，輸出獨立模型目錄。"""
import argparse, hashlib, sys, uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tranning"), str(ROOT / "web" / "backend")]
from object_labeling import project_path, read_labels


def validate_data(path):
    import yaml
    from PIL import Image
    path = project_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names", {})
    if not isinstance(names, dict) or list(names) != list(range(len(names))) or not names or not all(isinstance(v, str) and v.strip() for v in names.values()):
        raise ValueError("data.yaml 需要從 0 起的連續類別 ID。")
    folder, seen = project_path(config["path"]), set()
    for split in ("train", "val"):
        images = project_path(folder / config[split])
        counts = [0] * len(names)
        files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if images.is_dir() else []
        if not files:
            raise ValueError(f"{split} 沒有圖片。")
        for file in files:
            file = project_path(file)
            with Image.open(file) as image:
                image = image.convert("RGB")
                digest = hashlib.sha256(f"{image.width}x{image.height}:".encode()+image.tobytes()).hexdigest()
                if digest in seen:
                    raise ValueError("資料集包含重複影像，請移除以免 train/val 洩漏。")
                seen.add(digest)
                label = project_path(folder / "labels" / split / f"{file.stem}.txt")
                if not label.is_file():
                    raise ValueError(f"缺少標註：{label.name}")
                boxes = read_labels(label.read_text(encoding="utf-8"), image.width, image.height, len(names))
                for box in boxes:
                    counts[box[0]] += 1
        if not any(counts):
            raise ValueError(f"{split} 沒有物品正例。")
        print(f"{split} 標註框數：" + "、".join(f"{names[i]}={count}" for i, count in enumerate(counts)))
        missing = [names[i] for i, count in enumerate(counts) if count == 0]
        if missing:
            purpose = "訓練" if split == "train" else "驗證"
            print(f"警告：{split} 以下類別沒有{purpose}框：{'、'.join(missing)}；不能據此確認這些類別的辨識品質。")
    return path


def train(data, weights=ROOT / "yolo11n.pt", epochs=150, batch=4, imgsz=640, out_dir=None):
    data, weights = validate_data(data), project_path(weights)
    if not weights.is_file():
        raise ValueError("起始權重不存在，請選擇專案內的本地 .pt。")
    if min(epochs, batch, imgsz) <= 0:
        raise ValueError("epochs、batch、imgsz 必須大於零。")
    out_dir = project_path(out_dir or ROOT / "tranning" / "object_detection_runs" / uuid.uuid4().hex)
    if out_dir.exists():
        raise ValueError("輸出目錄已存在，請選擇新目錄以保留舊模型。")
    from ultralytics import YOLO
    from detector import _device
    from yolo_train_utils import summarize_overfitting
    model = YOLO(str(weights))
    # 8.4.90 的 check_det_dataset 即使 plots=False 仍下載字型。
    # 此入口不繪製訓練圖；僅暫停該字型檢查，完整保留資料驗證，離開時還原。
    with patch("ultralytics.data.utils.check_font", return_value=None):
        model.train(data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, workers=0, device=_device(),
                    project=str(out_dir.parent), name=out_dir.name, exist_ok=False, plots=False,
                    nbs=batch, optimizer="AdamW", lr0=0.001, warmup_epochs=0, cos_lr=True,
                    freeze=10, degrees=15, mosaic=0.2, close_mosaic=10, scale=0.3)
    actual = Path(model.trainer.save_dir)
    summary = summarize_overfitting(actual / "results.csv")
    if summary:
        print(summary)
    best = actual / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("訓練未產生 best.pt，請檢查訓練紀錄。")
    print(f"自訂物品模型：{best}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(args.data, args.weights, args.epochs, args.batch, args.imgsz, args.out_dir)


if __name__ == "__main__":
    main()
