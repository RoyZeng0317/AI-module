"""Shared Ultralytics YOLO 訓練診斷工具，給 yolo_custom.py / circuit_diagram_train.py /
之後新增的 YOLO 訓練腳本共用，避免同一段 results.csv 判讀邏輯複製多份。"""
import csv
from pathlib import Path


def summarize_overfitting(results_csv: Path, loss_ratio_warn: float = 1.5, map_warn: float = 0.1) -> str | None:
    """讀 Ultralytics 的 results.csv，回傳最後一個 epoch 的 train/val loss 與
    mAP50-95 摘要字串（含 overfitting/underfitting 警告）；檔案不存在或無資料回傳 None。"""
    if not results_csv.exists():
        return None
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}

    def _loss_sum(prefix: str) -> float:
        return sum(float(last[k]) for k in last if k.startswith(prefix) and k.endswith("_loss"))

    train_loss, val_loss = _loss_sum("train/"), _loss_sum("val/")
    map_key = next((k for k in last if k.startswith("metrics/mAP50-95")), None)
    map50_95 = float(last[map_key]) if map_key else None

    lines = [f"final epoch: train_loss={train_loss:.4f} val_loss={val_loss:.4f}" +
             (f" mAP50-95={map50_95:.3f}" if map50_95 is not None else "")]
    if train_loss > 0 and val_loss > train_loss * loss_ratio_warn:
        lines.append("  [warning: val loss well above train loss — possible overfitting; "
                      "try more augmentation or freezing more layers]")
    if map50_95 is not None and map50_95 < map_warn:
        lines.append("  [warning: mAP50-95 still very low — possible underfitting; "
                      "try unfreezing the backbone or more epochs]")
    return "\n".join(lines)
