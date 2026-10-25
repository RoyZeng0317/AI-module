"""Pipeline smoke test for materials_yolo_train.py — no real resistor/capacitor
bounding-box dataset exists yet, so this only proves the training loop runs
end-to-end on a tiny synthetic YOLO-format dataset (same convention as
test_circuit_diagram_train.py).
"""

from pathlib import Path

import cv2
import numpy as np

from circuit_diagram_train import build_data_yaml
from materials_yolo_train import DEFAULT_WEIGHTS, status, train

CLASSES = ["resistor", "capacitor"]
IMG_SIZE = 64
BOX = (16, 16, 48, 48)


def _write_label(path: Path, class_id: int):
    x1, y1, x2, y2 = BOX
    cx, cy = (x1 + x2) / 2 / IMG_SIZE, (y1 + y2) / 2 / IMG_SIZE
    w, h = (x2 - x1) / IMG_SIZE, (y2 - y1) / IMG_SIZE
    path.write_text(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")


def _make_synthetic_yolo_dataset(root: Path, n_train: int = 6, n_val: int = 2):
    x1, y1, x2, y2 = BOX
    for split, count in (("train", n_train), ("val", n_val)):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True); lbl_dir.mkdir(parents=True)
        for class_id, class_name in enumerate(CLASSES):
            for i in range(count):
                img = np.full((IMG_SIZE, IMG_SIZE, 3), 40, dtype=np.uint8)
                if class_id == 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 30), -1)
                else:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, (30, 200, 200), -1)
                name = f"{class_name}_{i}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
                _write_label(lbl_dir / f"{name}.txt", class_id)


def test_training_loop_runs_end_to_end(tmp_path):
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "runs" / "train"

    _make_synthetic_yolo_dataset(dataset_dir)
    data_yaml = build_data_yaml(dataset_dir, CLASSES)

    model, save_dir = train(
        data=data_yaml, out_dir=out_dir, epochs=1, batch=4, imgsz=IMG_SIZE, patience=1,
        freeze=0, lr0=1e-3, weight_decay=5e-4, device="cpu", weights=DEFAULT_WEIGHTS,
    )

    assert save_dir.exists()
    assert (save_dir / "results.csv").exists()
    assert model.trainer.best.exists()
    assert "epoch 1/1" in status(save_dir)


def test_status_before_training_starts(tmp_path):
    assert "尚無訓練紀錄" in status(tmp_path / "no_such_dir")
