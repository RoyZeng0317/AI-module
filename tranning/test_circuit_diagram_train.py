"""Pipeline smoke test for circuit_diagram_train.py.

This does NOT claim any detection accuracy — there is no real circuit-
component dataset yet. It only proves the training loop itself (data.yaml
generation, Ultralytics YOLO training, checkpoint + results.csv output,
overfitting summary) runs end-to-end without crashing, using a tiny
synthetic YOLO-format dataset (rectangles/circles standing in for
components at known bounding boxes).
"""

from pathlib import Path

import cv2
import numpy as np

from circuit_diagram_train import DEFAULT_WEIGHTS, build_data_yaml, train

CLASSES = ["component_a", "component_b"]
IMG_SIZE = 64
BOX = (16, 16, 48, 48)  # x1, y1, x2, y2 — same spot in every synthetic image


def _write_label(path: Path, class_id: int):
    x1, y1, x2, y2 = BOX
    cx, cy = (x1 + x2) / 2 / IMG_SIZE, (y1 + y2) / 2 / IMG_SIZE
    w, h = (x2 - x1) / IMG_SIZE, (y2 - y1) / IMG_SIZE
    path.write_text(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")


def _make_synthetic_yolo_dataset(root: Path, n_train: int = 6, n_val: int = 2):
    x1, y1, x2, y2 = BOX
    for split, count in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
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
