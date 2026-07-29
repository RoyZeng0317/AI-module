"""Pipeline smoke test for image_classifier_bnn.py.

Does NOT claim any recognition accuracy — there is no real dataset yet.
Only proves the training loop (dataloaders, model, optimizer, scheduler,
early stopping, checkpoint + classes.json/config.json output) and the
Bayesian (MC Dropout) classify_image() inference path run end-to-end
without crashing, using a tiny synthetic image dataset produced via
data_split.py (same fixture style as test_road_sign_train.py).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_split import split_dataset
from image_classifier_bnn import classify_image, train


def _make_synthetic_dataset(root: Path):
    classes = {
        "cat": (200, 30, 30),
        "dog": (30, 30, 200),
    }
    for name, color in classes.items():
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for i in range(12):
            arr = np.full((64, 64, 3), color, dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"{name}_{i}.jpg")


def test_training_loop_runs_end_to_end(tmp_path):
    source = tmp_path / "source"
    split_dir = tmp_path / "split"
    out_dir = tmp_path / "runs"

    _make_synthetic_dataset(source)
    split_dataset(source, split_dir, train=0.7, val=0.15, test=0.15, seed=1)

    model, classes, history = train(
        data_dir=split_dir, out_dir=out_dir, epochs=2, batch_size=4, lr=1e-3,
        patience=5, image_size=96, freeze_backbone=True, dropout=0.3, weight_decay=1e-4,
    )

    assert classes == sorted(["cat", "dog"])
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "classes.json").read_text()) == classes
    assert json.loads((out_dir / "config.json").read_text())["dropout"] == 0.3
    assert all(0.0 <= h["val_acc"] <= 1.0 for h in history)


def test_classify_image_after_training_returns_bayesian_stats(tmp_path):
    source = tmp_path / "source"
    split_dir = tmp_path / "split"
    out_dir = tmp_path / "runs"

    _make_synthetic_dataset(source)
    split_dataset(source, split_dir, train=0.7, val=0.15, test=0.15, seed=1)
    train(
        data_dir=split_dir, out_dir=out_dir, epochs=1, batch_size=4, lr=1e-3,
        patience=5, image_size=96, freeze_backbone=True, dropout=0.3, weight_decay=1e-4,
    )

    sample_image = next((split_dir / "train" / "cat").glob("*.jpg"))
    label, confidence, entropy = classify_image(sample_image, out_dir=out_dir, mc_samples=5)

    assert label in {"cat", "dog"}
    assert 0.0 <= confidence <= 1.0
    assert entropy >= 0.0


def test_classify_image_without_checkpoint_returns_placeholder(tmp_path):
    label, confidence, entropy = classify_image(
        Path("does_not_matter.jpg"), out_dir=tmp_path / "no_checkpoint_here"
    )
    assert "尚未訓練" in label
    assert confidence == 0.0
    assert entropy == 0.0
