"""Pipeline smoke test for road_sign_train.py.

This does NOT claim any recognition accuracy — there is no real road-sign
dataset yet (CLAUDE.md to-do #4 is waiting on the user to provide one). It
only proves the training loop itself (dataloaders, model, optimizer,
scheduler, early stopping, checkpoint + classes.json output) runs
end-to-end without crashing, using a tiny synthetic image dataset produced
via data_split.py.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_split import split_dataset
from road_sign_train import train


def _make_synthetic_dataset(root: Path):
    # Distinct colors stand in for distinct sign classes — content doesn't
    # matter, only that the shapes/labels flow through the pipeline intact.
    classes = {
        "stop_sign": (200, 30, 30),
        "speed_limit_50": (30, 30, 200),
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

    assert classes == sorted(["stop_sign", "speed_limit_50"])
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "classes.json").read_text()) == classes
    assert all(0.0 <= h["val_acc"] <= 1.0 for h in history)
