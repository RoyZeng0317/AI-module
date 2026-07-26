"""Unit tests for train_utils.py — the shared early-stopping / scheduler /
checkpoint / warning helpers pulled out of road_sign_train.py, OCR.py, and
speech_to_text.py. These test the helpers in isolation (not through a full
training run — that's already covered by each script's own smoke test).
"""

import json

import torch
import torch.nn as nn

from train_utils import (
    EarlyStopper,
    accuracy_gap_warning,
    loss_gap_warning,
    plateau_scheduler,
    save_checkpoint,
)


def _tiny_model():
    return nn.Linear(2, 2)


def test_early_stopper_tracks_best_and_signals_stop():
    model = _tiny_model()
    stopper = EarlyStopper(patience=2)

    assert stopper.step(1.0, model, epoch=1) is False
    assert stopper.best_loss == 1.0
    assert stopper.step(0.5, model, epoch=2) is False
    assert stopper.best_loss == 0.5
    assert stopper.step(0.6, model, epoch=3) is False  # 1st epoch without improvement
    assert stopper.step(0.7, model, epoch=4) is True    # 2nd epoch without improvement -> patience hit


def test_early_stopper_restores_best_not_last():
    model = _tiny_model()
    stopper = EarlyStopper(patience=5)
    stopper.step(0.5, model, epoch=1)
    best_weight = model.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(1.0)  # simulate further (worse) training
    stopper.step(0.9, model, epoch=2)

    stopper.restore_best(model)
    assert torch.equal(model.weight, best_weight)


def test_plateau_scheduler_uses_shared_hyperparameters():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = plateau_scheduler(optimizer)

    assert scheduler.mode == "min"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 2


def test_save_checkpoint_writes_model_and_history(tmp_path):
    model = _tiny_model()
    history = [{"epoch": 1, "train_loss": 1.0}]

    save_checkpoint(model, tmp_path, history)

    assert (tmp_path / "best_model.pt").exists()
    assert json.loads((tmp_path / "history.json").read_text()) == history


def test_loss_gap_warning_overfitting_and_underfitting():
    assert "overfitting" in loss_gap_warning(train_loss=1.0, val_loss=2.0, epoch=5, epochs=10)
    assert "underfitting" in loss_gap_warning(train_loss=4.0, val_loss=4.1, epoch=5, epochs=10)
    assert loss_gap_warning(train_loss=0.5, val_loss=0.6, epoch=5, epochs=10) == ""


def test_accuracy_gap_warning_overfitting_and_underfitting():
    overfit = accuracy_gap_warning(train_acc=0.9, val_acc=0.5, epoch=5, epochs=10)
    assert "overfitting" in overfit

    underfit = accuracy_gap_warning(train_acc=0.2, val_acc=0.2, epoch=5, epochs=10, underfit_hint="--unfreeze-backbone")
    assert "underfitting" in underfit
    assert "--unfreeze-backbone" in underfit

    assert accuracy_gap_warning(train_acc=0.8, val_acc=0.75, epoch=5, epochs=10) == ""
