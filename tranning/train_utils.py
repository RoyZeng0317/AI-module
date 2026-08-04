"""Shared training-loop building blocks for tranning/'s from-scratch models.

Extracted after road_sign_train.py, OCR.py, and speech_to_text.py each grew
a byte-for-byte identical early-stopping / best-checkpoint / history block
around three otherwise-different train() functions. Only what was actually
duplicated verbatim moved here — circuit_diagram_train.py is untouched
because Ultralytics owns its epoch loop end to end; there is nothing to
share with it beyond ideas already documented in its own module docstring.
"""

import json
from pathlib import Path

import torch


class EarlyStopper:
    """Validation-loss early stopping with best-checkpoint tracking.

    Same "patience epochs with no val-loss improvement" rule
    road_sign_train.py, OCR.py and speech_to_text.py each hand-rolled inline
    (best_val_loss / best_state / epochs_without_improvement). Keeps the
    best state_dict cloned to CPU rather than only the last epoch's
    weights, so a caller can restore_best() after the loop even if training
    kept going past the true optimum before patience ran out.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self.epochs_without_improvement = 0

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Call once per epoch after computing val_loss. Returns True when
        the caller should stop training now.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {self.patience} epochs)")
            return True
        return False

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def plateau_scheduler(optimizer: torch.optim.Optimizer, factor: float = 0.5, patience: int = 2):
    """Same ReduceLROnPlateau settings road_sign_train.py / OCR.py /
    speech_to_text.py each independently constructed — backs the learning
    rate off only once validation loss actually stalls, instead of decaying
    on a fixed schedule that might cut it too early (underfitting) or too
    late (overfitting).
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)


def save_checkpoint(model: torch.nn.Module, out_dir: Path, history: list[dict]) -> None:
    """Writes best_model.pt + history.json — the same two files all three
    manual-loop scripts wrote at the end of train()."""
    out_dir = Path(out_dir)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


def loss_gap_warning(train_loss: float, val_loss: float, epoch: int, epochs: int,
                      overfit_ratio: float = 1.5, loss_ceiling: float = 5.0,
                      underfit_floor: float = 3.0) -> str:
    """The exact heuristic OCR.py and speech_to_text.py each hard-coded
    inline: val loss well above train loss (while train loss is still low
    enough that the ratio is meaningful, not just early-epoch noise) hints
    at overfitting; train loss still high a third of the way through hints
    at underfitting.
    """
    if val_loss > train_loss * overfit_ratio and train_loss < loss_ceiling:
        return "  [warning: val loss well above train loss — possible overfitting]"
    if train_loss > underfit_floor and epoch >= max(3, epochs // 3):
        return "  [warning: train loss still high this far in — possible underfitting]"
    return ""


def accuracy_gap_warning(train_acc: float, val_acc: float, epoch: int, epochs: int,
                          gap_threshold: float = 0.15, underfit_acc: float = 0.4,
                          underfit_hint: str | None = None) -> str:
    """The exact heuristic road_sign_train.py hard-coded inline for
    classification tasks (accuracy-based rather than loss-based).
    `underfit_hint` lets a caller append a task-specific suggestion (e.g.
    road_sign_train.py's "--unfreeze-backbone") without this module knowing
    about that flag.
    """
    gap = train_acc - val_acc
    if gap > gap_threshold:
        return "  [warning: train/val accuracy gap suggests overfitting]"
    if train_acc < underfit_acc and epoch >= max(3, epochs // 3):
        hint = f"; try {underfit_hint}" if underfit_hint else ""
        return f"  [warning: low train accuracy this far in — possible underfitting{hint}]"
    return ""
