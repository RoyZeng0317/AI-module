"""Tests for neural_network.py.

Two different claims are checked, on purpose:
  1. test_backward_computes_correct_chain_rule_gradients — the actual math
     claim. torch.autograd.gradcheck compares loss.backward()'s analytical
     gradient against an independent finite-difference estimate, so this
     proves the chain rule through Linear->ReLU->Linear->ReLU->Linear->MSE
     is computed correctly, not just that the code runs.
  2. test_training_loop_runs_end_to_end — the pipeline smoke test used
     throughout tranning/ (no accuracy claim; there is no real regression
     dataset yet), using a tiny synthetic CSV.
"""

import csv
import json

import torch
import torch.nn as nn

from neural_network import build_model, train


def test_backward_computes_correct_chain_rule_gradients():
    torch.manual_seed(0)
    model = build_model(in_features=4).double()
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, 1, dtype=torch.float64)

    def loss_fn(inp):
        return nn.functional.mse_loss(model(inp), target)

    assert torch.autograd.gradcheck(loss_fn, (x,), eps=1e-6, atol=1e-4)


def _make_synthetic_csv(path, n=40, seed=0):
    generator = torch.Generator().manual_seed(seed)
    f1 = torch.randn(n, generator=generator)
    f2 = torch.randn(n, generator=generator)
    f3 = torch.randn(n, generator=generator)
    noise = torch.randn(n, generator=generator) * 0.01
    y = 2 * f1 - f2 + 0.5 * f3 + noise
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "f3", "y"])
        for row in zip(f1.tolist(), f2.tolist(), f3.tolist(), y.tolist()):
            writer.writerow(row)


def test_training_loop_runs_end_to_end(tmp_path):
    data_path = tmp_path / "data.csv"
    out_dir = tmp_path / "runs"
    _make_synthetic_csv(data_path)

    model, history = train(
        data=data_path, target="y", out_dir=out_dir, epochs=5, lr=1e-2,
        val_ratio=0.25, patience=10, weight_decay=1e-4,
    )

    assert len(history) == 5
    assert (out_dir / "best_model.pt").exists()
    assert all(h["grad_norm"] > 0 for h in history)

    stats = json.loads((out_dir / "stats.json").read_text())
    assert len(stats["x_mean"][0]) == 3
