"""Unit tests for bayesian_utils.py's MC Dropout building blocks."""

import math

import torch
import torch.nn as nn

from bayesian_utils import (
    majority_vote,
    mc_dropout_mode,
    mc_predict_classification,
    predictive_entropy,
)
from bayesian_utils import low_confidence_warning


def test_mc_dropout_mode_leaves_dropout_active_but_batchnorm_frozen():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.9))
    model.eval()

    with mc_dropout_mode(model):
        assert model[1].training is False  # BatchNorm must stay frozen
        assert model[2].training is True  # Dropout must be stochastic

    # mode is restored afterwards
    assert model.training is False
    for m in model.modules():
        assert m.training is False


def test_mc_dropout_mode_restores_previous_training_flag():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    model.train()

    with mc_dropout_mode(model):
        pass

    assert model.training is True


def test_mc_dropout_mode_batchnorm_running_stats_untouched():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.9))
    model.eval()
    running_mean_before = model[1].running_mean.clone()

    with mc_dropout_mode(model), torch.no_grad():
        for _ in range(10):
            model(torch.randn(8, 4))

    assert torch.equal(model[1].running_mean, running_mean_before)


def test_mc_predict_classification_shapes_and_normalization():
    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    inputs = torch.randn(2, 4)

    mean_probs, std_probs = mc_predict_classification(model, inputs, n_samples=15)

    assert mean_probs.shape == (2, 3)
    assert std_probs.shape == (2, 3)
    assert torch.allclose(mean_probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (std_probs >= 0).all()


def test_predictive_entropy_peaked_lower_than_uniform():
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    peaked_entropy = predictive_entropy(peaked).item()
    uniform_entropy = predictive_entropy(uniform).item()

    assert peaked_entropy < uniform_entropy
    assert math.isclose(uniform_entropy, math.log(4), rel_tol=1e-4)
    assert peaked_entropy >= 0


def test_majority_vote_picks_winner_and_confidence():
    winner, confidence = majority_vote(["a", "a", "a", "b"])
    assert winner == "a"
    assert confidence == 0.75


def test_majority_vote_unanimous_is_full_confidence():
    winner, confidence = majority_vote(["x", "x", "x"])
    assert winner == "x"
    assert confidence == 1.0


def test_majority_vote_empty_list():
    winner, confidence = majority_vote([])
    assert winner is None
    assert confidence == 0.0


def test_low_confidence_warning_below_threshold():
    assert "warning" in low_confidence_warning(0.3, threshold=0.6)


def test_low_confidence_warning_above_threshold_is_empty():
    assert low_confidence_warning(0.9, threshold=0.6) == ""
