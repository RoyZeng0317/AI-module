"""Shared Monte Carlo Dropout building blocks — the practical, from-scratch
way this project does "Bayesian neural network" on an RTX 4060 8GB budget.

Why MC Dropout instead of a full variational network (Bayes by Backprop):
a true Bayesian layer replaces every weight with its own (mean, variance)
pair, roughly doubling parameter count and requiring a reparameterized
forward pass everywhere. MC Dropout (Gal & Ghahramani, 2016, "Dropout as a
Bayesian Approximation") gets an approximate posterior for free out of a
dropout layer that was already there for ordinary regularization: training
is completely unchanged (still plain gradient descent on cross-entropy /
CTC / whatever loss, via Adam/AdamW — no new loss term, no ELBO), and at
inference time the trick is simply to NOT switch dropout off. Running the
same input through the network N times, with a different random dropout
mask each time, is mathematically an approximate sample from the weight
posterior each time — average the outputs and you get a mean prediction;
look at how much the N samples disagree and you get an honest uncertainty
number, instead of a single greedy forward pass that always looks equally
confident whether the model actually knows the answer or is guessing.

Honest limitation: this does NOT mean a model needs less training data to
learn — MC Dropout doesn't create information that wasn't in the data. What
it buys is a calibrated-ish confidence/entropy signal so a human (or a
calling script) can tell "the model is sure" from "the model produced N
different guesses", which matters most exactly when training data is small.

Used by chats.py (mc_chat_reply — majority vote over N greedy decodes) and
image_classifier_bnn.py (mc_predict_classification — mean softmax over N
forward passes).
"""

from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)


@contextmanager
def mc_dropout_mode(model: nn.Module):
    """Temporarily leaves only this model's Dropout layers stochastic
    ("training-mode") while everything else — most importantly BatchNorm,
    which MobileNetV2's backbone uses heavily — stays in eval mode. A plain
    `model.train()` would also let BatchNorm update its running mean/var
    from whatever single inference batch is passed in, silently corrupting
    the trained statistics; walking `model.modules()` and only flipping the
    Dropout instances avoids that while still getting the MC sampling.
    Restores the model's original mode on exit.
    """
    was_training = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
    try:
        yield
    finally:
        model.train(was_training)


def mc_predict_classification(model: nn.Module, inputs: torch.Tensor, n_samples: int = 20):
    """Runs `n_samples` stochastic forward passes (see mc_dropout_mode) and
    softmaxes each one, returning (mean_probs, std_probs) across samples —
    the MC Dropout approximation to a Bayesian posterior predictive for a
    classification head. Feed mean_probs to predictive_entropy() for a
    single scalar confidence number.
    """
    samples = []
    with mc_dropout_mode(model), torch.no_grad():
        for _ in range(n_samples):
            samples.append(torch.softmax(model(inputs), dim=-1))
    stacked = torch.stack(samples, dim=0)  # (n_samples, batch, num_classes)
    return stacked.mean(dim=0), stacked.std(dim=0)


def predictive_entropy(mean_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of the mean predictive distribution: ~0 when the
    model is confident and concentrated on one class, ln(num_classes) when
    it's spread evenly across all of them (maximally unsure).
    """
    probs = mean_probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def majority_vote(items: list):
    """Picks the most common item and reports how often it won — used to
    turn N stochastic greedy-decoded chat replies into one reply plus a
    confidence score (agreement fraction), the sequence-generation
    equivalent of mc_predict_classification's mean/std for classification.
    """
    if not items:
        return None, 0.0
    winner, count = Counter(items).most_common(1)[0]
    return winner, count / len(items)


def low_confidence_warning(confidence: float, threshold: float = 0.6) -> str:
    """Same bracketed-warning convention as train_utils.loss_gap_warning() /
    accuracy_gap_warning(), but for a single Bayesian prediction rather than
    a training epoch: low agreement across MC Dropout samples usually means
    "not enough training data covers this case" rather than a bug, so it's
    surfaced as a hint to be skeptical rather than an error.
    """
    if confidence < threshold:
        return f"  [warning: low MC Dropout confidence ({confidence:.0%}) — treat this prediction with skepticism]"
    return ""
