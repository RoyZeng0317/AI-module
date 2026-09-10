"""Pipeline smoke test for transformer_chat.py.

Same contract as test_chats.py: this does NOT claim any reply-quality —
there is no real pretraining corpus or conversation dataset yet (see
CLAUDE.md to-do). It only proves pretrain() -> finetune() -> reply() runs
end-to-end without crashing, on a tiny synthetic corpus/pair set with a
tiny model config so it stays fast.
"""

import json

import torch

from transformer_chat import GPT, _resolve_device, complete, finetune, pretrain, reply

_TINY_MODEL_KWARGS = dict(
    batch_size=4, block_size=24, n_layer=2, n_embd=16, n_head=2, vocab_size=80,
)


def _make_synthetic_corpus() -> str:
    return (
        "sinco 是一個從零打造的聊天模型。sinco 正在學習中文與英文。"
        "hello world, sinco is a small language model. "
    ) * 20


def _make_synthetic_pairs() -> list[dict]:
    return [
        {"prompt": "hello", "reply": "hi there"},
        {"prompt": "how are you", "reply": "i am fine"},
        {"prompt": "what is your name", "reply": "i am sinco"},
        {"prompt": "bye", "reply": "goodbye"},
    ] * 5


def test_pretrain_runs_end_to_end(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    out_dir = tmp_path / "pretrain_runs"

    model, tokenizer, history = pretrain(
        corpus_path=corpus_path, out_dir=out_dir, epochs=2, val_split=0.2, patience=5,
        **_TINY_MODEL_KWARGS,
    )

    assert len(history) == 2
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "bpe_vocab.json").exists()
    assert (out_dir / "bpe_merges.json").exists()
    assert (out_dir / "config.json").exists()
    assert all(h["train_loss"] >= 0 for h in history)
    assert tokenizer.vocab_size > 0


def test_finetune_runs_end_to_end_after_pretrain(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)

    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    finetune_dir = tmp_path / "finetune_runs"

    model, tokenizer, history = finetune(
        data_path=data_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
        epochs=2, batch_size=4, val_split=0.2, patience=5,
    )

    assert len(history) == 2
    assert (finetune_dir / "model.pt").exists()
    assert (finetune_dir / "config.json").exists()

    generated = reply("hello", out_dir=finetune_dir, max_new_tokens=10)
    assert isinstance(generated, str)
    assert len(generated) > 0


def test_reply_without_checkpoint_returns_placeholder(tmp_path):
    result = reply("hello", out_dir=tmp_path / "no_such_run")
    assert "尚未訓練" in result


def test_complete_runs_on_pretrain_only_checkpoint(tmp_path):
    """complete() is the entry point for a checkpoint that only ever went
    through pretrain() (e.g. gpt_code_pretrain_runs/) -- no <sep>/reply
    structure to expect, unlike reply()."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)

    generated = complete("sinco", out_dir=pretrain_dir, max_new_tokens=10)
    assert isinstance(generated, str)
    assert len(generated) > 0


def test_complete_without_checkpoint_returns_placeholder(tmp_path):
    result = complete("sinco", out_dir=tmp_path / "no_such_run")
    assert "尚未訓練" in result


def test_generate_repetition_penalty_breaks_degenerate_repeat_loop():
    """to-do #17 observed finetune()'d replies collapse into a single token
    repeated to the output cap (e.g. "看看看看..."). Reproduces that failure
    mode directly (monkeypatched forward() that always scores one token far
    above the rest, independent of any real training) and checks
    repetition_penalty actually breaks the loop instead of just not crashing.
    """
    model = GPT(vocab_size=10, block_size=16, n_layer=1, n_embd=8, n_head=1, dropout=0.0)
    model.eval()

    def _dominant_token_forward(idx, targets=None, return_hidden=False):
        # every other token still scores positively (2.0) -- repetition
        # penalty divides a *positive* logit down rather than flipping its
        # sign, so the competing tokens need a positive score of their own
        # to have any chance of overtaking token 3 once it gets penalized.
        logits = torch.full((idx.size(0), idx.size(1), 10), 2.0)
        logits[:, :, 3] = 6.0
        return logits, None

    model.forward = _dominant_token_forward

    idx = torch.tensor([[1, 2]])
    torch.manual_seed(0)
    out_no_penalty = model.generate(idx, max_new_tokens=8, temperature=0.2,
                                     top_k=None, top_p=None, repetition_penalty=1.0)
    without_penalty = out_no_penalty[0, 2:].tolist()
    assert without_penalty.count(3) == len(without_penalty)

    torch.manual_seed(0)
    out_with_penalty = model.generate(idx, max_new_tokens=8, temperature=0.2,
                                       top_k=None, top_p=None, repetition_penalty=5.0)
    with_penalty = out_with_penalty[0, 2:].tolist()
    assert with_penalty.count(3) < len(with_penalty)


def test_reply_and_complete_accept_repetition_penalty(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)

    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    finetune_dir = tmp_path / "finetune_runs"
    finetune(data_path=data_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
              epochs=2, batch_size=4, val_split=0.2, patience=5)

    generated = reply("hello", out_dir=finetune_dir, max_new_tokens=10, repetition_penalty=1.5)
    assert isinstance(generated, str) and len(generated) > 0

    completed = complete("sinco", out_dir=pretrain_dir, max_new_tokens=10, repetition_penalty=1.5)
    assert isinstance(completed, str) and len(completed) > 0


def test_resolve_device_prefers_explicit_device_over_detection():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("xpu") == "xpu"


def test_resolve_device_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert _resolve_device(None) == "cpu"
