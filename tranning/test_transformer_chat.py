"""Pipeline smoke test for transformer_chat.py.

Same contract as test_chats.py: this does NOT claim any reply-quality —
there is no real pretraining corpus or conversation dataset yet (see
CLAUDE.md to-do). It only proves pretrain() -> finetune() -> reply() runs
end-to-end without crashing, on a tiny synthetic corpus/pair set with a
tiny model config so it stays fast.
"""

import json

import torch

from transformer_chat import _resolve_device, complete, finetune, pretrain, reply

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


def test_finetune_checkpoint_every_writes_resumable_mid_run_checkpoint(tmp_path):
    """to_do_list.md #37 / ErrorLog #32: a long finetune killed mid-run had
    no checkpoint to resume from, only the final (never-written) weights.
    checkpoint_every writes the same tokenizer/config/model.pt/history.json
    layout finetune()'s own final output uses, to <out_dir>/checkpoint/, so
    resuming needs no separate code path -- just point a new finetune() call's
    pretrain_dir at it."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)

    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    finetune_dir = tmp_path / "finetune_runs"

    finetune(data_path=data_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
              epochs=4, batch_size=4, val_split=0.2, patience=100, checkpoint_every=2)

    ckpt_dir = finetune_dir / "checkpoint"
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "config.json").exists()
    assert (ckpt_dir / "bpe_vocab.json").exists()
    ckpt_history = json.loads((ckpt_dir / "history.json").read_text(encoding="utf-8"))
    # last checkpoint write happens at epoch 4 (checkpoint_every=2 -> epochs 2 and 4)
    assert ckpt_history[-1]["epoch"] == 4

    # resuming: point a fresh finetune() call's pretrain_dir at the checkpoint
    resumed_dir = tmp_path / "resumed_runs"
    model, tokenizer, history = finetune(
        data_path=data_path, pretrain_dir=ckpt_dir, out_dir=resumed_dir,
        epochs=1, batch_size=4, val_split=0.2, patience=5,
    )
    assert len(history) == 1
    assert (resumed_dir / "model.pt").exists()


def test_pretrain_checkpoint_every_zero_writes_no_checkpoint_dir(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    out_dir = tmp_path / "pretrain_runs"

    pretrain(corpus_path=corpus_path, out_dir=out_dir, epochs=2, val_split=0.2,
              patience=5, checkpoint_every=0, **_TINY_MODEL_KWARGS)

    assert not (out_dir / "checkpoint").exists()


def test_resolve_device_prefers_explicit_device_over_detection():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("xpu") == "xpu"


def test_resolve_device_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert _resolve_device(None) == "cpu"


def _pretrain_and_write_pairs(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    return corpus_path, pretrain_dir, data_path


def test_finetune_freezing_keeps_frozen_weights_identical_to_pretrain(tmp_path):
    """freeze_embeddings + freeze_layers: the frozen part of model.pt must
    come out byte-identical to the pretrain checkpoint, while the unfrozen
    top block still gets trained."""
    corpus_path, pretrain_dir, data_path = _pretrain_and_write_pairs(tmp_path)
    finetune_dir = tmp_path / "finetune_runs"

    finetune(data_path=data_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
             epochs=2, batch_size=4, val_split=0.2, patience=5,
             freeze_embeddings=True, freeze_layers=1)

    before = torch.load(pretrain_dir / "model.pt", map_location="cpu")
    after = torch.load(finetune_dir / "model.pt", map_location="cpu")
    for key in before:
        if key.startswith(("tok_emb.", "pos_emb.", "head.", "blocks.0.")):
            assert torch.equal(before[key], after[key]), key
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("blocks.1."))

    config = json.loads((finetune_dir / "config.json").read_text(encoding="utf-8"))
    assert config["finetune_regularization"]["freeze_layers"] == 1
    assert reply("hello", out_dir=finetune_dir, max_new_tokens=5)


def test_finetune_rejects_freeze_layers_beyond_model_depth(tmp_path):
    corpus_path, pretrain_dir, data_path = _pretrain_and_write_pairs(tmp_path)
    import pytest
    with pytest.raises(ValueError):
        finetune(data_path=data_path, pretrain_dir=pretrain_dir, out_dir=tmp_path / "ft",
                 epochs=1, batch_size=4, freeze_layers=3)  # tiny model has n_layer=2


def test_finetune_label_smoothing_typo_noise_and_rehearsal_run_end_to_end(tmp_path):
    corpus_path, pretrain_dir, data_path = _pretrain_and_write_pairs(tmp_path)
    finetune_dir = tmp_path / "finetune_runs"

    model, tokenizer, history = finetune(
        data_path=data_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
        epochs=2, batch_size=4, val_split=0.2, patience=5,
        label_smoothing=0.1, typo_noise_prob=1.0, lm_corpus=corpus_path, lm_mix_ratio=0.5,
    )

    assert len(history) == 2
    assert all(h["train_loss"] >= 0 and h["val_loss"] >= 0 for h in history)
    assert model.label_smoothing == 0.1


def test_label_smoothing_only_affects_training_mode_loss():
    from transformer_chat import GPT
    torch.manual_seed(0)
    model = GPT(vocab_size=20, block_size=8, n_layer=1, n_embd=8, n_head=2, dropout=0.0)
    x = torch.randint(4, 20, (2, 8))
    y = torch.randint(4, 20, (2, 8))

    model.eval()
    _, plain = model(x, y)
    model.label_smoothing = 0.2
    _, eval_smoothed = model(x, y)
    model.train()
    _, train_smoothed = model(x, y)

    assert torch.allclose(plain, eval_smoothed)
    assert not torch.allclose(plain, train_smoothed)


def test_sft_split_is_stable_and_keeps_prompt_variants_together():
    import random
    from transformer_chat import _split_sft_pairs
    pairs = _make_synthetic_pairs() + [{"prompt": " HELLO ", "reply": "another answer"}]
    train, val = _split_sft_pairs(pairs, 0.2, 42)
    reordered = pairs[::-1]
    random.seed(123)
    train2, val2 = _split_sft_pairs(reordered, 0.2, 42)
    normalize = lambda rows: {p["prompt"].strip().casefold() for p in rows}
    assert normalize(train) == normalize(train2)
    assert normalize(val) == normalize(val2)
    assert not normalize(train) & normalize(val)
    assert len(train) + len(val) == len(pairs)


def test_sft_split_rejects_fake_or_leaking_validation():
    import pytest
    from transformer_chat import _split_sft_pairs
    pairs = _make_synthetic_pairs()
    for bad in ([], [{"prompt": " HELLO ", "reply": "different"}]):
        with pytest.raises(ValueError):
            _split_sft_pairs(pairs, 0.2, 42, bad)
    with pytest.raises(ValueError):
        _split_sft_pairs([pairs[0]], 0.2, 42)
    for fraction in (0, 1, -0.1):
        with pytest.raises(ValueError):
            _split_sft_pairs(pairs, fraction, 42)


def test_evaluation_matches_token_cross_entropy_across_batch_sizes():
    from transformer_chat import GPT, _evaluate_loss
    from bpe_tokenizer import PAD
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    torch.manual_seed(4)
    model = GPT(12, 4, 1, 8, 2, 0.5)
    model.label_smoothing = 0.4
    x = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    y = torch.tensor([[2, PAD, PAD, PAD], [3, 4, 5, 6], [4, 5, PAD, PAD]])
    dataset = TensorDataset(x, y)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        expected = F.cross_entropy(logits.reshape(-1, 12), y.reshape(-1), ignore_index=PAD).item()
    for size in (1, 2, 3):
        result = _evaluate_loss(model, DataLoader(dataset, batch_size=size), "cpu")
        assert abs(result - expected) < 1e-6
    assert model.training is False


def test_finetune_records_clean_loss_and_validation_membership(tmp_path):
    _, pretrain_dir, data_path = _pretrain_and_write_pairs(tmp_path)
    out = tmp_path / "finetune"
    _, _, history = finetune(data_path, pretrain_dir, out, epochs=1, device="cpu",
                            label_smoothing=0.2, typo_noise_prob=0.5)
    assert history[0]["train_eval_loss"] >= 0
    config = json.loads((out / "config.json").read_text())
    split = config["validation"]
    assert split["split_seed"] == 42
    assert not set(split["train_prompts"]) & set(split["val_prompts"])
    assert split["train_pairs"] + split["val_pairs"] == 20
