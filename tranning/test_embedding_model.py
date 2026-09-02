"""Pipeline smoke test for embedding_model.py -- proves train() -> checkpoint
-> load_encoder() -> embed_texts() runs end-to-end on a tiny synthetic
sentence set, and that output vectors have the right shape/unit norm. Does
NOT claim any real semantic-similarity quality (see test_transformer_chat.py
for the same contract on the chat model).
"""

import torch

from embedding_model import _resolve_device, embed_texts, encode_text, load_encoder, train

_TINY_MODEL_KWARGS = dict(batch_size=4, max_len=16, n_layer=2, n_embd=16, n_head=2, vocab_size=80)


def _make_synthetic_sentences() -> list[str]:
    return [
        "sinco 是一個從零打造的聊天模型。",
        "hello world, sinco is a small language model.",
        "今天天氣真好，適合出門散步。",
        "the weather is nice today, good for a walk.",
    ]


def test_train_runs_end_to_end(tmp_path):
    out_dir = tmp_path / "embed_runs"
    model, tokenizer, history = train(
        sentences=_make_synthetic_sentences(), out_dir=out_dir, epochs=2, **_TINY_MODEL_KWARGS,
    )

    assert len(history) == 2
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "bpe_vocab.json").exists()
    assert (out_dir / "bpe_merges.json").exists()
    assert (out_dir / "config.json").exists()
    assert all(h["loss"] >= 0 for h in history)
    assert tokenizer.vocab_size > 0


def test_load_encoder_and_embed_texts_after_train(tmp_path):
    out_dir = tmp_path / "embed_runs"
    train(sentences=_make_synthetic_sentences(), out_dir=out_dir, epochs=1, **_TINY_MODEL_KWARGS)

    loaded = load_encoder(out_dir)
    assert loaded is not None
    encoder, tokenizer, max_len, _device = loaded

    vectors = embed_texts(["sinco 是誰", "what is the weather"], encoder, tokenizer)
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 16  # n_embd
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-3


def test_embed_texts_handles_empty_list(tmp_path):
    out_dir = tmp_path / "embed_runs"
    train(sentences=_make_synthetic_sentences(), out_dir=out_dir, epochs=1, **_TINY_MODEL_KWARGS)
    encoder, tokenizer, _max_len, _device = load_encoder(out_dir)
    assert embed_texts([], encoder, tokenizer) == []


def test_load_encoder_without_checkpoint_returns_none(tmp_path):
    assert load_encoder(tmp_path / "no_such_run") is None


def test_encode_text_without_checkpoint_returns_empty_list(tmp_path):
    assert encode_text("hello", out_dir=tmp_path / "no_such_run") == []


def test_resolve_device_prefers_explicit_device_over_detection():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("xpu") == "xpu"


def test_resolve_device_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert _resolve_device(None) == "cpu"
