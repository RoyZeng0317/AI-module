"""Pipeline smoke test for transformer_chat.py.

Same contract as test_chats.py: this does NOT claim any reply-quality —
there is no real pretraining corpus or conversation dataset yet (see
CLAUDE.md to-do). It only proves pretrain() -> finetune() -> reply() runs
end-to-end without crashing, on a tiny synthetic corpus/pair set with a
tiny model config so it stays fast.
"""

import json

from transformer_chat import finetune, pretrain, reply

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
