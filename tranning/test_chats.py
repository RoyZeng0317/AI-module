"""Pipeline smoke test for chats.py.

This does NOT claim any reply quality — there is no real conversation
dataset yet. It only proves the seq2seq training loop and greedy-decoding
inference run end-to-end without crashing, using a handful of synthetic
prompt/reply pairs.
"""

import json

from chats import chat_reply, train


def _make_synthetic_pairs():
    return [
        {"prompt": "hello", "reply": "hi there"},
        {"prompt": "how are you", "reply": "i am fine"},
        {"prompt": "what is your name", "reply": "i am a chatbot"},
        {"prompt": "bye", "reply": "goodbye"},
    ] * 5  # repeated so each batch has more than one distinct example


def test_training_loop_runs_end_to_end(tmp_path):
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    out_dir = tmp_path / "runs"

    encoder, decoder, vocab, history = train(
        data_path=data_path, out_dir=out_dir, epochs=3, batch_size=4,
        embed_size=16, hidden_size=32, lr=1e-2, max_len=12, teacher_forcing_ratio=0.5,
    )

    assert len(history) == 3
    assert (out_dir / "encoder.pt").exists()
    assert (out_dir / "decoder.pt").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "config.json").exists()
    assert all(h["loss"] >= 0 for h in history)


def test_chat_reply_without_checkpoint_returns_placeholder(tmp_path):
    reply = chat_reply("hello", out_dir=tmp_path / "no_such_run")
    assert "尚未訓練" in reply


def test_chat_reply_after_training_returns_string(tmp_path):
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    out_dir = tmp_path / "runs"

    train(data_path=data_path, out_dir=out_dir, epochs=2, batch_size=4,
          embed_size=16, hidden_size=32, lr=1e-2, max_len=12, teacher_forcing_ratio=0.5)

    reply = chat_reply("hello", out_dir=out_dir)
    assert isinstance(reply, str)
    assert len(reply) > 0
