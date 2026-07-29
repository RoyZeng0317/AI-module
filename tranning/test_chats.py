"""Pipeline smoke test for chats.py.

This does NOT claim any reply quality — there is no real conversation
dataset yet. It only proves the seq2seq training loop and greedy-decoding
inference run end-to-end without crashing, using a handful of synthetic
prompt/reply pairs.
"""

import json

from chats import chat_reply, is_code_request, mc_chat_reply, smart_reply_traced, train


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


def test_mc_chat_reply_without_checkpoint_returns_placeholder_and_zero_confidence(tmp_path):
    reply, confidence = mc_chat_reply("hello", out_dir=tmp_path / "no_such_run")
    assert "尚未訓練" in reply
    assert confidence == 0.0


def test_mc_chat_reply_after_training_returns_reply_and_confidence(tmp_path):
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    out_dir = tmp_path / "runs"

    train(data_path=data_path, out_dir=out_dir, epochs=2, batch_size=4,
          embed_size=16, hidden_size=32, lr=1e-2, max_len=12, teacher_forcing_ratio=0.5)

    reply, confidence = mc_chat_reply("hello", out_dir=out_dir, mc_samples=5)
    assert isinstance(reply, str)
    assert len(reply) > 0
    assert 0.0 <= confidence <= 1.0


def test_smart_reply_traced_includes_confidence_for_model_path(tmp_path):
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    out_dir = tmp_path / "runs"

    train(data_path=data_path, out_dir=out_dir, epochs=2, batch_size=4,
          embed_size=16, hidden_size=32, lr=1e-2, max_len=12, teacher_forcing_ratio=0.5)

    trace, reply = smart_reply_traced("hello", out_dir=out_dir)
    assert "信心度" in trace
    assert isinstance(reply, str)


def test_smart_reply_traced_force_mode_overrides_auto_routing(tmp_path):
    """/model 指令（CLAUDE.md 需求 #01）靠 force_mode 手動覆蓋 is_code_request()
    的自動判斷：force_mode="sinco" 時，即使訊息長得像程式碼請求，也要留在
    out_dir（一般聊天 checkpoint），不能被自動判斷搶走、跑去 CODE_OUT_DIR。
    """
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    out_dir = tmp_path / "runs"
    train(data_path=data_path, out_dir=out_dir, epochs=2, batch_size=4,
          embed_size=16, hidden_size=32, lr=1e-2, max_len=12, teacher_forcing_ratio=0.5)

    code_shaped_message = "寫一個 python 函式"
    assert is_code_request(code_shaped_message)  # 確認這句話本來就會觸發自動判斷

    auto_trace, _ = smart_reply_traced(code_shaped_message, out_dir=out_dir, force_mode="auto")
    assert "sinco-code" in auto_trace  # 自動模式：照舊被判斷成程式碼請求

    forced_trace, _ = smart_reply_traced(code_shaped_message, out_dir=out_dir, force_mode="sinco")
    assert "已手動切換為一般聊天模式" in forced_trace
    assert "sinco-code" not in forced_trace

    forced_code_trace, _ = smart_reply_traced("hello", out_dir=out_dir, force_mode="code")
    assert "已手動切換為程式碼模式" in forced_code_trace
