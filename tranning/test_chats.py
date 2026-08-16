"""Pipeline smoke test for chats.py.

This does NOT claim any reply quality — there is no real conversation
dataset yet. It only proves the seq2seq training loop and greedy-decoding
inference run end-to-end without crashing, using a handful of synthetic
prompt/reply pairs.
"""

import json

import chats
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


def test_smart_reply_traced_nvidia_mode_routes_to_nvidia_reply(monkeypatch, tmp_path):
    """/model nvidia（外部 NVIDIA 雲端 API，非本專案自訓練）：force_mode="nvidia"
    要完全跳過 sinco/sinco-code 的自動判斷與 checkpoint 讀取，直接呼叫
    chats._nvidia_reply()。這裡 monkeypatch 掉 _nvidia_reply 本身，不打真的
    NVIDIA API（不需要真的 NVIDIA_API_KEY，也不依賴網路）。
    """
    monkeypatch.setattr(chats, "_nvidia_reply", lambda message, history=None: ("因為...", f"echo: {message}"))

    trace, reply = smart_reply_traced("寫一個 python 函式", out_dir=tmp_path / "unused", force_mode="nvidia")

    assert "NVIDIA" in trace
    assert "因為..." in trace
    assert reply == "echo: 寫一個 python 函式"


def test_smart_reply_traced_nvidia_mode_without_reasoning(monkeypatch, tmp_path):
    """_nvidia_reply() 回傳空字串思考過程時（模型沒開啟 thinking，或 API 呼叫
    失敗走例外分支），trace 只保留原本的切換說明，不能出現空的「思考過程：」段落。
    """
    monkeypatch.setattr(chats, "_nvidia_reply", lambda message, history=None: ("", f"echo: {message}"))

    trace, reply = smart_reply_traced("hello", out_dir=tmp_path / "unused", force_mode="nvidia")

    assert "思考過程" not in trace


def test_smart_reply_traced_nvidia_mode_forwards_history(monkeypatch, tmp_path):
    """/resume 還原回來的歷史（或 GUI 累積的 Conversation.history）要原封不動
    轉交給 _nvidia_reply()，nemotron 才能把之前幾輪當多輪對話的 messages 脈絡，
    而不是每次都被當成全新對話（這是「NVIDIA 模型沒有還原對話紀錄」的根因）。
    """
    seen = {}

    def fake_nvidia_reply(message, history=None):
        seen["history"] = history
        return "", f"echo: {message}"

    monkeypatch.setattr(chats, "_nvidia_reply", fake_nvidia_reply)
    history = [("你好", "哈囉，我是 nemotron")]

    smart_reply_traced("再說一次", out_dir=tmp_path / "unused", force_mode="nvidia", history=history)

    assert seen["history"] == history
