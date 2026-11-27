import sys

import transformer_chat


def test_context_repl_retains_six_turns_and_starts_fresh(monkeypatch):
    calls = []
    def fake_reply(message, **kwargs):
        calls.append((message, list(kwargs["history"])))
        return "回答"
    monkeypatch.setattr(transformer_chat, "reply", fake_reply)
    monkeypatch.setattr(sys, "argv", ["transformer_chat.py", "chat"])
    messages = iter([*(str(i) for i in range(8)), "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(messages))
    transformer_chat.main()
    assert calls[0][1] == []
    assert calls[1][1] == [("0", "回答")]
    assert calls[-1][1] == [(str(i), "回答") for i in range(1, 7)]
    messages = iter(["新聊天", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(messages))
    transformer_chat.main()
    assert calls[-1][1] == []
