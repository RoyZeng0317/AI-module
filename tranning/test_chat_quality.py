"""Observable routing and quality checks; no model-quality claims or GPU work."""
import pytest
import chats, chat_quality


@pytest.mark.parametrize("message", ["寫一個故事", "寫一段感謝信", "用中文寫一個童話", "write a story"])
def test_creative_writing_is_not_code(message):
    assert not chats.is_code_request(message)


@pytest.mark.parametrize("message", ["寫一個 python 函式", "用 C++ 寫一個氣泡排序法", "寫一段 SQL", "write Python code", "用 C# 寫一個類別"])
def test_code_intent_retained(message):
    assert chats.is_code_request(message)


def test_exact_dialogue_is_not_fuzzy():
    assert "你好" in chat_quality.dialogue_reply(" 你好！ ")
    assert chat_quality.dialogue_reply("你不好嗎") is None
    assert chat_quality.dialogue_reply("謝謝但我還是不懂") is None


@pytest.mark.parametrize("message", ["你懂得股票投資嗎?", "你了解股票嗎", "你懂投資嗎"])
def test_investment_capability_is_not_a_price_or_trade_recommendation(message):
    reply = chat_quality.dialogue_reply(message)
    assert "基本概念" in reply
    assert "想先了解" in reply
    assert chat_quality.dialogue_reply("你知道股票今天跌多少嗎") is None


@pytest.mark.parametrize("reply", ["", "...", "<unk>", "哈哈哈哈哈哈", "Hello, I can help.", "請幫我整理資料"])
def test_observable_bad_outputs(reply):
    assert chat_quality.output_issue("請幫我整理資料", reply)


def test_valid_output_is_not_claimed_as_confident():
    assert chat_quality.output_issue("請幫我整理資料", "請提供資料與想整理的欄位。") is None
    # This structural guard deliberately cannot prove factual relevance.
    assert chat_quality.output_issue("請幫我整理資料", "蘋果是一種水果。") is None


def test_observed_unrelated_topic_regressions_and_counterexamples():
    assert chat_quality.output_issue("你懂得股票投資嗎", "可以陪你討論程式邏輯。")
    assert chat_quality.output_issue("名詞解釋什麼是RAG", "S+V，主詞動詞受詞補語。")
    assert chat_quality.output_issue("用 Python 分析股票", "請提供你的 Python 程式碼。") is None
    assert chat_quality.output_issue("RAG 是用來回答英文文法嗎", "可以先檢索英文文法，例如主詞與受詞的資料。") is None


def test_known_without_history_skips_but_followup_with_history_generates(monkeypatch):
    monkeypatch.setattr(chats.transformer_chat, "context_enabled", lambda: True)
    monkeypatch.setattr(chats, "route_reply", lambda message: None)
    monkeypatch.setattr(chats.transformer_chat, "reply", lambda message: pytest.fail("must not generate"))
    assert "你好" in chats.smart_reply_traced("你好")[1]
    seen = {}
    def contextual(message, *, history):
        seen.update(message=message, history=history)
        return "你指的是火星。"
    monkeypatch.setattr(chats.transformer_chat, "reply", contextual)
    monkeypatch.setattr(chats, "_emotion_trace_suffix", lambda message: "")
    trace, answer = chats.smart_reply_traced("那個呢", history=[("火星", "紅色星球")])
    assert answer == "你指的是火星。"
    assert seen == {"message": "那個呢", "history": [("火星", "紅色星球")]}
    assert "精確比對" not in trace


def test_legacy_checkpoint_keeps_existing_followup_clarification(monkeypatch):
    monkeypatch.setattr(chats, "route_reply", lambda message: None)
    monkeypatch.setattr(chats.transformer_chat, "context_enabled", lambda: False)
    monkeypatch.setattr(chats.transformer_chat, "reply", lambda *args, **kwargs: pytest.fail("舊權重不應假裝支援上下文"))
    trace, answer = chats.smart_reply_traced("那個呢", history=[("火星", "紅色星球")])
    assert "補充" in answer and "精確比對" in trace


def test_unknown_prompt_still_generates_and_bad_output_is_guarded(monkeypatch):
    monkeypatch.setattr(chats, "route_reply", lambda message: None)
    monkeypatch.setattr(chats, "_emotion_trace_suffix", lambda message: "")
    monkeypatch.setattr(chats.transformer_chat, "reply", lambda message: "可以，請貼上文章內容。")
    assert chats.smart_reply_traced("我有一篇文章想整理")[1] == "可以，請貼上文章內容。"
    monkeypatch.setattr(chats.transformer_chat, "reply", lambda message: "AI AI AI AI AI")
    trace, answer = chats.smart_reply_traced("我有一篇文章想整理")
    assert "輸出檢查" in trace
    assert "未能產生可用" in answer


def test_manual_external_mode_preserved(monkeypatch):
    monkeypatch.setattr(chats, "route_reply", lambda message: None)
    monkeypatch.setattr(chats, "_nvidia_reply", lambda message, history=None: ("", "manual"))
    assert chats.smart_reply_traced("hello", force_mode="nvidia")[1] == "manual"


@pytest.mark.parametrize("message", ["how are you?", "ok, got it.", "what are you doing now?", "what can you do?", "can you speak English?"])
def test_english_dialogue_and_clarification_stay_english(message):
    reply = chat_quality.dialogue_reply(message)
    assert reply and all(ord(char) < 128 for char in reply)
    assert all(ord(char) < 128 for char in chat_quality.clarification(message))


def test_english_mixed_language_guard_and_requested_translation():
    assert chat_quality.output_issue("Can you help me?", "Sure, 可以幫忙。")
    assert chat_quality.output_issue("Translate hello into Chinese", "你好") is None


@pytest.mark.parametrize("message", ["Do you know about stock investing?", "Can you explain stock investing?", "Do you understand stock investing?"])
def test_english_investment_capability_is_explicit_and_english(message):
    reply = chat_quality.dialogue_reply(message)
    assert "basics" in reply and "risk" in reply
    assert "lose money" in reply
    assert all(ord(char) < 128 for char in reply)
    assert chat_quality.dialogue_reply("Can you explain why stock investing lost money yesterday?") is None
