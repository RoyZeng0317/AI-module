"""Behavior regressions for explicit emotional replies and route boundaries."""

import pytest

from emotion_response import emotion_reply, explicit_emotion


@pytest.mark.parametrize("text,kind", [
    ("我今天心情不好", "negative"), ("我今天心情不好。", "negative"),
    ("今天我心情不好", "negative"), ("我今天不開心", "negative"),
    ("我很傷心", "negative"), ("我覺得很難過", "negative"),
    ("我好想哭", "negative"), ("我不難過", "relief"),
    ("我沒有很難過", "relief"), ("我現在不焦慮", "relief"),
    ("我今天很開心", "positive"), ("我今天心情很好", "positive"),
    ("我不想聊天", "boundary"), ("我今天心情不好，不想聊", "boundary"),
    ("我今天心情不好，先不要問了", "boundary"),
])
def test_explicit_emotion_contrasts(text, kind):
    assert explicit_emotion(text) == kind


@pytest.mark.parametrize("text", [
    "我朋友很難過", "我朋友今天心情不好", "你今天心情不好嗎？",
    "請解釋什麼是難過", "我今天心情不好是歌名", '「我今天心情不好」',
    "我不是不開心", "我不是難過，是生氣", "我心情不好，但現在好多了",
    "我不想不開心", "我今天心情不好，請搜尋附近餐廳", "今天天氣如何", "hello",
])
def test_ambiguous_or_unrelated_input_keeps_normal_chat(text):
    assert emotion_reply(text) is None


def test_user_requested_reaction_asks_what_happened():
    trace, reply = emotion_reply("我今天心情不好")
    assert "你怎麼了" in reply and "願意" in reply
    assert "非模型生成" in trace
    assert "開心" not in reply and "慶祝" not in reply


def test_boundary_reply_does_not_question_or_pressure_user():
    _, reply = emotion_reply("我今天心情不好，不想聊")
    assert "不追問" in reply and "？" not in reply
