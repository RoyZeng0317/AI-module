"""Tests for tools.route_reply()'s calculus-quiz routing.

Weather/search routing isn't covered here (those need a real network call,
same reason chats.py/test_chats.py never exercises them either) — this file
only covers the calculus-request detection and routing added on top, which
is pure local computation (calculus_generator.py) and needs no network.
"""

from tools import _calculus_topic_if_requested, route_reply


def test_generic_calculus_request_detected_as_random_topic():
    assert _calculus_topic_if_requested("出一題微積分") == "random"


def test_specific_subtopic_keywords_detected():
    assert _calculus_topic_if_requested("出一題微分") == "derivative"
    assert _calculus_topic_if_requested("出一題導數") == "derivative"
    assert _calculus_topic_if_requested("出一題積分") == "integral"
    assert _calculus_topic_if_requested("出一題極限") == "limit"
    assert _calculus_topic_if_requested("give me a problem on derivative") == "derivative"
    assert _calculus_topic_if_requested("quiz me on integral") == "integral"


def test_specific_subtopic_inside_generic_phrase_overrides_random():
    # "積分" is a literal substring of "微積分" — the specific mention should
    # still win over treating this as a generic "pick any topic" request.
    assert _calculus_topic_if_requested("出一題微積分裡的積分題") == "integral"


def test_no_request_hint_returns_none_even_with_topic_word():
    # bare mention of the subject, not a "quiz me" request — must NOT fire,
    # same "needs both parts" shape as the weather/search patterns in tools.py.
    assert _calculus_topic_if_requested("我今天要交微積分作業") is None
    assert _calculus_topic_if_requested("微積分") is None


def test_no_topic_word_returns_none_even_with_request_hint():
    assert _calculus_topic_if_requested("考我英文單字") is None


def test_route_reply_returns_formatted_problem_for_calculus_request():
    result = route_reply("出一題微分")
    assert result is not None
    reason, reply = result
    assert "出題請求" in reason
    assert "sinco 微積分出題" in reply
    assert "題目" in reply
    assert "答案" in reply


def test_route_reply_calculus_request_does_not_fall_through_to_search():
    # "出一題微積分" must not accidentally match a _SEARCH_PATTERNS rule and
    # trigger a real web_search() call (which needs network).
    reason, _reply = route_reply("出一題微積分")
    assert "搜尋" not in reason


def test_route_reply_plain_calculus_mention_falls_back_to_none():
    assert route_reply("我今天要交微積分作業") is None
