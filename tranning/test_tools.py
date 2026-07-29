"""Tests for tools.route_reply()'s calculus routing (quiz / reveal-last /
free-form solve).

Weather/search routing isn't covered here (those need a real network call,
same reason chats.py/test_chats.py never exercises them either) — this file
only covers the calculus additions on top, which are pure local computation
(calculus_generator.py / calculus_solver.py) and need no network.

route_reply() caches the most recently posed quiz question in a module-level
global (tools._last_calculus_problem) — see tools.py's comment on why a
plain global is enough for this single-user desktop app. Because it's
module-level state shared across tests, every test that depends on it
starting empty resets it explicitly first, rather than relying on test
execution order.
"""

import tools
from tools import _calculus_topic_if_requested, _is_solve_last_request, route_reply


def _reset_last_problem():
    tools._last_calculus_problem = None


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
    assert _calculus_topic_if_requested("我今天要交微積分作業") is None
    assert _calculus_topic_if_requested("微積分") is None


def test_no_topic_word_returns_none_even_with_request_hint():
    assert _calculus_topic_if_requested("考我英文單字") is None


def test_is_solve_last_request_exact_match_only():
    assert _is_solve_last_request("解題")
    assert _is_solve_last_request("解題！")
    assert _is_solve_last_request("Show Answer")
    assert not _is_solve_last_request("幫我解 3x^2 的微分")  # contains "解" but isn't a bare reveal request


def test_route_reply_quiz_request_returns_question_only_and_stores_state():
    _reset_last_problem()
    result = route_reply("出一題微分")
    assert result is not None
    reason, reply = result
    assert "出題請求" in reason
    assert "sinco 微積分出題" in reply
    assert "題目" in reply
    assert tools._last_calculus_problem is not None
    assert tools._last_calculus_problem["topic"] == "derivative"
    # two-step flow: the actual worked answer/steps are not revealed yet
    assert tools._last_calculus_problem["answer"] not in reply
    for step in tools._last_calculus_problem["steps"]:
        assert step not in reply


def test_route_reply_solve_last_reveals_stored_problem():
    _reset_last_problem()
    route_reply("出一題積分")  # pose a question, storing it
    reason, reply = route_reply("解題")
    assert "解題」請求" in reason
    assert "答案" in reply
    assert "詳解" in reply


def test_route_reply_solve_last_without_prior_quiz_returns_friendly_message():
    _reset_last_problem()
    reason, reply = route_reply("解題")
    assert "沒有已出的題目" in reason
    assert "出一題微積分" in reply


def test_route_reply_calculus_quiz_does_not_fall_through_to_search():
    _reset_last_problem()
    reason, _reply = route_reply("出一題微積分")
    assert "搜尋" not in reason


def test_route_reply_plain_calculus_mention_falls_back_to_none():
    _reset_last_problem()
    assert route_reply("我今天要交微積分作業") is None


def test_route_reply_freeform_solve_derivative():
    _reset_last_problem()
    reason, reply = route_reply("3x^2+5x 的微分")
    assert "自由輸入" in reason
    assert "sinco 解題" in reply
    assert "答案" in reply
    assert tools._last_calculus_problem["topic"] == "derivative"


def test_route_reply_freeform_solve_integral():
    _reset_last_problem()
    reason, reply = route_reply("∫ x^2 dx")
    assert "自由輸入" in reason
    assert "積分" in reply


def test_route_reply_freeform_solve_limit():
    _reset_last_problem()
    reason, reply = route_reply("lim(x->0) sin(x)/x")
    assert "自由輸入" in reason
    assert "= 1" in reply
