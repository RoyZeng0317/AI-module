"""Tests for tools.route_reply()'s math-word-problem and logic-reasoning
quiz routing — same two-step "出題 -> 解題" shape as
test_tools.py's calculus tests, sharing the same tools._last_quiz_problem
reveal state (see that file's module docstring for why a plain global is
enough here). Every test that depends on the state starting empty resets it
explicitly first.
"""

import tools
from tools import (
    _logic_topic_if_requested,
    _math_word_problem_topic_if_requested,
    route_reply,
)


def _reset_last_problem():
    tools._last_quiz_problem = None


# ---------------------------------------------------------------------------
# topic detection
# ---------------------------------------------------------------------------

def test_math_generic_request_detected_as_random_topic():
    assert _math_word_problem_topic_if_requested("出一題數學應用題") == "random"
    assert _math_word_problem_topic_if_requested("出一題數學") == "random"


def test_math_specific_subtopic_keywords_detected():
    assert _math_word_problem_topic_if_requested("出一題加減乘除") == "arithmetic"
    assert _math_word_problem_topic_if_requested("出一題單位換算") == "unit_conversion"
    assert _math_word_problem_topic_if_requested("出一題比例") == "ratio"
    assert _math_word_problem_topic_if_requested("出一題百分比") == "ratio"
    assert _math_word_problem_topic_if_requested("give me a problem on arithmetic") == "arithmetic"


def test_math_no_request_hint_returns_none_even_with_topic_word():
    assert _math_word_problem_topic_if_requested("我今天要交數學作業") is None
    assert _math_word_problem_topic_if_requested("數學") is None


def test_math_no_topic_word_returns_none_even_with_request_hint():
    assert _math_word_problem_topic_if_requested("考我英文單字") is None


def test_logic_generic_request_detected_as_random_topic():
    assert _logic_topic_if_requested("出一題邏輯") == "random"
    assert _logic_topic_if_requested("出一題推理") == "random"


def test_logic_specific_subtopic_keywords_detected():
    assert _logic_topic_if_requested("出一題空間推理") == "spatial"
    assert _logic_topic_if_requested("出一題演繹") == "deduction"
    assert _logic_topic_if_requested("出一題比較") == "comparison"
    assert _logic_topic_if_requested("quiz me on deduction") == "deduction"


def test_logic_no_request_hint_returns_none_even_with_topic_word():
    assert _logic_topic_if_requested("這句話邏輯不通") is None
    assert _logic_topic_if_requested("邏輯") is None


# ---------------------------------------------------------------------------
# route_reply() end-to-end quiz flow
# ---------------------------------------------------------------------------

def test_route_reply_math_quiz_returns_question_only_and_stores_state():
    _reset_last_problem()
    reason, reply = route_reply("出一題數學應用題")
    assert "數學應用題出題請求" in reason
    assert "題目" in reply
    assert tools._last_quiz_problem is not None
    assert tools._last_quiz_problem["topic"] in ("arithmetic", "unit_conversion", "ratio")
    assert tools._last_quiz_problem["answer"] not in reply
    for step in tools._last_quiz_problem["steps"]:
        assert step not in reply


def test_route_reply_logic_quiz_returns_question_only_and_stores_state():
    _reset_last_problem()
    reason, reply = route_reply("出一題邏輯推理")
    assert "邏輯推理出題請求" in reason
    assert "題目" in reply
    assert tools._last_quiz_problem is not None
    assert tools._last_quiz_problem["topic"] in ("spatial", "deduction", "comparison")
    # two-step flow: the worked-out steps are never shown at "出題" time.
    # (unlike calculus, a "spatial" question's answer room name can
    # legitimately already appear inside the narrative itself — e.g. "...
    # 走到了花園。請問...現在在哪裡？" — that's the task's premise, not a
    # spoiler, so this checks the *steps* aren't leaked rather than
    # asserting the bare answer string is absent from the question text.)
    for step in tools._last_quiz_problem["steps"]:
        assert step not in reply


def test_route_reply_solve_last_reveals_math_problem():
    _reset_last_problem()
    route_reply("出一題加減乘除")
    reason, reply = route_reply("解題")
    assert "解題」請求" in reason
    assert "答案" in reply
    assert "詳解" in reply


def test_route_reply_solve_last_reveals_logic_problem():
    _reset_last_problem()
    route_reply("出一題演繹")
    reason, reply = route_reply("解題")
    assert "解題」請求" in reason
    assert "答案" in reply


def test_route_reply_math_and_logic_quiz_headings_name_the_right_domain():
    # regression test: format_question()/format_problem() default to a
    # calculus-flavoured heading ("sinco 微積分出題：...") — tools.py must
    # override it per domain, otherwise a math/logic quiz gets mislabeled
    # as a calculus one.
    _reset_last_problem()
    _reason, math_question = route_reply("出一題數學應用題")
    assert "sinco 數學應用題出題" in math_question
    assert "微積分" not in math_question
    _reason, math_reveal = route_reply("解題")
    assert "sinco 數學應用題出題" in math_reveal
    assert "微積分" not in math_reveal

    _reset_last_problem()
    _reason, logic_question = route_reply("出一題邏輯推理")
    assert "sinco 邏輯推理出題" in logic_question
    assert "微積分" not in logic_question
    _reason, logic_reveal = route_reply("解題")
    assert "sinco 邏輯推理出題" in logic_reveal
    assert "微積分" not in logic_reveal


def test_route_reply_math_quiz_does_not_fall_through_to_search():
    _reset_last_problem()
    reason, _reply = route_reply("出一題數學應用題")
    assert "搜尋" not in reason


def test_route_reply_plain_math_mention_falls_back_to_none():
    _reset_last_problem()
    assert route_reply("我今天要交數學作業") is None


def test_route_reply_calculus_and_math_quizzes_share_reveal_state():
    # posing a calculus quiz then a math quiz then revealing should show the
    # *math* problem (the most recent one), proving both domains share one
    # _last_quiz_problem slot rather than each keeping a separate cache.
    _reset_last_problem()
    route_reply("出一題微分")
    calc_problem = tools._last_quiz_problem
    route_reply("出一題單位換算")
    math_problem = tools._last_quiz_problem
    assert math_problem is not calc_problem
    _reason, reply = route_reply("解題")
    assert math_problem["answer"] in reply
