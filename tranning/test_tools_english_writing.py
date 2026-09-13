"""Tests for tools.route_reply()'s English-writing quiz + rule-based
grading routing — same two-step "出題 -> 解題" shape as calculus/math/logic
(see test_tools_math_logic.py's module docstring), plus an additional
"批改"/"check my answer" step unique to this domain: english_writing.py's
check_answer() rule-based-grades the user's OWN submitted answer against
the most recently posed english_writing question, since (unlike sympy for
calculus or plain arithmetic for word problems) there is no engine that can
verify an arbitrary free-form English sentence — grading is scoped to a
fixed, hand-verified item bank (see english_writing.py's module docstring).
"""

import tools
from tools import (
    _english_writing_topic_if_requested,
    _writing_submission_if_requested,
    route_reply,
)


def _reset_last_problem():
    tools._last_quiz_problem = None


# ---------------------------------------------------------------------------
# topic detection
# ---------------------------------------------------------------------------

def test_generic_request_detected_as_random_topic():
    assert _english_writing_topic_if_requested("出一題英文寫作") == "random"
    assert _english_writing_topic_if_requested("出一題英文") == "random"


def test_specific_subtopic_keywords_detected():
    assert _english_writing_topic_if_requested("出一題英文文法") == "grammar"
    assert _english_writing_topic_if_requested("出一題倒裝句") == "grammar"
    assert _english_writing_topic_if_requested("出一題假設語氣") == "grammar"
    assert _english_writing_topic_if_requested("出一題學術詞彙") == "vocabulary"
    assert _english_writing_topic_if_requested("出一題慣用語") == "vocabulary"
    assert _english_writing_topic_if_requested("出一題作文") == "essay"
    assert _english_writing_topic_if_requested("quiz me on grammar") == "grammar"
    assert _english_writing_topic_if_requested("give me a problem on idiom") == "vocabulary"


def test_no_request_hint_returns_none_even_with_topic_word():
    assert _english_writing_topic_if_requested("我今天英文作文寫不出來") is None
    assert _english_writing_topic_if_requested("英文文法") is None


def test_no_topic_word_returns_none_even_with_request_hint():
    assert _english_writing_topic_if_requested("考我數學") is None


# ---------------------------------------------------------------------------
# writing-submission prefix extraction
# ---------------------------------------------------------------------------

def test_writing_submission_prefix_extraction():
    assert _writing_submission_if_requested("批改：Never have I seen this.") == "Never have I seen this."
    assert _writing_submission_if_requested("check my answer: consensus") == "consensus"
    assert _writing_submission_if_requested("這只是普通對話") is None


def test_writing_submission_longest_prefix_wins():
    assert _writing_submission_if_requested("批改我的答案：consensus") == "consensus"


def test_writing_submission_empty_after_prefix_returns_empty_string():
    assert _writing_submission_if_requested("批改") == ""


# ---------------------------------------------------------------------------
# route_reply() end-to-end quiz flow
# ---------------------------------------------------------------------------

def test_route_reply_quiz_returns_question_only_and_stores_state():
    _reset_last_problem()
    reason, reply = route_reply("出一題英文文法")
    assert "英文寫作出題請求" in reason
    assert "題目" in reply
    assert tools._last_quiz_problem is not None
    assert tools._last_quiz_problem["topic"] == "grammar"
    assert tools._last_quiz_problem["answer"] not in reply


def test_route_reply_solve_last_reveals_english_writing_problem():
    _reset_last_problem()
    route_reply("出一題英文文法")
    reason, reply = route_reply("解題")
    assert "解題」請求" in reason
    assert "答案" in reply


def test_route_reply_heading_names_the_right_domain():
    _reset_last_problem()
    _reason, question = route_reply("出一題英文文法")
    assert "sinco 英文寫作出題" in question
    assert "微積分" not in question
    _reason, reveal = route_reply("解題")
    assert "sinco 英文寫作出題" in reveal


def test_route_reply_quiz_does_not_fall_through_to_search():
    _reset_last_problem()
    reason, _reply = route_reply("出一題英文文法")
    assert "搜尋" not in reason


def test_route_reply_plain_english_mention_falls_back_to_none():
    _reset_last_problem()
    assert route_reply("我今天英文作文寫不出來") is None


# ---------------------------------------------------------------------------
# route_reply() grading ("批改") flow
# ---------------------------------------------------------------------------

def test_route_reply_grading_without_active_quiz():
    _reset_last_problem()
    reason, reply = route_reply("批改：Never have I seen this.")
    assert "沒有正在進行" in reason
    assert "沒有正在進行" in reply


def test_route_reply_grading_correct_grammar_answer():
    _reset_last_problem()
    route_reply("出一題倒裝句")
    problem = tools._last_quiz_problem
    reason, reply = route_reply(f"批改：{problem['answer']}")
    assert "批改請求" in reason
    assert "正確" in reply


def test_route_reply_grading_wrong_grammar_answer():
    _reset_last_problem()
    route_reply("出一題倒裝句")
    reason, reply = route_reply("批改：I saw a beautiful sunset yesterday.")
    assert "批改請求" in reason
    assert "參考答案" in reply


def test_route_reply_grading_missing_answer_text():
    _reset_last_problem()
    route_reply("出一題英文文法")
    reason, reply = route_reply("批改")
    assert "沒有附上" in reason


def test_route_reply_grading_after_non_english_writing_quiz():
    # posing a calculus quiz then trying to "批改" should say there's no
    # english-writing question in progress, not crash trying to grade a
    # calculus problem with english_writing's checker.
    _reset_last_problem()
    route_reply("出一題微分")
    reason, reply = route_reply("批改：Never have I seen this.")
    assert "沒有正在進行" in reason
    assert "沒有正在進行" in reply
