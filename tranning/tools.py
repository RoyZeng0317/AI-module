"""Live data lookups for sinco — weather + web search + calculus.

The weather/search lookups call plain data APIs (wttr.in for weather,
DuckDuckGo's Instant Answer API for search) — not another AI model — so
they stay inside Rule 06: sinco's own routing logic in route_reply()
decides *when* to call them and formats the final reply itself; no
external service does the "thinking". Neither API needs a key.

The calculus routing (quiz / reveal-last / free-form solve) is the same
idea applied to calculus_generator.py/calculus_solver.py instead of a web
API — real sympy computation, not a model, decided and formatted by
route_reply() itself. See those two modules' docstrings for why that's
inside Rule 06's boundary and how the actual math works.

route_reply() is a small keyword/regex router, not real language
understanding — it only catches messages that look like an explicit
weather/search request (e.g. "台北天氣", "搜尋 X", "你認識 X 嗎") or an
explicit calculus quiz/solve request (e.g. "出一題微積分", "解題",
"3x^2 的微分"). Anything else returns None so the caller (see chats.py
smart_reply()) falls back to the trained seq2seq model.
"""

import re

import requests

import calculus_generator
import calculus_solver

WEATHER_TIMEOUT = 6
SEARCH_TIMEOUT = 6

_WEATHER_KEYWORD = re.compile(r"天氣")
_WEATHER_PREFIXES = ("幫我查詢", "幫我查", "查詢", "查一下", "查")

_SEARCH_PATTERNS = [
    re.compile(r"^(?:搜尋|search)\s*[:：]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:查詢|查一下|幫我查)\s*(.+)$"),
    re.compile(r"^你(?:認識|知道)\s*(.+?)\s*嗎[!?？]*$"),
    re.compile(r"^(?:do you know|who is|what is)\s+(.+?)[!?？]*$", re.IGNORECASE),
    re.compile(r"^(.+?)是誰[!?？]*$"),
]
# subjects that mean "sinco itself" — let the trained identity pairs answer
# those instead of firing a pointless web search for "你"/"you".
_SELF_REFERENCE = {"你", "妳", "你們", "sinco", "the", "you"}

# --- calculus: quiz request ("出一題微積分") -------------------------------
# needs BOTH a "quiz me" verb and a topic keyword, same two-part shape as
# the weather/search patterns above (a bare mention of "微積分" with no
# request verb, e.g. "我要交微積分作業", should NOT fire).
_CALCULUS_REQUEST_HINTS = (
    "出一題", "出題", "來一題", "考我", "給我一題", "出个题",
    "quiz me", "give me a problem", "practice",
)
# checked in this order: specific subtopic keywords first, generic "微積分"
# last — "積分" is a literal substring of "微積分" (微/積/分), so if the
# generic phrase were stripped out of the message *after* checking "積分"
# instead of before, "出一題微積分" (no explicit subtopic) would always be
# misread as an integral-only request instead of "pick any of the three".
_CALCULUS_TOPIC_KEYWORDS = (
    ("導數", "derivative"), ("求導", "derivative"), ("differentiate", "derivative"),
    ("極限", "limit"), ("limit", "limit"),
    ("微分", "derivative"), ("derivative", "derivative"),
    ("積分", "integral"), ("integral", "integral"), ("integrate", "integral"),
)

# --- calculus: reveal the previously posed quiz question ("解題") ---------
_SOLVE_LAST_HINTS = {
    "解題", "解答案", "公佈答案", "看答案", "揭曉答案", "解答",
    "show answer", "reveal answer", "solve it",
}

# module-level "last quiz question" cache: this is a single-user desktop
# app (one Tk process, no concurrent sessions), so a plain module global is
# enough state to support "出題 -> 解題" as two separate chat turns without
# threading a session object through chats.py's otherwise-stateless
# smart_reply_traced()/route_reply() call chain.
_last_calculus_problem: dict | None = None


def _calculus_topic_if_requested(message: str) -> str | None:
    """None unless the message both (a) reads as a "quiz me" request and
    (b) names a calculus topic — returns "derivative"/"integral"/"limit"
    for an explicit subtopic, or "random" for a generic "微積分"/"calculus"
    mention with no subtopic (calculus_generator.generate_problem() then
    picks one at call time).
    """
    text = message.strip().lower()
    if not any(hint in text for hint in _CALCULUS_REQUEST_HINTS):
        return None

    remaining = text.replace("微積分", "").replace("calculus", "")
    mentioned_generic = remaining != text
    for keyword, topic in _CALCULUS_TOPIC_KEYWORDS:
        if keyword in remaining:
            return topic
    return "random" if mentioned_generic else None


def _is_solve_last_request(message: str) -> bool:
    normalized = message.strip().strip("。！？!?").lower()
    return normalized in {h.lower() for h in _SOLVE_LAST_HINTS}


def get_weather(location: str) -> str:
    resp = requests.get(
        f"https://wttr.in/{location}",
        params={"format": "3", "lang": "zh"},
        timeout=WEATHER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text.strip()


def web_search(query: str) -> str | None:
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("AbstractText"):
        return data["AbstractText"]
    for topic in data.get("RelatedTopics") or []:
        if isinstance(topic, dict) and topic.get("Text"):
            return topic["Text"]
    return None


def _extract_location(message: str) -> str:
    text = message
    for prefix in _WEATHER_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip()
    if text.endswith("天氣"):
        text = text[: -len("天氣")]
    return text.rstrip("的").strip() or "Taipei"


def route_reply(message: str) -> tuple[str, str] | None:
    """Return (reason, reply) for weather/search/calculus-shaped messages,
    else None.

    `reason` names the actual rule that fired (e.g. which keyword/pattern
    matched) so callers can show a real trace of the decision instead of a
    generic label — still just reporting what really happened, not a
    fabricated reasoning chain.
    """
    global _last_calculus_problem

    message = message.strip()
    if not message:
        return None

    if _WEATHER_KEYWORD.search(message):
        location = _extract_location(message)
        reason = f'偵測到關鍵字「天氣」，判斷地點為「{location}」，查詢 wttr.in'
        try:
            return reason, f"{location} 目前天氣：{get_weather(location)}"
        except requests.RequestException:
            return reason, "抱歉，天氣查詢暫時失敗，請稍後再試。"

    if _is_solve_last_request(message):
        if _last_calculus_problem is None:
            reason = "偵測到「解題」請求，但目前沒有已出的題目"
            return reason, "目前還沒有出過題目，請先輸入「出一題微積分」之類的指令。"
        reason = f'偵測到「解題」請求，公佈上一題答案，主題「{_last_calculus_problem["topic_zh"]}」'
        return reason, calculus_generator.format_problem(_last_calculus_problem)

    calculus_topic = _calculus_topic_if_requested(message)
    if calculus_topic is not None:
        problem = calculus_generator.generate_problem(topic=calculus_topic)
        _last_calculus_problem = problem
        reason = (
            f'偵測到出題請求，主題「{problem["topic_zh"]}」，'
            f'呼叫 calculus_generator 即時運算產生新題目（sympy，非模型記憶）'
        )
        return reason, calculus_generator.format_question(problem)

    try:
        solved = calculus_solver.parse_and_solve(message)
    except calculus_solver.SolveError as exc:
        reason = "偵測到解題請求，但 sympy 無法求出封閉形式的解"
        return reason, f"看得懂這題，但 sinco（sympy）目前算不出封閉形式的解：{exc}。可以換一題試試。"
    if solved is not None:
        _last_calculus_problem = solved
        reason = f'偵測到自由輸入的解題請求，主題「{solved["topic_zh"]}」，呼叫 calculus_solver 解析並用 sympy 計算'
        return reason, calculus_generator.format_problem(solved, heading=f"sinco 解題：{solved['topic_zh']}")

    for pattern in _SEARCH_PATTERNS:
        m = pattern.match(message)
        if not m:
            continue
        subject = m.group(1).strip().strip("的?？!！")
        if not subject or subject.lower() in _SELF_REFERENCE:
            return None
        reason = f'比對到搜尋句型，查詢主題「{subject}」，查詢 DuckDuckGo'
        try:
            result = web_search(subject)
        except requests.RequestException:
            result = None
        return reason, (result or f"抱歉，沒有找到「{subject}」的相關資料。")

    return None
