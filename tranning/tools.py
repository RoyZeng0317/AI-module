"""Live data lookups for sinco — weather + web search.

These call plain data APIs (wttr.in for weather, DuckDuckGo's Instant
Answer API for search) — not another AI model — so they stay inside
Rule 06: sinco's own routing logic in route_reply() decides *when* to
call them and formats the final reply itself; no external service does
the "thinking". Neither API needs a key.

route_reply() is a small keyword/regex router, not real language
understanding — it only catches messages that look like an explicit
weather or "look this up" request (e.g. "台北天氣", "搜尋 X", "你認識 X 嗎").
Anything else returns None so the caller (see chats.py smart_reply())
falls back to the trained seq2seq model.
"""

import re

import requests

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
    """Return (reason, reply) for weather/search-shaped messages, else None.

    `reason` names the actual rule that fired (e.g. which keyword/pattern
    matched) so callers can show a real trace of the decision instead of a
    generic label — still just reporting what really happened, not a
    fabricated reasoning chain.
    """
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
