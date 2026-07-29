"""Live data lookups for sinco — weather + web search (+ single-page fetch).

These call plain data APIs (wttr.in for weather, DuckDuckGo's Instant
Answer API for search) and, as a fallback, fetch one real webpage's text —
not another AI model — so they stay inside Rule 06: sinco's own routing
logic in route_reply() decides *when* to call them and formats the final
reply itself; no external service does the "thinking". None of this needs
an API key.

route_reply() is a small keyword/regex router, not real language
understanding — it only catches messages that look like an explicit
weather or "look this up" request (e.g. "台北天氣", "搜尋 X", "你認識 X 嗎"),
or a "出一題微積分/微分/積分/極限" -shaped request for a random calculus
problem (handled by calculus_generator.py — deterministic sympy math, not
another model, see that module's docstring for why this is inside Rule
06's "no external AI model" boundary). Anything else returns None so the
caller (see chats.py smart_reply()) falls back to the trained seq2seq
model.

fetch_page_text() is deliberately NOT a crawler (it never follows links,
never fetches more than the one page it was given) — it exists only to back
web_search() up when DuckDuckGo's Instant Answer API has nothing (it often
doesn't; it's an instant-answer box, not a real search index). Kept
memory-bounded on purpose, since a runaway fetch is the easy way to make a
32GB machine feel laggy:
  - streamed with requests(stream=True) + a hard byte cap (MAX_PAGE_BYTES),
    reading stops the moment the cap is hit instead of buffering the whole
    response first
  - Content-Length is checked before reading any body, Content-Type before
    trying to parse it, so a large non-HTML response (video, zip, ...) is
    rejected before its bytes ever land in memory
  - no headless browser (Selenium/Playwright) — those keep a whole Chromium
    process resident just to render one page, which is a much bigger and
    longer-lived memory footprint than an HTTP GET + BeautifulSoup parse
  - the parsed text is truncated to max_chars before it's returned, so
    nothing large lingers past this function call
"""

import re

import requests
from bs4 import BeautifulSoup

import calculus_generator

WEATHER_TIMEOUT = 6
SEARCH_TIMEOUT = 6
PAGE_FETCH_TIMEOUT = 8
MAX_PAGE_BYTES = 2 * 1024 * 1024  # stop reading a page past 2MB
PAGE_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; sinco-bot/1.0)"}

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

# calculus problem request: needs BOTH a "quiz me" verb and a topic keyword,
# same two-part shape as the weather/search patterns above (a bare mention
# of "微積分" with no request verb, e.g. "我要交微積分作業", should NOT fire).
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


def get_weather(location: str) -> str:
    resp = requests.get(
        f"https://wttr.in/{location}",
        params={"format": "3", "lang": "zh"},
        timeout=WEATHER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text.strip()


def fetch_page_text(url: str, max_chars: int = 3000) -> str | None:
    """Fetch a single page and return its visible text, or None on any
    failure (bad status, non-HTML content, too large, network error).

    Memory-bounded: streamed with a hard byte cap (see module docstring),
    never fetches a second page, never keeps the raw HTML around past this
    call.
    """
    try:
        with requests.get(url, timeout=PAGE_FETCH_TIMEOUT, stream=True,
                           headers=PAGE_FETCH_HEADERS) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                return None

            declared_length = resp.headers.get("Content-Length")
            if declared_length and int(declared_length) > MAX_PAGE_BYTES:
                return None

            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > MAX_PAGE_BYTES:
                    break
                chunks.append(chunk)
            raw_html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore")
    except (requests.RequestException, ValueError):
        return None

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars] if text else None


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

    # DuckDuckGo's Instant Answer API is an instant-answer box, not a real
    # search index — it comes up empty a lot (this is exactly what happened
    # earlier querying "周柯宇"). When it points at a source page but has no
    # text of its own, fall back to reading that one page directly instead
    # of giving up.
    source_url = data.get("AbstractURL") or next(
        (t.get("FirstURL") for t in (data.get("RelatedTopics") or []) if isinstance(t, dict) and t.get("FirstURL")),
        None,
    )
    if source_url:
        return fetch_page_text(source_url)
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

    calculus_topic = _calculus_topic_if_requested(message)
    if calculus_topic is not None:
        problem = calculus_generator.generate_problem(topic=calculus_topic)
        reason = (
            f'偵測到出題請求，主題「{problem["topic_zh"]}」，'
            f'呼叫 calculus_generator 即時運算產生新題目（sympy，非模型記憶）'
        )
        return reason, calculus_generator.format_problem(problem)

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
