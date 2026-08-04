"""Live data lookups for sinco — weather + stock + web search + calculus + image.

The weather/stock/search lookups call plain data APIs (wttr.in for weather,
TWSE's public MIS quote API for the TAIEX index, DuckDuckGo's Instant Answer
API for search) — not another AI model — so they stay inside Rule 06: sinco's
own routing logic in route_reply() decides *when* to call them and formats
the final reply itself; no external service does the "thinking". None of
these APIs need a key.

The calculus routing (quiz / reveal-last / free-form solve) is the same
idea applied to calculus_generator.py/calculus_solver.py instead of a web
API — real sympy computation, not a model, decided and formatted by
route_reply() itself. See those two modules' docstrings for why that's
inside Rule 06's boundary and how the actual math works.

The image routing (recognize_image()) reuses web/backend/detector.py's
local, self-trained-or-transfer-learned YOLO model (the same one
app/components/camera.py already points at live camera frames) — this file
just gives it a second input path: a local file or a plain http(s) image
URL, decided and formatted by route_reply() itself, same Rule 06 boundary
as everything else here (the "thinking" is a local model call, not a cloud
vision API).

route_reply() is a small keyword/regex router, not real language
understanding — it only catches messages that look like an explicit
weather/search request (e.g. "台北天氣", "搜尋 X", "你認識 X 嗎"), an
explicit calculus quiz/solve request (e.g. "出一題微積分", "解題",
"3x^2 的微分"), or an explicit image-recognition request (e.g. "辨識圖片
<路徑或網址>"). As a last resort before giving up, it also catches a few
implicit "do you know about X" phrasings (e.g. "你會微積分嗎", "什麼是
微積分") that don't already have a trained reply in data/pairs.json — same
DuckDuckGo lookup, and the result gets stashed as an auto_learn.py candidate
for you to review with `/learn` rather than silently retraining anything.
Anything else still returns None so the caller (see chats.py smart_reply())
falls back to the trained seq2seq model.
"""

import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

import auto_learn
import calculus_generator
import calculus_solver

_PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "pairs.json"
_WEB_BACKEND_DIR = Path(__file__).resolve().parent.parent / "web" / "backend"

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
# "那你知道 X 嗎"／"欸你認識 X 嗎" 這類前面加了口語連接詞的問法，本來會因為
# 上面每一條 pattern 都用 ^ 錨定開頭而直接比對失敗（"那" 卡在 "你" 前面），
# 整句掉回死記式聊天模型硬答一句文不對題的內容——實測抓到："你知道token嗎"
# 能正確觸發搜尋，但只多了開頭一個「那」字的「那你知道張凌赫嗎」就比對不到。
# 在真正丟進 _SEARCH_PATTERNS/_KNOWLEDGE_QUESTION_PATTERNS 之前，先去掉這些
# 純語氣連接詞前綴，不影響 auto_learn 存檔用的原始 message。
_LEADING_FILLERS = re.compile(r"^(?:那麼|那|想請問|請問|所以|欸|誒|嗯|話說|對了|順便問一下|順便問)+\s*")


def _strip_leading_filler(message: str) -> str:
    return _LEADING_FILLERS.sub("", message, count=1)
# subjects that mean "sinco itself" — let the trained identity pairs answer
# those instead of firing a pointless web search for "你"/"you".
_SELF_REFERENCE = {"你", "妳", "你們", "sinco", "the", "you"}

# --- implicit "do you know about X" fallback --------------------------------
# CLAUDE.md to-do: 使用者問「你會微積分嗎」時，這句沒有比對到上面任何句型，
# 掉進死記式聊天模型（只有 ~29 筆訓練資料）給出文不對題的亂碼回覆。這裡補上
# 幾種常見的「問 sinco 知不知道/懂不懂 X」問法，一樣呼叫上面 web_search()
# 同一個 DuckDuckGo 資料源（不是新的資料來源，只是多幾種句型觸發它），並把
# 查到的結果存成候選訓練資料（auto_learn.py），之後由你在 /learn 指令審核、
# 確認後才會真的進到 data/pairs.json、由你手動重訓——不會自動學習。
#
# 獨立於 _SEARCH_PATTERNS 之外（不合併進同一個清單），因為這裡多了一個
# _SEARCH_PATTERNS 沒有的守門機制：_is_known_trained_prompt()。理由是這批
# 句型「你會 X 嗎」跟訓練資料裡已經有的 "你會說英文嗎"（sinco 自己能力的
# 固定回覆）長得一模一樣，若不排除掉，"你會說英文嗎" 會被誤導去搜尋
# 「說英文」而不是用原本就答得很好的訓練回覆。
_KNOWLEDGE_QUESTION_PATTERNS = [
    re.compile(r"^你(?:會|懂|了解)\s*(.+?)\s*嗎[!?？]*$"),
    re.compile(r"^什麼是\s*(.+?)[!?？]*$"),
]


def _is_known_trained_prompt(message: str) -> bool:
    """True if `message`（trim/lower 後）跟 data/pairs.json 裡某一筆訓練
    資料的 prompt 完全一致——代表 sinco 已經有專門訓練過的回覆，不該被下面
    的知識詢問句型搶走去查網路。不快取，每次重新讀檔：`/learn approve` 可能
    在同一次執行過程中把新的 prompt 加進 pairs.json，快取住的話當次執行
    session 抓不到剛核准的新資料。
    """
    try:
        pairs = json.loads(_PAIRS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    normalized = message.strip().lower()
    return any(p.get("prompt", "").strip().lower() == normalized for p in pairs)

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


# --- stock: 台股加權指數（TAIEX）------------------------------------------
# 同 get_weather() 的單關鍵字判斷方式（"台股"/"大盤"/"加權指數" 幾乎不會出現
# 在非查詢語境，不需要像影片搜尋那樣的「動詞+關鍵字」兩段式判斷）。資料源是
# 台灣證券交易所（TWSE）自己公開的即時行情 API（mis.twse.com.tw），不需要
# 金鑰、純資料查詢，符合 Rule 06。ex_ch=tse_t00.tw 是「發行量加權股價指數」
# 這個特定代碼（不是個股），實測直接 GET 就有回應，不需要先取得 session
# cookie；帶上 `_`（毫秒時間戳）避免中間層快取回傳過期報價。
STOCK_TIMEOUT = 6
_STOCK_KEYWORD = re.compile(r"台股|大盤|加權指數|加权指数")


def get_taiex() -> str:
    """查詢台股加權指數（TAIEX）最新報價，回傳格式化好的中文字串。查詢成功
    但沒有資料（例如非交易日、TWSE 那端回傳格式有變動）丟 ValueError，由
    route_reply() 接住並回覆查詢失敗訊息——跟 get_weather() 只靠
    requests.RequestException 判斷失敗不同，這裡多了「請求成功但沒有可用
    資料」這種失敗型態，所以額外用 ValueError 涵蓋。
    """
    resp = requests.get(
        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
        params={"ex_ch": "tse_t00.tw", "_": str(int(time.time() * 1000))},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=STOCK_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json().get("msgArray") or []
    if not rows:
        raise ValueError("查無台股加權指數資料")

    row = rows[0]
    try:
        price, prev_close = float(row["z"]), float(row["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("台股加權指數資料格式異常") from exc

    change = price - prev_close
    pct = change / prev_close * 100 if prev_close else 0.0
    sign = "+" if change >= 0 else ""
    date = row.get("d", "")
    if len(date) == 8:
        date = f"{date[:4]}/{date[4:6]}/{date[6:]}"

    return (
        f"{row.get('n', '台股加權指數')}（{date}）：{price:.2f} 點，"
        f"較前一交易日 {sign}{change:.2f} 點（{sign}{pct:.2f}%）"
    )


def _fetch_abstract(query: str) -> dict:
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _expand_truncated_related_topic(first_url: str) -> str | None:
    """DuckDuckGo 對模糊/有歧義的詞（例如查 "Tokenization"，同時撞到
    "Tokenization (data security)" 跟 "Tokenism" 兩個條目）會回傳空的
    AbstractText，改把每個候選塞進 RelatedTopics，而且每一則自己就先截斷成
    一句「...」結尾的預覽（不是我們的程式碼截斷的——這是 DuckDuckGo API
    本身回傳的資料就長這樣，實測過 AbstractText 是空字串）。RelatedTopics
    的 FirstURL 帶著消歧義後的完整詞條名（例如
    ".../Tokenization_(data_security)"），拿這個更精確的詞重新查一次，通常
    就能拿到完整的 AbstractText，而不是這句話說到一半的預覽。查失敗（網路
    錯誤或這次還是沒有 AbstractText）就回傳 None，呼叫端會改用原本那句
    截斷預覽，不會讓使用者什麼都沒看到。
    """
    slug = first_url.rstrip("/").rsplit("/", 1)[-1]
    title = urllib.parse.unquote(slug).replace("_", " ")
    try:
        data = _fetch_abstract(title)
    except requests.RequestException:
        return None
    return data.get("AbstractText") or None


def web_search(query: str) -> str | None:
    data = _fetch_abstract(query)
    if data.get("AbstractText"):
        return data["AbstractText"]
    for topic in data.get("RelatedTopics") or []:
        if not (isinstance(topic, dict) and topic.get("Text")):
            continue
        text = topic["Text"]
        if text.endswith("...") and topic.get("FirstURL"):
            expanded = _expand_truncated_related_topic(topic["FirstURL"])
            if expanded:
                return expanded
        return text
    return None


def wikipedia_search(query: str) -> str | None:
    """中文維基百科摘要 API——備援資料源。DuckDuckGo Instant Answer 主要是
    英文/維基百科導向的資料，中文詞條常常查無結果（實測「微積分」「台北101」
    這類常見中文詞 web_search() 都查不到，但英文 "Calculus" 查得到），這裡
    補上查無結果時的第二個嘗試。一樣是純資料 API，不是 AI 模型，不需要金鑰，
    符合 Rule 06。必須帶 User-Agent：Wikimedia 的 API 政策會直接擋掉沒有
    標頭的請求（回 403），不是查無此詞條。
    """
    resp = requests.get(
        f"https://zh.wikipedia.org/api/rest_v1/page/summary/{query}",
        headers={"User-Agent": "sinco-AI-module/1.0 (local desktop app; no contact URL)"},
        timeout=SEARCH_TIMEOUT,
    )
    if resp.status_code != 200:
        return None
    extract = resp.json().get("extract")
    return extract.strip() if extract else None


def _lookup(subject: str) -> tuple[str | None, str]:
    """依序試 DuckDuckGo（web_search，英文/國際詞條較強）、查無結果再試中文
    維基百科（wikipedia_search，中文詞條較強）。回傳 (結果或 None, 實際命中
    的資料源代號)——找不到時 source 固定回 "duckduckgo"（第一個嘗試的來源，
    純粹當個預設值，不影響任何邏輯，因為此時 result 是 None 不會被拿去用）。
    """
    try:
        result = web_search(subject)
    except requests.RequestException:
        result = None
    if result is not None:
        return result, "duckduckgo"

    try:
        result = wikipedia_search(subject)
    except requests.RequestException:
        result = None
    return result, "wikipedia"


_SOURCE_LABELS = {"duckduckgo": "DuckDuckGo", "wikipedia": "中文維基百科"}

# --- image recognition: "辨識圖片 <本機路徑或網址>" ---------------------------
IMAGE_TIMEOUT = 10
_IMAGE_PREFIXES = (
    "辨識圖片", "辨識圖像", "識別圖片", "識別圖像", "看看這張圖片", "看看這張圖",
    "recognize image", "recognise image", "what's in this image", "whats in this image",
)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def _image_source_if_requested(message: str) -> str | None:
    """None unless `message` starts with one of _IMAGE_PREFIXES followed by a
    local path or http(s) URL — returns that path/URL, else None. A bare
    mention of "圖片"/"image" with no prefix should NOT fire (same two-part
    shape as the weather/calculus request checks above).
    """
    text = message.strip()
    lower = text.lower()
    for prefix in _IMAGE_PREFIXES:
        if lower.startswith(prefix.lower()):
            source = text[len(prefix):].strip(" :：")
            return source or None
    return None


def _load_image_bgr(source: str):
    """回傳 (image, error)：image 是 cv2 讀進來的 BGR ndarray，成功時
    error 是 None；失敗時 image 是 None、error 是給使用者看的中文訊息。
    本機路徑跟 http(s) 網址都支援——網址的部分只下載圖片本身的 bytes 交給
    本機的 YOLO 模型判讀，不是把圖片丟給任何雲端視覺 API，一樣符合 Rule 06。
    """
    import cv2
    import numpy as np

    if source.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(source, timeout=IMAGE_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            return None, f"圖片下載失敗：{exc}"
        image = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            return None, f"找不到圖片檔案：{source}"
        image = cv2.imread(str(path))

    if image is None:
        return None, "無法解析這個檔案，確認它是有效的圖片格式（jpg/png/bmp/webp/gif）。"
    return image, None


def recognize_image(source: str) -> str:
    """對 `source`（本機路徑或 http(s) 網址）跑本機 YOLO 偵測，回傳中文格式化
    結果。沿用 web/backend/detector.py 同一個 detect()——跟
    app/components/camera.py 即時攝影機用的是同一個模型，只是這裡的輸入是
    單張靜態圖片而非連續影格。sys.path 手動補 web/backend 的路徑，因為
    tools.py 可能在沒有先跑過 home_screen.py/cli.py（它們自己的進入點才會
    加這個路徑）的情況下被單獨測試或呼叫。
    """
    if str(_WEB_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_WEB_BACKEND_DIR))
    from detector import detect

    image, error = _load_image_bgr(source)
    if error is not None:
        return error
    assert image is not None  # _load_image_bgr 的合約：error 是 None 時 image 一定不是 None

    try:
        detections = detect(image, conf=0.35)
    except Exception as exc:  # ultralytics/模型端任何未預期錯誤都要看得到原因
        return f"圖片辨識失敗：{exc}"

    if not detections:
        return "沒有在這張圖片裡偵測到任何已知物件。"

    items = "、".join(f'{d["label"]}（信心度 {d["confidence"]:.0%}）' for d in detections[:10])
    return f"偵測到 {len(detections)} 個物件：{items}"


# --- video search: "找...的影片" / "搜尋影片 X" / "YouTube 搜尋 X" ------------
# 直接發送請求到 YouTube 搜尋結果頁——實測過會擋自動化請求的是百度百科
# （403＋JS 驗證碼，見上方 wikipedia_search() 旁的教訓），YouTube 搜尋結果頁
# 沒有這層阻擋，一般 requests.get() 就能拿到 200＋內嵌的 ytInitialData JSON，
# 一樣是純資料擷取，不是呼叫任何 AI 摘要服務，符合 Rule 06。
YOUTUBE_TIMEOUT = 10
_YOUTUBE_KEYWORD = re.compile(r"影片|視頻|video|youtube", re.IGNORECASE)
_YOUTUBE_REQUEST_HINTS = ("找", "搜尋", "查一下", "查詢", "search", "find")
_YOUTUBE_INITIAL_DATA = re.compile(r"var ytInitialData = (\{.*?\});</script>")


def _video_query_if_requested(message: str) -> str | None:
    """None unless the message both (a) reads as a "find/search" request and
    (b) mentions 影片/視頻/video/youtube——同一種「兩件事都要有」的兩段式判斷
    （同上方天氣/微積分/圖片的設計），避免「我剛看了一部很好看的電影」這種
    沒有搜尋意圖的句子被誤判。找到就把 hint 詞跟關鍵字都剝掉，剩下的當查詢字。
    """
    text = message.strip()
    lower = text.lower()
    if not _YOUTUBE_KEYWORD.search(text):
        return None
    if not any(hint in lower for hint in _YOUTUBE_REQUEST_HINTS):
        return None

    query = text
    for hint in sorted(_YOUTUBE_REQUEST_HINTS, key=len, reverse=True):
        query = re.sub(re.escape(hint), "", query, flags=re.IGNORECASE)
    query = _YOUTUBE_KEYWORD.sub("", query)
    query = query.strip(" 的:：")
    return query or None


def _youtube_result_text(video: dict) -> str:
    title = "".join(r.get("text", "") for r in video.get("title", {}).get("runs", []))
    channel = "".join(
        r.get("text", "")
        for r in video.get("ownerText", {}).get("runs", []) or video.get("longBylineText", {}).get("runs", [])
    )
    duration = video.get("lengthText", {}).get("simpleText", "直播/未知長度")
    views = (video.get("viewCountText") or video.get("shortViewCountText") or {}).get("simpleText", "")
    snippets = video.get("detailedMetadataSnippets") or []
    summary = "".join(r.get("text", "") for r in snippets[0].get("snippetText", {}).get("runs", [])) if snippets else ""

    line = f"《{title}》（{channel}，{duration}"
    if views:
        line += f"，{views}"
    line += f"）\n連結：https://www.youtube.com/watch?v={video.get('videoId')}"
    if summary:
        line += f"\n重點：{summary}"
    return line


def youtube_search(query: str, max_results: int = 3) -> str | None:
    """搜尋 YouTube，回傳前 `max_results` 部影片的整理結果（標題／頻道／長度／
    觀看次數／連結，有簡介片段的話也一併附上）。找不到結果或請求失敗回傳
    None，交由呼叫端（route_reply()）決定要顯示什麼訊息。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "zh-TW,zh;q=0.9",
    }
    try:
        resp = requests.get(
            "https://www.youtube.com/results",
            params={"search_query": query},
            headers=headers,
            timeout=YOUTUBE_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    match = _YOUTUBE_INITIAL_DATA.search(resp.text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    videos: list[dict] = []

    def walk(node):
        if len(videos) >= max_results:
            return
        if isinstance(node, dict):
            if "videoRenderer" in node:
                videos.append(node["videoRenderer"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    if not videos:
        return None
    return "\n\n".join(_youtube_result_text(v) for v in videos[:max_results])


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

    stock_match = _STOCK_KEYWORD.search(message)
    if stock_match:
        reason = f'偵測到關鍵字「{stock_match.group()}」，查詢台灣證交所（TWSE）即時行情 API'
        try:
            return reason, get_taiex()
        except (requests.RequestException, ValueError):
            return reason, "抱歉，台股行情查詢暫時失敗，請稍後再試。"

    image_source = _image_source_if_requested(message)
    if image_source is not None:
        reason = f'偵測到圖片辨識請求，來源「{image_source}」，呼叫本機 YOLO 模型（web/backend/detector.py）'
        return reason, recognize_image(image_source)

    video_query = _video_query_if_requested(message)
    if video_query is not None:
        reason = f'偵測到影片搜尋請求，查詢主題「{video_query}」，查詢 YouTube 搜尋結果'
        result = youtube_search(video_query)
        return reason, (result or f"抱歉，沒有找到「{video_query}」的相關影片。")

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

    stripped_message = _strip_leading_filler(message)
    for pattern in _SEARCH_PATTERNS:
        m = pattern.match(stripped_message)
        if not m:
            continue
        subject = m.group(1).strip().strip("的?？!！")
        if not subject or subject.lower() in _SELF_REFERENCE:
            return None
        result, source = _lookup(subject)
        reason = f'比對到搜尋句型，查詢主題「{subject}」，查詢 {_SOURCE_LABELS[source]}'
        if result is not None:
            auto_learn.save_candidate(prompt=message, reply=result, topic=subject, source=source)
        return reason, (result or f"抱歉，沒有找到「{subject}」的相關資料。")

    # 上面都沒比對到、也還不是死心的時候：試著把這句話當成「問 sinco 知不知道
    # X」的隱含查詢句型，而不是直接讓死記式聊天模型硬答一句文不對題的內容。
    # 跟上面的 _SEARCH_PATTERNS 不同，這裡找不到結果就放棄（continue／最後
    # return None），不會回一句「抱歉沒找到」——使用者沒有明講要搜尋，這種
    # 情況安靜地退回一般聊天模型（sinco 至少會用 hi/嗨 之類的閒聊資料接住），
    # 好過為了每一句猜測性查詢都跳一句道歉訊息。
    if not _is_known_trained_prompt(message):
        for pattern in _KNOWLEDGE_QUESTION_PATTERNS:
            m = pattern.match(stripped_message)
            if not m:
                continue
            subject = m.group(1).strip().strip("的?？!！")
            if not subject or subject.lower() in _SELF_REFERENCE:
                continue
            result, source = _lookup(subject)
            if result is None:
                continue
            auto_learn.save_candidate(prompt=message, reply=result, topic=subject, source=source)
            reason = (
                f'沒有比對到現成訓練回覆，判斷為知識詢問句型，主題「{subject}」，'
                f'自動查詢 {_SOURCE_LABELS[source]} 並存為候選訓練資料（/learn 審核）'
            )
            return reason, result

    return None
