"""PyScript 按鈕功能列 + 完整前端 UI。

在瀏覽器中由 PyScript (Pyodide) 執行，建構整個頁面的 DOM，
包含頂部功能列、物件偵測區塊、對話區塊，並透過 fetch 呼叫後端 API。
"""

from pyscript import document, window
from js import console, WebSocket, Object
from js import URL as _JsURL
from pyodide.ffi import create_proxy, to_js
import asyncio
import json as _json
import re as _re

# ═══════════════════════════════════════════════════════════════
# 設定
# ═══════════════════════════════════════════════════════════════
# 留空字串代表「後端跟這個頁面同網域」——這時 API 路徑一律用**沒有開頭斜線
# 的相對路徑**（"api/health" 而不是 "/api/health"），瀏覽器會自動相對於
# 目前頁面的網址去解析。這樣不管這個頁面本身是掛在網域根目錄（本機直接
# python app.py、Render.com）還是掛在子路徑下（Tailscale Serve 的
# `--set-path=/AI-Module`，見 前端部屬scp.md），都不用改這支檔案——後端
# `web/backend/app.py` 用同一套 MOUNT_PREFIX 環境變數掛同一個前綴，兩邊
# 路徑會自動對齊。只有後端跟前端不同主機時才需要把 BACKEND_URL 填成完整
# 網址（此時就一定是網域根目錄，不會有子路徑問題）。
BACKEND_URL = ""


def _api_url(path):
    if BACKEND_URL:
        return f"{BACKEND_URL}/{path}"
    return path


HEALTH_URL = _api_url("api/health")
CHAT_URL = _api_url("api/chat")
GPU_INFO_URL = _api_url("api/gpu_info")
CONVERSATIONS_URL = _api_url("api/conversations")
NOTIFY_URL = _api_url("api/notifications")
DETECT_INTERVAL_MS = 500
CAPTURE_WIDTH = 640


def _detect_ws_url():
    """算出 /ws/detect 的 ws(s) 網址。

    原本借 `new URL("ws/detect", 目前頁面網址)` 做相對路徑解析，但這個
    建構子要求「base」必須是瀏覽器判定為可以當 base 的絕對網址——只要目前
    頁面不是用一般 http(s) 網址「導覽」進來的（例如某些預覽/沙盒環境把頁面
    塞進 `about:srcdoc` 的 iframe），`window.location.href` 就會是不能當
    base 的網址，建構子會直接丟出
    `TypeError: Failed to construct 'URL': Invalid base URL`，讓整支
    action.py 在載入階段就當掉。改成直接用 `window.location` 的
    protocol/host/pathname 自己拼字串，不經過會失敗的相對路徑解析，
    對一般 http(s) 頁面行為完全相同（照樣會自動帶上子路徑前綴，例如
    Tailscale Serve 的 /AI-Module/），但不會在奇怪的頁面情境下整個炸掉。
    """
    if BACKEND_URL:
        parsed = _JsURL.new(BACKEND_URL)
        scheme = str(parsed.protocol).rstrip(":")
        host = str(parsed.host)
        prefix = str(parsed.pathname)
    else:
        loc = window.location
        scheme = str(loc.protocol).rstrip(":")
        host = str(loc.host)
        path = str(loc.pathname)
        prefix = path.rsplit("/", 1)[0] + "/" if "/" in path else "/"
    if not prefix.endswith("/"):
        prefix += "/"
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{host}{prefix}ws/detect"


DETECT_WS_URL = _detect_ws_url()

# ═══════════════════════════════════════════════════════════════
# SVG 圖示
# ═══════════════════════════════════════════════════════════════
ICONS = {
    "logo": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 8a2 2 0 0 1 2-2h1.5l1-1.5h7l1 1.5H18a2 2 0 0 1 2 2v9'
        'a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/>'
        '<circle cx="12" cy="13" r="3.3"/></svg>'
    ),
    "camera": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 8a2 2 0 0 1 2-2h1.5l1-1.5h7l1 1.5H18a2 2 0 0 1 2 2v9'
        'a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/>'
        '<circle cx="12" cy="13" r="3.3"/></svg>'
    ),
    "cameraOff": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 3l18 18"/>'
        '<path d="M9.5 4.5H15l1 1.5H18a2 2 0 0 1 2 2v9c0 .4-.09.77-.25 1.1"/>'
        '<path d="M4 8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h11c.5 0 .95-.15 1.33-.4"/>'
        '<circle cx="12" cy="13" r="3.3"/></svg>'
    ),
    "chat": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 5h16v11H8l-4 4Z"/></svg>'
    ),
    "send": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 2 11 13"/>'
        '<path d="M22 2 15 22l-4-9-9-4Z"/></svg>'
    ),
    "model": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<path d="M9 9h6v6H9z"/>'
        '<path d="M9 1v3M15 1v3M9 20v3M15 20v3"/>'
        '<path d="M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>'
    ),
    "character": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="8" r="4"/>'
        '<path d="M4 21v-1a6 6 0 0 1 12 0v1"/></svg>'
    ),
    "memory": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5v6c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
        '<path d="M3 11v6c0 1.66 4 3 9 3s9-1.34 9-3v-6"/></svg>'
    ),
    "learn": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
        '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>'
    ),
    "clear": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 6h18"/>'
        '<path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>'
        '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>'
    ),
    "dots": (
        '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none">'
        '<circle cx="12" cy="5" r="1.8"/>'
        '<circle cx="12" cy="12" r="1.8"/>'
        '<circle cx="12" cy="19" r="1.8"/></svg>'
    ),
    "edit": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 20h9"/>'
        '<path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>'
    ),
    "settings": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="3.3"/>'
        '<path d="M19.4 13.5a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
        'a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.04 1.56V19.6a2 2 0 1 1-4 0v-.09'
        'a1.7 1.7 0 0 0-1.11-1.56 1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06'
        'a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1.04H4.4a2 2 0 1 1 0-4h.09'
        'a1.7 1.7 0 0 0 1.56-1.11 1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06'
        'a1.7 1.7 0 0 0 1.87.34H10.5a1.7 1.7 0 0 0 1.04-1.56V4.4a2 2 0 1 1 4 0v.09'
        'a1.7 1.7 0 0 0 1.04 1.56 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06'
        'a1.7 1.7 0 0 0-.34 1.87V10.5a1.7 1.7 0 0 0 1.56 1.04h.09a2 2 0 1 1 0 4h-.09'
        'a1.7 1.7 0 0 0-1.56 1.04Z"/></svg>'
    ),
    "close": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M5 5l14 14M19 5L5 19"/></svg>'
    ),
    "bell": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 8a6 6 0 0 1 12 0c0 3.5 1 5.5 2 7H4c1-1.5 2-3.5 2-7Z"/>'
        '<path d="M10 19a2 2 0 0 0 4 0"/></svg>'
    ),
}

# ═══════════════════════════════════════════════════════════════
# DOM 輔助函式
# ═══════════════════════════════════════════════════════════════

def el(tag, cls=None, text=None, html=None, **attrs):
    """建立 DOM 元素並回傳。"""
    elem = document.createElement(tag)
    if cls:
        elem.className = cls
    if text:
        elem.textContent = text
    if html:
        elem.innerHTML = html
    for k, v in attrs.items():
        elem.setAttribute(k.replace("_", "-"), str(v))
    return elem


def svg_span(html_str, cls=None):
    """把一段 SVG innerHTML 字串包成一個真正的 DOM 節點再回傳。

    appendChild() 需要的是 Node，不是字串——瀏覽器 DOM API 沒有隱式字串轉
    Node 這種東西，直接把字串塞進 appendChild() 會丟 TypeError。之前幾個
    呼叫點（見下面 _build_action_bar/_build_camera_card/_build_chat_card）
    是直接 `parent.appendChild(ICONS[...].replace(...))`，第一次執行到就
    整個 build_ui() 中斷在那一行，shell 永遠沒被 append 進 #root，畫面上
    只剩下 body 的 CSS 背景——這支函式就是修正這個問題用的。
    """
    return el("span", cls=cls, html=html_str)


def icon_button(icon_key, label, on_click=None, btn_id=None, cls="nav-item", **extra_attrs):
    """建立帶圖示的側邊欄按鈕（導覽項目／頁尾動作共用同一個結構）。"""
    btn = el("button", cls=cls, html=f'{ICONS.get(icon_key, "")}<span>{label}</span>')
    if btn_id:
        btn.id = btn_id
    for k, v in extra_attrs.items():
        btn.setAttribute(k.replace("_", "-"), str(v))
    if on_click:
        btn.addEventListener("click", create_proxy(on_click))
    return btn


# ═══════════════════════════════════════════════════════════════
# API 呼叫
# ═══════════════════════════════════════════════════════════════

async def api_get(url):
    resp = await window.fetch(url)
    # resp.json() 回傳的是包著 JS 物件的 JsProxy，不是 Python dict——JsProxy
    # 沒有 .get()（除非包的是 JS Map），呼叫 data.get(...) 會直接
    # AttributeError: get。.to_py() 遞迴轉成真正的 Python dict/list，下面
    # 跟其他呼叫端的 .get()/["key"] 才會如預期運作。
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


def _js_opts(opts):
    # pyodide 呼叫 JS 函式時，Python dict 引數預設會轉成 JS Map，不是一般
    # 的 plain object——fetch() 的 RequestInit 是用 `.method`/`.headers`/
    # `.body` 這種一般物件屬性讀設定，Map 沒有這些屬性，讀不到就靜靜地
    # 全部當沒帶，實際送出的請求永遠是預設的 GET、沒有 body。這裡 explicit
    # 用 to_js(..., dict_converter=Object.fromEntries) 轉成真正的 plain
    # object，fetch() 才吃得到 method/headers/body（連同巢狀的 headers
    # dict 一起遞迴轉換）。之前 api_post()/api_delete() 沒轉、直接把
    # Python dict 傳給 window.fetch()，送出去的其實一直都是 GET
    # ——包括原本 /api/chat 的送出訊息，不是這次新增對話紀錄功能才有的問題。
    return to_js(opts, dict_converter=Object.fromEntries)


async def api_post(url, body):
    resp = await window.fetch(url, _js_opts({
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": _json.dumps(body),
    }))
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


async def api_patch(url, body):
    resp = await window.fetch(url, _js_opts({
        "method": "PATCH",
        "headers": {"Content-Type": "application/json"},
        "body": _json.dumps(body),
    }))
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


async def api_delete(url):
    resp = await window.fetch(url, _js_opts({"method": "DELETE"}))
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


# ═══════════════════════════════════════════════════════════════
# 極簡 Markdown → HTML
# ═══════════════════════════════════════════════════════════════

_ESCAPE_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


def _escape(s):
    for ch, ent in _ESCAPE_MAP.items():
        s = s.replace(ch, ent)
    return s


def _inline(line):
    h = _escape(line)
    h = _re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: f"<strong>{m[1] or m[2]}</strong>", h)
    h = _re.sub(r"`([^`]+)`", lambda m: f'<code class="md-code-inline">{m[1]}</code>', h)
    h = _re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", lambda m: f"<em>{m[1]}</em>", h)
    return h


_TABLE_ROW_RE = _re.compile(r"^\s*\|(.*)\|\s*$")
_TABLE_SEP_CELL_RE = _re.compile(r"^:?-+:?$")


def _split_table_row(line):
    m = _TABLE_ROW_RE.match(line)
    if not m:
        return []
    inner = m.group(1)
    return [cell.strip() for cell in inner.split("|")]


def _is_table_separator(line):
    if not _TABLE_ROW_RE.match(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_SEP_CELL_RE.match(c) for c in cells)


def _render_table(header, rows):
    thead = "".join(f"<th>{_inline(c)}</th>" for c in header)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_inline(row[i] if i < len(row) else '')}</td>" for i in range(len(header)))
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="md-table-wrap"><table class="md-table"><thead><tr>' + thead
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def render_markdown(text):
    lines = text.split("\n")
    n_lines = len(lines)
    out = []
    in_code = False
    code_buf = []
    in_list = False

    i = 0
    while i < n_lines:
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                out.append(f'<pre class="md-code-block"><code>{_escape(chr(10).join(code_buf))}</code></pre>')
                code_buf = []
                in_code = False
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if _TABLE_ROW_RE.match(line) and i + 1 < n_lines and _is_table_separator(lines[i + 1]):
            if in_list:
                out.append("</ul>")
                in_list = False
            header = _split_table_row(line)
            i += 2
            rows = []
            while i < n_lines and _TABLE_ROW_RE.match(lines[i]):
                rows.append(_split_table_row(lines[i]))
                i += 1
            out.append(_render_table(header, rows))
            continue

        # h1~h3 縮到 h3~h5、h4~h6 全部封頂在 h6（HTML 標題最深只到 h6）——
        # data/pairs.json 裡從 md-chat 匯入的真實回覆偶爾會有四級以上標題，
        # 沒有這行封頂，"####" 這種寫法會因為規則式只到 h3 而完全不吃、原封
        # 不動地當成一般段落文字露出來。
        m = _re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            lvl = min(len(m[1]) + 2, 6)
            out.append(f'<h{lvl} class="md-heading">{_inline(m[2])}</h{lvl}>')
            i += 1
            continue

        m = _re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            if not in_list:
                out.append('<ul class="md-list">')
                in_list = True
            out.append(f"<li>{_inline(m[1])}</li>")
            i += 1
            continue

        if in_list:
            out.append("</ul>")
            in_list = False

        if line.strip() == "":
            out.append("<br>")
        else:
            out.append(f'<p class="md-p">{_inline(line)}</p>')
        i += 1

    if in_code and code_buf:
        out.append(f'<pre class="md-code-block"><code>{_escape(chr(10).join(code_buf))}</code></pre>')
    if in_list:
        out.append("</ul>")

    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════
# 全域狀態
# ═══════════════════════════════════════════════════════════════
_state = {
    "stream": None,
    "loop_handle": None,
    "in_flight": False,
    "current_model": "sinco",
    "ws": None,
    "ws_ready": False,
    "last_send_ts": None,
    "conversation_id": None,
    "open_menu": None,
    "open_menu_id": None,
}

# DOM 引用（build_ui 時填入）
_dom = {}


# ═══════════════════════════════════════════════════════════════
# 健康檢查
# ═══════════════════════════════════════════════════════════════

async def check_health():
    pill = _dom["status_pill"]
    text = _dom["status_text"]
    pill.setAttribute("data-state", "checking")
    text.textContent = "連線中…"
    try:
        await api_get(HEALTH_URL)
        pill.setAttribute("data-state", "ok")
        text.textContent = "後端已連線"
    except Exception as exc:
        pill.setAttribute("data-state", "down")
        text.textContent = "無法連線後端"
        console.error(f"Health check failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# 顯卡資訊
# ═══════════════════════════════════════════════════════════════

async def check_gpu():
    """查詢後端主機的顯卡資訊並顯示成一顆跟連線狀態一樣樣式的小藥丸。

    真正的查詢邏輯在後端 gpu_info.py（torch.cuda + nvidia-smi）——瀏覽器端
    的 PyScript/Pyodide 是 WASM 沙盒，本來就摸不到主機顯卡，這裡只負責
    fetch 結果、排版成文字。
    """
    pill = _dom["gpu_pill"]
    text = _dom["gpu_text"]
    pill.setAttribute("data-state", "checking")
    text.textContent = "顯卡查詢中…"
    try:
        data = await api_get(GPU_INFO_URL)
        if data.get("available"):
            parts = [data.get("name") or "GPU"]
            total_mb = data.get("total_memory_mb")
            if total_mb:
                used_gb = (data.get("used_memory_mb") or 0) / 1024
                total_gb = total_mb / 1024
                parts.append(f"{used_gb:.1f}/{total_gb:.1f} GB")
            temp = data.get("temperature_c")
            if temp is not None:
                parts.append(f"{temp:.0f}°C")
            pill.setAttribute("data-state", "ok")
            text.textContent = " · ".join(parts)
        else:
            pill.setAttribute("data-state", "down")
            text.textContent = "無可用顯卡（CPU 模式）"
    except Exception as exc:
        pill.setAttribute("data-state", "down")
        text.textContent = "顯卡查詢失敗"
        console.error(f"GPU info check failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# 相機偵測
# ═══════════════════════════════════════════════════════════════

def start_camera():
    video = _dom["video"]
    overlay = _dom["overlay"]
    stage_empty = _dom["stage_empty"]
    toggle_btn = _dom["toggle_btn"]
    results_el = _dom["results"]

    async def _go():
        try:
            # 跟 _js_opts() 同一個坑：pyodide 把 Python dict 轉成 JS Map，
            # getUserMedia() 是用 constraints.video/constraints.audio 這種
            # 一般物件屬性讀設定，Map 沒有這些屬性，讀出來全部是 undefined，
            # 瀏覽器就會丟 "At least one of audio and video must be
            # requested"——不是使用者真的沒開視訊權限，是這個型別轉換問題。
            stream = await window.navigator.mediaDevices.getUserMedia(_js_opts({
                "video": True, "audio": False,
            }))
        except Exception as exc:
            results_el.textContent = f"相機錯誤：{exc}"
            return

        _state["stream"] = stream
        video.srcObject = stream
        await video.play()
        overlay.width = video.videoWidth
        overlay.height = video.videoHeight
        stage_empty.style.display = "none"
        toggle_btn.innerHTML = f'{ICONS["cameraOff"]}<span>關閉相機</span>'
        toggle_btn.classList.add("is-active")
        results_el.textContent = "連線中…"

        _open_detect_socket(_dom["conf_slider"].value)

        # setInterval 是瀏覽器全域，pyscript 沒有把它塞進頂層命名空間，裸寫
        # `setInterval(...)` 會是 NameError；跟下面 stop_camera() 的
        # window.clearInterval(...) 對齊，一律走 window.。callback
        # capture_and_send 是 async function，JS 的 setInterval 直接呼叫
        # 它只會拿到一個「還沒開始跑」的 coroutine（呼叫 async function
        # 本身不會執行函式內容，要交給事件迴圈排程才會真的跑），所以包一層
        # asyncio.ensure_future() 才能讓它真的執行。
        # 這個 lambda 本身也要用 create_proxy() 包起來——傳給 setInterval 的
        # callback 會被重複呼叫很多次（每 500ms 一次），pyodide 只有在函式
        # 呼叫「當下同步結束前」才保證裸 Python callable 轉成的 borrowed
        # proxy 還活著；setInterval() 這個外層呼叫一返回，那個 borrowed
        # proxy 就被銷毀，之後每次計時器真正觸發都會撞上
        # 「This borrowed proxy was automatically destroyed at the end of a
        # function call」——偵測畫面看起來連上了（顯示「偵測中…」），實際上
        # capture_and_send() 從來沒有真的被呼叫過，物件偵測結果永遠不會更新。
        _state["loop_handle"] = window.setInterval(
            create_proxy(lambda: asyncio.ensure_future(capture_and_send())), DETECT_INTERVAL_MS
        )

    # _go() 是 async function：直接呼叫只會產生一個 coroutine 物件、不會
    # 執行任何內容（跟上面 setInterval 的 callback 是同一個道理）。這裡
    # 呼叫 start_camera() 的是同步的 toggle_camera()（再上一層是
    # addEventListener 的同步 lambda），沒有人會去 await 這個 coroutine，
    # 原本寫法會讓「開啟相機」按鈕點了完全沒反應（相機權限視窗都不會跳出來）
    # ——asyncio.ensure_future() 明確把它排進事件迴圈執行。
    asyncio.ensure_future(_go())


def _open_detect_socket(conf):
    """開一條 /ws/detect 連線，相機開著就一直用同一條連線送畫面、收偵測
    結果，取代原本「每一格畫面都重新 fetch 一次 /api/detect」的作法。

    conf 只在連線當下帶一次（見 _detect_ws_url() 的註解），中途拉動信心值
    滑桿不會立即生效，要關掉相機再重開才會用新的值——這是刻意的取捨，
    避免要另外設計一套「同一條連線裡混雜二進位畫面跟文字設定訊息」的協定。
    """
    ws = WebSocket.new(f"{DETECT_WS_URL}?conf={conf}")
    ws.binaryType = "arraybuffer"

    def _on_open(_evt):
        _state["ws_ready"] = True
        _dom["results"].textContent = "偵測中…"

    def _on_message(evt):
        _state["in_flight"] = False
        try:
            data = _json.loads(str(evt.data))
        except Exception as exc:
            console.error(f"detect ws parse error: {exc}")
            return

        import time
        t0 = _state.get("last_send_ts")
        if t0 is not None:
            _dom["latency"].textContent = f"{round(time.time() * 1000 - t0)} ms"

        overlay = _dom["overlay"]
        _draw_detections(data, overlay.getContext("2d"), overlay, _dom["results"])

    def _on_close(_evt):
        _state["ws_ready"] = False
        _state["ws"] = None
        _state["in_flight"] = False

    def _on_error(_evt):
        console.error("detect ws error")
        _dom["results"].textContent = "偵測連線發生錯誤"

    ws.addEventListener("open", create_proxy(_on_open))
    ws.addEventListener("message", create_proxy(_on_message))
    ws.addEventListener("close", create_proxy(_on_close))
    ws.addEventListener("error", create_proxy(_on_error))

    _state["ws"] = ws
    _state["ws_ready"] = False


def stop_camera():
    handle = _state.get("loop_handle")
    if handle is not None:
        window.clearInterval(handle)
        _state["loop_handle"] = None

    ws = _state.get("ws")
    if ws is not None:
        ws.close()
        _state["ws"] = None
        _state["ws_ready"] = False

    stream = _state.get("stream")
    if stream is not None:
        for track in stream.getTracks():
            track.stop()
        _state["stream"] = None

    _dom["overlay"].getContext("2d").clearRect(0, 0, _dom["overlay"].width, _dom["overlay"].height)
    _dom["stage_empty"].style.display = "flex"
    _dom["toggle_btn"].innerHTML = f'{ICONS["camera"]}<span>開啟相機</span>'
    _dom["toggle_btn"].classList.remove("is-active")
    _dom["results"].textContent = ""
    _dom["latency"].textContent = ""


def toggle_camera():
    if _state["stream"]:
        stop_camera()
    else:
        start_camera()


async def capture_and_send():
    ws = _state.get("ws")
    if ws is None or not _state["ws_ready"] or _state["in_flight"]:
        return
    video = _dom["video"]
    if not video.videoWidth:
        return

    _state["in_flight"] = True
    try:
        cap_canvas = document.createElement("canvas")
        cap_ctx = cap_canvas.getContext("2d")
        scale = CAPTURE_WIDTH / video.videoWidth
        cap_canvas.width = CAPTURE_WIDTH
        cap_canvas.height = round(video.videoHeight * scale)
        cap_ctx.drawImage(video, 0, 0, cap_canvas.width, cap_canvas.height)

        # `new Promise(...)` 是 JS 語法，不是合法的 Python——pyodide 執行的
        # 是真正的 CPython，`new` 不是關鍵字，這裡會直接 SyntaxError，整支
        # action.py 連編譯都過不了，build_ui() 根本沒機會被呼叫到（畫面上
        # 只剩下 body 的 CSS 背景，看起來就是完全沒有 JS/Python 錯誤那樣）。
        # pyodide 呼叫 JS 建構子的慣例是 `<JS 類別>.new(...)`，Promise 也
        # 一樣。`blob.arrayBuffer()` 本身回傳的就是真正的 JS Promise（不是
        # callback-based API），pyodide 的 JsProxy 對 Promise 原生支援
        # `await`，不需要再包一層 Promise.new()。
        # toBlob() 是非同步 callback API，resolve/reject 要留到「稍後」瀏覽器
        # 編碼完成才會被呼叫；但 pyodide 對這種借用代理（borrowed proxy）
        # 預設只在當次同步呼叫（Promise.new 的 executor）結束前有效，一返回
        # 就會被自動銷毀，等 toBlob 真正非同步呼叫 resolve 時就會撞上
        # 「This borrowed proxy was automatically destroyed at the end of a
        # function call」。用 create_once_callable() 包住 resolve，讓它的
        # 生命週期改成「被呼叫一次之後才銷毀」，不受這次同步呼叫範圍限制。
        from js import Promise
        from pyodide.ffi import create_once_callable

        def _executor(resolve, reject):
            cap_canvas.toBlob(create_once_callable(resolve), "image/jpeg", 0.7)

        blob = await Promise.new(_executor)
        buf = await blob.arrayBuffer()

        import time
        _state["last_send_ts"] = time.time() * 1000
        ws.send(buf)
    except Exception as exc:
        _state["in_flight"] = False
        _dom["results"].textContent = f"偵測錯誤：{exc}"


def _draw_detections(data, ctx, overlay, results_el):
    ctx.clearRect(0, 0, overlay.width, overlay.height)
    if not data.get("width") or not data.get("height"):
        return

    sx = overlay.width / data["width"]
    sy = overlay.height / data["height"]
    ctx.lineWidth = 2
    ctx.font = "14px system-ui, sans-serif"
    ctx.textBaseline = "top"

    counts = {}
    for det in data.get("detections", []):
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1
        x1, y1, x2, y2 = det["box"]
        x, y = x1 * sx, y1 * sy
        w, h = (x2 - x1) * sx, (y2 - y1) * sy

        ctx.strokeStyle = "#5b8cff"
        ctx.strokeRect(x, y, w, h)
        tag = f'{label} {det["confidence"] * 100:.0f}%'
        tw = ctx.measureText(tag).width
        ctx.fillStyle = "#5b8cff"
        ctx.fillRect(x, max(0, y - 18), tw + 8, 18)
        ctx.fillStyle = "#0b0d12"
        ctx.fillText(tag, x + 4, max(0, y - 18) + 2)

    results_el.innerHTML = ""
    for lbl, n in counts.items():
        chip = el("span", cls="chip", html=f'<span class="dot"></span>{lbl} × {n}')
        results_el.appendChild(chip)
    if not counts:
        results_el.textContent = "沒有偵測到物件"


# ═══════════════════════════════════════════════════════════════
# 對話
# ═══════════════════════════════════════════════════════════════

def _build_message_element(role, content):
    """組出一則訊息的 DOM，結構對齊 default.html 的
    `.message.{role}-message > .message-bubble`（外層負責左右靠齊，
    內層才是真正的氣泡/文字），不是自創的 `.chat-message` 扁平結構。
    `_append_chat()`（送出當下即時附加一則）跟 `_render_messages()`
    （切換對話時整批重繪歷史訊息）共用這支，兩邊的氣泡樣式才不會走鐘。
    """
    wrapper = el("div", cls=f"message {role}-message")
    bubble = el("div", cls="message-bubble")
    if role == "assistant":
        bubble.innerHTML = render_markdown(content)
    else:
        bubble.textContent = content
    wrapper.appendChild(bubble)
    return wrapper, bubble


def _append_chat(role, content, pending=False):
    chat_log = _dom["chat_log"]
    chat_empty = _dom.get("chat_empty")
    if chat_empty is not None:
        chat_empty.remove()
        _dom["chat_empty"] = None

    wrapper, bubble = _build_message_element(role, content)
    if pending:
        bubble.classList.add("is-pending")
    chat_log.appendChild(wrapper)
    chat_log.scrollTop = chat_log.scrollHeight
    return bubble


def _render_messages(messages):
    """切換/新增/清除對話時，整批重繪 chat_log——取代掉目前畫面上顯示的
    那一批訊息，不是附加。"""
    chat_log = _dom["chat_log"]
    chat_log.innerHTML = ""
    if not messages:
        empty = el("div", cls="chat-empty", id="chatEmpty", text="說點什麼開始對話吧")
        chat_log.appendChild(empty)
        _dom["chat_empty"] = empty
        return
    _dom["chat_empty"] = None
    for msg in messages:
        wrapper, _bubble = _build_message_element(msg["role"], msg["content"])
        chat_log.appendChild(wrapper)
    chat_log.scrollTop = chat_log.scrollHeight


async def send_chat(text):
    if not text.strip():
        return
    if _state.get("conversation_id") is None:
        # 保底：正常流程下 _init_conversations() 已經在頁面載入時建好/選好
        # 一筆對話，只有初始化失敗或發生競態時才會落到這裡，避免送出按鈕
        # 因為沒有 conversation_id 而整個沒反應。
        await new_conversation()
    conversation_id = _state["conversation_id"]

    _dom["chat_input"].value = ""
    _dom["chat_input"].style.height = "auto"
    _dom["chat_send"].disabled = True
    _append_chat("user", text)
    pending = _append_chat("assistant", "…", pending=True)

    try:
        data = await api_post(CHAT_URL, {"message": text, "conversation_id": conversation_id})
        pending.innerHTML = render_markdown(data["reply"])
        pending.classList.remove("is-pending")
        await refresh_history_list()
    except Exception as exc:
        pending.textContent = f"錯誤：{exc}"
    finally:
        _dom["chat_send"].disabled = False
        _dom["chat_input"].focus()


# ═══════════════════════════════════════════════════════════════
# 對話紀錄（側邊欄清單，對齊 default.html 的 .history-list/.chat-item，
# 但接了真正的後端存檔——不是寫死的假資料）
# ═══════════════════════════════════════════════════════════════

async def fetch_conversations():
    return await api_get(CONVERSATIONS_URL)


async def refresh_history_list():
    """重新抓一次清單、重繪側邊欄，並把目前對話的標題同步回標題列——
    第一則訊息送出後後端會自動幫對話取標題（見 conversation_store.py 的
    append_message()），這裡是唯一需要把新標題套回畫面的地方。"""
    try:
        conversations = await fetch_conversations()
    except Exception as exc:
        console.error(f"list conversations failed: {exc}")
        return
    _render_history_list(conversations)
    current_id = _state.get("conversation_id")
    for conv in conversations:
        if conv["id"] == current_id:
            _dom["chat_title"].textContent = conv["title"]
            break


async def switch_conversation(conversation_id):
    if _state.get("conversation_id") == conversation_id:
        return
    try:
        conv = await api_get(f"{CONVERSATIONS_URL}/{conversation_id}")
    except Exception as exc:
        console.error(f"load conversation failed: {exc}")
        return
    _state["conversation_id"] = conversation_id
    _render_messages(conv["messages"])
    await refresh_history_list()


async def new_conversation():
    conv = await api_post(CONVERSATIONS_URL, {})
    _state["conversation_id"] = conv["id"]
    _render_messages([])
    await refresh_history_list()


async def delete_conversation_item(conversation_id):
    try:
        await api_delete(f"{CONVERSATIONS_URL}/{conversation_id}")
    except Exception as exc:
        console.error(f"delete conversation failed: {exc}")
        return
    if _state.get("conversation_id") != conversation_id:
        await refresh_history_list()
        return
    # 刪掉的剛好是目前開著的對話：換去清單裡最新的一筆，清單也空了才
    # 建一筆新的——不能讓畫面停在一個已經不存在的 conversation_id 上。
    _state["conversation_id"] = None
    remaining = await fetch_conversations()
    if remaining:
        await switch_conversation(remaining[0]["id"])
    else:
        await new_conversation()


async def rename_conversation_item(conversation_id, title):
    try:
        await api_patch(f"{CONVERSATIONS_URL}/{conversation_id}", {"title": title})
    except Exception as exc:
        console.error(f"rename conversation failed: {exc}")
        return
    await refresh_history_list()


def _close_open_menu():
    """關掉目前開著的 chat-item 下拉選單（重新命名／刪除），沒有開著的話
    就什麼都不做——document 層級的關閉監聽器跟每次開新選單前都會呼叫這支，
    所以要能在「本來就沒開」的情況下安全地重複呼叫。"""
    menu = _state.get("open_menu")
    if menu is not None:
        menu.remove()
    _state["open_menu"] = None
    _state["open_menu_id"] = None


def _start_rename(title_el, conversation_id, current_title):
    """把標題 <span> 換成一個輸入框，Enter/失焦送出、Esc 取消。

    `settled` 是為了擋掉「Enter/Esc 換掉輸入框」跟「換掉輸入框造成瀏覽器
    自動觸發 blur」疊在一起的重複送出——兩個事件都可能呼叫到 _finish()，
    沒有這個旗標會在使用者按 Enter 時多打一次沒必要的 rename API。"""
    settled = {"done": False}
    input_el = el("input", cls="chat-item-rename-input")
    input_el.value = current_title
    input_el.setAttribute("maxlength", "60")
    title_el.replaceWith(input_el)
    input_el.focus()
    input_el.select()

    def _finish(commit):
        if settled["done"]:
            return
        settled["done"] = True
        new_title = input_el.value.strip()
        display_title = new_title if (commit and new_title) else current_title
        restored = el("span", cls="chat-item-title", text=display_title)
        input_el.replaceWith(restored)
        if commit and new_title and new_title != current_title:
            asyncio.ensure_future(rename_conversation_item(conversation_id, new_title))

    def _on_keydown(e):
        if e.key == "Enter":
            e.preventDefault()
            _finish(True)
        elif e.key == "Escape":
            e.preventDefault()
            _finish(False)

    input_el.addEventListener("keydown", create_proxy(_on_keydown))
    input_el.addEventListener("blur", create_proxy(lambda _e: _finish(True)))
    input_el.addEventListener("click", create_proxy(lambda e: e.stopPropagation()))


def _open_item_menu(item, title_el, conversation_id, current_title, anchor_btn):
    _close_open_menu()

    menu = el("div", cls="chat-item-menu")
    rename_btn = el("button", cls="chat-item-menu-item", html=f'{ICONS["edit"]}<span>重新命名</span>')
    delete_btn = el("button", cls="chat-item-menu-item danger", html=f'{ICONS["clear"]}<span>刪除對話</span>')

    def _on_rename(e):
        e.stopPropagation()
        _close_open_menu()
        _start_rename(title_el, conversation_id, current_title)

    def _on_delete(e):
        e.stopPropagation()
        _close_open_menu()
        asyncio.ensure_future(delete_conversation_item(conversation_id))

    rename_btn.addEventListener("click", create_proxy(_on_rename))
    delete_btn.addEventListener("click", create_proxy(_on_delete))
    menu.appendChild(rename_btn)
    menu.appendChild(delete_btn)

    item.appendChild(menu)
    # 選單釘在被點的「更多選項」按鈕旁邊（貼著側邊欄右緣往右彈出），不是整個
    # 視窗右下角——之前 CSS 寫死 bottom/right 會讓選單跟點的是哪個對話項目
    # 完全無關。position:fixed 不受 .sidebar 的 overflow-y:auto 裁切影響，
    # 所以座標用 getBoundingClientRect() 現算即可。
    rect = anchor_btn.getBoundingClientRect()
    menu.style.top = f"{rect.top}px"
    menu.style.left = f"{rect.right + 6}px"
    _state["open_menu"] = menu
    _state["open_menu_id"] = conversation_id


def _render_history_list(conversations):
    # 每次重繪都先關掉舊選單——舊選單掛在舊的 DOM 節點上，innerHTML = ""
    # 會把它一起清空，_state 裡的參照卻還留著、後續 _close_open_menu()
    # 會去 remove() 一個早就不在文件裡的節點。
    _close_open_menu()
    history_list = _dom["history_list"]
    history_list.innerHTML = ""
    if not conversations:
        history_list.appendChild(el("div", cls="history-empty", text="還沒有任何對話"))
        return

    current_id = _state.get("conversation_id")
    for conv in conversations:
        item = el("div", cls="chat-item", data_chat_id=conv["id"])
        if conv["id"] == current_id:
            item.classList.add("active")

        item.appendChild(svg_span(ICONS["chat"], cls="chat-item-icon"))
        title_el = el("span", cls="chat-item-title", text=conv["title"])
        item.appendChild(title_el)

        more_btn = el("button", cls="chat-item-more", title="更多選項", html=ICONS["dots"])

        def _make_switch_handler(cid):
            return lambda _e: asyncio.ensure_future(switch_conversation(cid))

        def _make_more_handler(it, t_el, cid, title, btn):
            def _handler(e):
                e.stopPropagation()
                if _state.get("open_menu_id") == cid:
                    _close_open_menu()
                else:
                    _open_item_menu(it, t_el, cid, title, btn)
            return _handler

        item.addEventListener("click", create_proxy(_make_switch_handler(conv["id"])))
        more_btn.addEventListener(
            "click", create_proxy(_make_more_handler(item, title_el, conv["id"], conv["title"], more_btn))
        )
        item.appendChild(more_btn)

        history_list.appendChild(item)


async def _init_conversations():
    try:
        conversations = await fetch_conversations()
    except Exception as exc:
        console.error(f"init conversations failed: {exc}")
        return
    if conversations:
        await switch_conversation(conversations[0]["id"])
    else:
        await new_conversation()


async def clear_chat():
    conversation_id = _state.get("conversation_id")
    if conversation_id is None:
        return
    try:
        await api_post(f"{CONVERSATIONS_URL}/{conversation_id}/clear", {})
    except Exception as exc:
        console.error(f"clear conversation failed: {exc}")
        return
    _render_messages([])
    await refresh_history_list()


# ═══════════════════════════════════════════════════════════════
# 功能列按鈕處理
# ═══════════════════════════════════════════════════════════════

def _on_model_change(event):
    model = event.target.value
    _state["current_model"] = model
    badge = _dom.get("model_badge")
    if badge:
        badge.textContent = model


def _on_character_click():
    _append_chat("assistant", "角色選擇功能需要搭配後端 /character 指令使用，請先在桌面版或 CLI 中設定角色。")


async def _on_memory_click():
    try:
        data = await api_post(CHAT_URL, {"message": "/memory"})
        _append_chat("assistant", data.get("reply", "記憶功能暫無回覆。"))
    except Exception as exc:
        _append_chat("assistant", f"記憶查詢失敗：{exc}")


async def _on_learn_click():
    try:
        data = await api_post(CHAT_URL, {"message": "/learn"})
        _append_chat("assistant", data.get("reply", "學習功能需要搭配資料集使用。"))
    except Exception as exc:
        _append_chat("assistant", f"學習功能呼叫失敗：{exc}")


# ═══════════════════════════════════════════════════════════════
# 建構 UI
# ═══════════════════════════════════════════════════════════════

def _build_sidebar():
    """左側邊欄：品牌 + 連線狀態 + 模型切換 + 功能導覽 + 相機控制 + 頁尾。

    原本這些全部擠在同一條水平 action-bar 裡（見版本歷史），按鈕一多就
    在窄螢幕上擠成一團、文字標籤還要靠 CSS 隱藏才塞得下。改成直向的
    側邊欄清單後，每個功能一行、圖示+文字永遠都在，佈局思路對齊
    Claude.ai 那種「側邊欄放導覽、主區塊放內容」的分法。
    """
    aside = el("aside", cls="sidebar")

    # ── 品牌 ──
    brand = el("div", cls="sidebar-brand")
    brand.appendChild(el("div", cls="brand-mark", html=ICONS["logo"]))
    bt = el("div", cls="brand-text")
    bt.appendChild(el("h1", text="AI Modules"))
    bt.appendChild(el("span", text="本機視覺與對話模型 Playground"))
    brand.appendChild(bt)
    aside.appendChild(brand)

    # ── 對話紀錄：對齊 default.html 的 .history-list/.chat-item，
    #    但接了 conversation_store.py 的真實存檔，不是寫死的假資料 ──
    history_section = el("div", cls="sidebar-section history-section")

    new_chat_btn = el("button", cls="btn-new-chat", id="btnNewChat")
    new_chat_btn.appendChild(svg_span(ICONS["chat"].replace(
        'viewBox', 'style="width:15px;height:15px;stroke:currentColor;flex-shrink:0" viewBox'
    )))
    new_chat_btn.appendChild(el("span", text="新增對話"))
    new_chat_btn.addEventListener("click", create_proxy(lambda _: asyncio.ensure_future(new_conversation())))
    history_section.appendChild(new_chat_btn)

    history_list = el("div", cls="history-list", id="historyList")
    _dom["history_list"] = history_list
    history_section.appendChild(history_list)

    aside.appendChild(history_section)

    # ── 彈性空間：把下面「連線狀態／功能」跟頁尾一起推到底部，貼著頁尾上方 ──
    aside.appendChild(el("div", cls="sidebar-spacer"))

    # ── 連線狀態 ──
    status_section = el("div", cls="sidebar-section")

    pill = el("div", cls="status-pill", id="statusPill")
    pill.setAttribute("data-state", "checking")
    pill.setAttribute("title", "點一下重新檢查連線")
    pill.appendChild(el("span", cls="status-dot"))
    status_text = el("span", id="statusText", text="連線中…")
    pill.appendChild(status_text)
    pill.addEventListener("click", create_proxy(lambda _: asyncio.ensure_future(check_health())))
    _dom["status_pill"] = pill
    _dom["status_text"] = status_text
    status_section.appendChild(pill)

    gpu_pill = el("div", cls="status-pill", id="gpuPill")
    gpu_pill.setAttribute("data-state", "checking")
    gpu_pill.setAttribute("title", "點一下重新查詢顯卡資訊")
    gpu_pill.appendChild(el("span", cls="status-dot"))
    gpu_text = el("span", id="gpuText", text="顯卡查詢中…")
    gpu_pill.appendChild(gpu_text)
    gpu_pill.addEventListener("click", create_proxy(lambda _: asyncio.ensure_future(check_gpu())))
    _dom["gpu_pill"] = gpu_pill
    _dom["gpu_text"] = gpu_text
    status_section.appendChild(gpu_pill)

    aside.appendChild(status_section)

    # ── 功能導覽：角色／記憶／學習 ──
    tools_section = el("div", cls="sidebar-section nav-section")
    tools_section.appendChild(el("span", cls="section-label", text="功能"))
    tools_section.appendChild(icon_button("character", "角色", on_click=lambda _: _on_character_click()))
    tools_section.appendChild(icon_button("memory", "記憶", on_click=lambda _: asyncio.ensure_future(_on_memory_click())))
    tools_section.appendChild(icon_button("learn", "學習", on_click=lambda _: asyncio.ensure_future(_on_learn_click())))
    aside.appendChild(tools_section)

    # ── 頁尾：清除對話 + 設定（模型／物件偵測信心值彙整到設定彈窗）+ 版權 ──
    footer = el("div", cls="sidebar-footer")

    footer.appendChild(icon_button(
        "clear", "清除對話", on_click=lambda _: asyncio.ensure_future(clear_chat()), cls="footer-action"
    ))
    footer.appendChild(icon_button(
        "settings", "設定", on_click=lambda _: toggle_settings_modal(), cls="footer-action", btn_id="settingsBtn"
    ))
    footer.appendChild(icon_button(
        "bell", "發送通知", on_click=lambda _: toggle_notify_composer(), cls="footer-action", btn_id="notifyComposerBtn"
    ))
    footer.appendChild(el("div", cls="sidebar-credit", text="© Author: Roy Zeng"))

    aside.appendChild(footer)

    return aside


def _close_on_backdrop(box):
    """點在遮罩背景（不是點在面板內容）就關閉——共用給偵測/設定兩個彈窗。"""
    def _handler(e):
        if e.target == box:
            box.classList.remove("is-open")
    return _handler


def toggle_detect_overlay():
    box = _dom["detect_overlay"]
    if box.classList.contains("is-open"):
        close_detect_overlay()
    else:
        box.classList.add("is-open")
        cam_btn = _dom.get("cam_toggle_btn")
        if cam_btn:
            cam_btn.classList.add("is-active")


def close_detect_overlay():
    _dom["detect_overlay"].classList.remove("is-open")
    cam_btn = _dom.get("cam_toggle_btn")
    if cam_btn:
        cam_btn.classList.remove("is-active")
    # 關掉物件偵測介面時順便關相機，避免鏡頭在畫面看不到的情況下繼續佔用。
    if _state["stream"]:
        stop_camera()


def toggle_settings_modal():
    box = _dom["settings_modal"]
    if box.classList.contains("is-open"):
        box.classList.remove("is-open")
    else:
        box.classList.add("is-open")


def toggle_notify_composer():
    box = _dom["notify_composer"]
    if box.classList.contains("is-open"):
        box.classList.remove("is-open")
    else:
        box.classList.add("is-open")
        _dom["notify_composer_input"].focus()


async def _send_notification():
    text_input = _dom["notify_composer_input"]
    status = _dom["notify_composer_status"]
    text = text_input.value.strip()
    if not text:
        return
    send_btn = _dom["notify_composer_send"]
    send_btn.disabled = True
    try:
        await api_post(NOTIFY_URL, {"message": text})
        text_input.value = ""
        status.textContent = "已發送"
    except Exception as exc:
        status.textContent = f"發送失敗：{exc}"
    finally:
        send_btn.disabled = False


def _build_detect_overlay():
    """物件偵測獨立介面：借用 index.html 既有的 `<div class="box" id="box">`
    當彈窗容器，不再跟對話面板並排塞在同一個 main-grid 裡——點側邊欄/輸入框
    旁的相機圖示才會浮現，再點一次或點背景/✕關閉（見 toggle_detect_overlay()）。
    """
    box = document.getElementById("box")
    box.innerHTML = ""

    panel = el("div", cls="overlay-panel")

    head = el("div", cls="overlay-head")
    head.appendChild(svg_span(ICONS["camera"].replace('viewBox', 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox')))
    ht = el("div")
    ht.appendChild(el("h2", text="物件偵測"))
    ht.appendChild(el("p", text="即時攝影機影像 + YOLO 偵測"))
    head.appendChild(ht)
    close_btn = el("button", cls="overlay-close", title="關閉", html=ICONS["close"])
    close_btn.addEventListener("click", create_proxy(lambda _: close_detect_overlay()))
    head.appendChild(close_btn)
    panel.appendChild(head)

    body = el("div", cls="overlay-body")

    controls = el("div", cls="overlay-controls")
    toggle_btn = icon_button("camera", "開啟相機", on_click=lambda _: toggle_camera(), cls="btn-camera-toggle")
    toggle_btn.id = "toggleBtn"
    _dom["toggle_btn"] = toggle_btn
    controls.appendChild(toggle_btn)
    latency = el("span", id="latency")
    _dom["latency"] = latency
    controls.appendChild(latency)
    body.appendChild(controls)

    stage = el("div", id="stage")
    video = el("video", id="video")
    video.setAttribute("autoplay", "")
    video.setAttribute("playsinline", "")
    video.setAttribute("muted", "")
    overlay_canvas = el("canvas", id="overlay")
    stage_empty = el("div", cls="stage-empty", id="stageEmpty")
    stage_empty.appendChild(svg_span(ICONS["cameraOff"].replace('viewBox', 'style="width:34px;height:34px;stroke:var(--text-faint)" viewBox')))
    stage_empty.appendChild(el("span", text="相機尚未開啟"))
    stage.appendChild(video)
    stage.appendChild(overlay_canvas)
    stage.appendChild(stage_empty)
    body.appendChild(stage)

    _dom["video"] = video
    _dom["overlay"] = overlay_canvas
    _dom["stage_empty"] = stage_empty

    results = el("div", id="results")
    _dom["results"] = results
    body.appendChild(results)

    panel.appendChild(body)
    box.appendChild(panel)
    box.addEventListener("click", create_proxy(_close_on_backdrop(box)))

    _dom["detect_overlay"] = box


def _build_settings_modal():
    """設定彈窗：彙整原本分散在側邊欄的模型選擇跟物件偵測信心值滑桿，
    側邊欄只留一顆「設定」圖示當入口（見 _build_sidebar()）。"""
    box = el("div", cls="box", id="settingsModal")

    panel = el("div", cls="overlay-panel")

    head = el("div", cls="overlay-head")
    head.appendChild(svg_span(ICONS["settings"].replace('viewBox', 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox')))
    ht = el("div")
    ht.appendChild(el("h2", text="設定"))
    ht.appendChild(el("p", text="模型與物件偵測參數"))
    head.appendChild(ht)
    close_btn = el("button", cls="overlay-close", title="關閉", html=ICONS["close"])
    close_btn.addEventListener("click", create_proxy(lambda _: toggle_settings_modal()))
    head.appendChild(close_btn)
    panel.appendChild(head)

    body = el("div", cls="overlay-body")

    model_section = el("div", cls="settings-field")
    model_section.appendChild(el("span", cls="section-label", text="模型"))
    model_wrap = el("div", cls="model-select")
    model_wrap.appendChild(svg_span(ICONS["model"].replace('viewBox', 'style="width:15px;height:15px;stroke:var(--accent);flex-shrink:0" viewBox')))
    select = el("select", id="modelSelect")
    for val in ("auto", "sinco", "code", "nvidia"):
        opt = el("option", value=val, text=val)
        if val == "sinco":
            opt.selected = True
        select.appendChild(opt)
    select.addEventListener("change", create_proxy(_on_model_change))
    model_wrap.appendChild(select)
    badge = el("span", cls="model-badge", id="modelBadge", text="sinco")
    model_wrap.appendChild(badge)
    _dom["model_badge"] = badge
    model_section.appendChild(model_wrap)
    body.appendChild(model_section)

    slider_section = el("div", cls="settings-field")
    slider_section.appendChild(el("span", cls="section-label", text="物件偵測信心值"))
    slider_field = el("div", cls="slider-field")
    slider = el("input", id="confSlider")
    slider.setAttribute("type", "range")
    slider.setAttribute("min", "0.1")
    slider.setAttribute("max", "0.9")
    slider.setAttribute("step", "0.05")
    slider.setAttribute("value", "0.35")
    conf_val = el("span", cls="val", id="confValue", text="0.35")
    slider.addEventListener("input", create_proxy(lambda e: setattr(conf_val, "textContent", e.target.value)))
    slider_field.appendChild(slider)
    slider_field.appendChild(conf_val)
    _dom["conf_slider"] = slider
    slider_section.appendChild(slider_field)
    body.appendChild(slider_section)

    panel.appendChild(body)
    box.appendChild(panel)
    box.addEventListener("click", create_proxy(_close_on_backdrop(box)))

    _dom["settings_modal"] = box
    return box


def _build_notify_composer_modal():
    """發送通知彈窗：輸入訊息後 POST /api/notifications，使用者端
    python.py 的通知面板會輪詢到並顯示（見該檔案 _build_notify_panel()）。
    """
    box = el("div", cls="box", id="notifyComposerBox")

    panel = el("div", cls="overlay-panel")

    head = el("div", cls="overlay-head")
    head.appendChild(svg_span(ICONS["bell"].replace('viewBox', 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox')))
    ht = el("div")
    ht.appendChild(el("h2", text="發送通知"))
    ht.appendChild(el("p", text="訊息會顯示在使用者前端的通知面板"))
    head.appendChild(ht)
    close_btn = el("button", cls="overlay-close", title="關閉", html=ICONS["close"])
    close_btn.addEventListener("click", create_proxy(lambda _: toggle_notify_composer()))
    head.appendChild(close_btn)
    panel.appendChild(head)

    body = el("div", cls="overlay-body")

    field = el("div", cls="settings-field")
    text_input = el("textarea", cls="notify-composer-input", id="notifyComposerInput")
    text_input.setAttribute("rows", "3")
    text_input.setAttribute("placeholder", "輸入要發送的通知內容…")
    _dom["notify_composer_input"] = text_input
    field.appendChild(text_input)
    body.appendChild(field)

    send_btn = el("button", cls="btn-camera-toggle", text="發送")
    send_btn.setAttribute("type", "button")
    send_btn.addEventListener("click", create_proxy(lambda _: asyncio.ensure_future(_send_notification())))
    _dom["notify_composer_send"] = send_btn
    body.appendChild(send_btn)

    status = el("span", cls="notify-composer-status")
    _dom["notify_composer_status"] = status
    body.appendChild(status)

    panel.appendChild(body)
    box.appendChild(panel)
    box.addEventListener("click", create_proxy(_close_on_backdrop(box)))

    _dom["notify_composer"] = box
    return box


def _build_chat_card():
    """對話面板：結構/命名對齊 default.html 的
    `.chat-header` / `.chat-messages` / `.chat-input-container` /
    `#chat-form`（textarea + 右下角送出鍵），取代原本借用 Claude.ai
    卡片外框的 `.card card-chat` 版型——這支才是真正在跑的前端
    （index.html 載入 action.py），default.html 本身只是示範用的
    靜態樣式參考，沒有接任何 JS/後端邏輯。
    """
    panel = el("section", cls="chat-panel")

    header = el("header", cls="chat-header")
    title_info = el("div", cls="chat-title-info")
    title_el = el("h2", text="對話")
    _dom["chat_title"] = title_el
    title_info.appendChild(title_el)
    header.appendChild(title_info)
    panel.appendChild(header)

    chat_log = el("div", cls="chat-messages", id="chatLog")
    chat_empty = el("div", cls="chat-empty", id="chatEmpty", text="說點什麼開始對話吧")
    chat_log.appendChild(chat_empty)
    _dom["chat_log"] = chat_log
    _dom["chat_empty"] = chat_empty
    panel.appendChild(chat_log)

    footer = el("footer", cls="chat-input-container")
    form = el("form", id="chatForm")

    chat_input = el("textarea", id="chatInput")
    chat_input.setAttribute("placeholder", "輸入訊息…（Shift + Enter 換行）")
    chat_input.setAttribute("rows", "1")
    _dom["chat_input"] = chat_input

    def _autosize(_evt=None):
        # textarea 高度要先重置成 auto 才能讓 scrollHeight 反映「內容縮短
        # 後」該有的高度，不然只會單調往上長、刪字時高度不會跟著縮回去。
        chat_input.style.height = "auto"
        chat_input.style.height = f"{min(chat_input.scrollHeight, 160)}px"

    def _submit_current():
        text = chat_input.value.strip()
        if text:
            asyncio.ensure_future(send_chat(text))

    def _on_keydown(e):
        # Enter 送出、Shift+Enter 換行，對齊 default.html 輸入框的提示文字；
        # 換行是 textarea 預設行為，只有「送出」這個情境需要攔截。
        if e.key == "Enter" and not e.shiftKey:
            e.preventDefault()
            _submit_current()

    chat_input.addEventListener("input", create_proxy(_autosize))
    chat_input.addEventListener("keydown", create_proxy(_on_keydown))
    form.appendChild(chat_input)

    input_footer = el("div", cls="input-footer")

    cam_btn = el("button", cls="btn-cam", id="camToggleBtn", title="物件偵測", html=ICONS["camera"])
    cam_btn.setAttribute("type", "button")
    cam_btn.addEventListener("click", create_proxy(lambda e: toggle_detect_overlay()))
    _dom["cam_toggle_btn"] = cam_btn
    input_footer.appendChild(cam_btn)

    send_btn = el("button", cls="btn-send", id="chatSend", html=ICONS["send"])
    send_btn.setAttribute("type", "submit")
    _dom["chat_send"] = send_btn
    input_footer.appendChild(send_btn)
    form.appendChild(input_footer)

    # 用同步 handler 呼叫 e.preventDefault()（事件觸發當下就要同步呼叫），
    # 實際的非同步工作交給 asyncio.ensure_future() 明確排程——跟這個檔案
    # 其餘事件處理的寫法一致，不依賴「傳一個 async function 給
    # addEventListener 會被自動排程」這種容易因 pyodide 版本而異的隱式行為。
    def _on_submit(e):
        e.preventDefault()
        _submit_current()

    form.addEventListener("submit", create_proxy(_on_submit))
    footer.appendChild(form)
    panel.appendChild(footer)

    return panel


def build_ui():
    root = document.getElementById("root")
    root.innerHTML = ""

    shell = el("div", cls="app-shell")
    shell.appendChild(_build_sidebar())

    main = el("main", cls="main")
    grid = el("div", cls="main-grid")
    grid.appendChild(_build_chat_card())
    main.appendChild(grid)
    shell.appendChild(main)

    root.appendChild(shell)
    root.appendChild(_build_settings_modal())
    root.appendChild(_build_notify_composer_modal())

    # 物件偵測介面掛在 index.html 既有的 `<div class="box" id="box">` 上，
    # 不是 #root 底下——build_ui() 每次重繪 #root 都不會動到它。
    _build_detect_overlay()


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════
build_ui()
document.addEventListener("click", create_proxy(lambda _e: _close_open_menu()))
asyncio.ensure_future(check_health())
asyncio.ensure_future(check_gpu())
asyncio.ensure_future(_init_conversations())
