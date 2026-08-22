"""PyScript 按鈕功能列 + 完整前端 UI。

在瀏覽器中由 PyScript (Pyodide) 執行，建構整個頁面的 DOM，
包含頂部功能列、物件偵測區塊、對話區塊，並透過 fetch 呼叫後端 API。
"""

from pyscript import document, window
from js import console, WebSocket
from js import URL as _JsURL
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


def icon_button(icon_key, label, on_click=None, btn_id=None, **extra_attrs):
    """建立帶圖示的 action-bar 按鈕。"""
    btn = el("button", cls="btn-action", html=f'{ICONS.get(icon_key, "")}<span>{label}</span>')
    if btn_id:
        btn.id = btn_id
    for k, v in extra_attrs.items():
        btn.setAttribute(k.replace("_", "-"), str(v))
    if on_click:
        btn.addEventListener("click", on_click)
    return btn


# ═══════════════════════════════════════════════════════════════
# API 呼叫
# ═══════════════════════════════════════════════════════════════

async def api_get(url):
    resp = await window.fetch(url)
    data = await resp.json()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


async def api_post(url, body):
    resp = await window.fetch(url, {
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": _json.dumps(body),
    })
    data = await resp.json()
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


def render_markdown(text):
    lines = text.split("\n")
    out = []
    in_code = False
    code_buf = []
    in_list = False

    for line in lines:
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
            continue

        if in_code:
            code_buf.append(line)
            continue

        m = _re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            lvl = len(m[1]) + 2
            out.append(f'<h{lvl} class="md-heading">{_inline(m[2])}</h{lvl}>')
            continue

        m = _re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            if not in_list:
                out.append('<ul class="md-list">')
                in_list = True
            out.append(f"<li>{_inline(m[1])}</li>")
            continue

        if in_list:
            out.append("</ul>")
            in_list = False

        if line.strip() == "":
            out.append("<br>")
        else:
            out.append(f'<p class="md-p">{_inline(line)}</p>')

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
            stream = await window.navigator.mediaDevices.getUserMedia({
                "video": True, "audio": False,
            })
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
        _state["loop_handle"] = window.setInterval(
            lambda: asyncio.ensure_future(capture_and_send()), DETECT_INTERVAL_MS
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

    ws.addEventListener("open", _on_open)
    ws.addEventListener("message", _on_message)
    ws.addEventListener("close", _on_close)
    ws.addEventListener("error", _on_error)

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

def _append_chat(role, content, pending=False):
    chat_log = _dom["chat_log"]
    chat_empty = _dom.get("chat_empty")
    if chat_empty is not None:
        chat_empty.remove()
        _dom["chat_empty"] = None

    div = el("div", cls=f"chat-message chat-{role}")
    if role == "assistant":
        div.innerHTML = render_markdown(content)
    else:
        div.textContent = content
    if pending:
        div.classList.add("is-pending")
    chat_log.appendChild(div)
    chat_log.scrollTop = chat_log.scrollHeight
    return div


async def send_chat(text):
    if not text.strip():
        return
    _dom["chat_input"].value = ""
    _dom["chat_send"].disabled = True
    _append_chat("user", text)
    pending = _append_chat("assistant", "…", pending=True)

    try:
        data = await api_post(CHAT_URL, {"message": text})
        pending.innerHTML = render_markdown(data["reply"])
        pending.classList.remove("is-pending")
    except Exception as exc:
        pending.textContent = f"錯誤：{exc}"
    finally:
        _dom["chat_send"].disabled = False
        _dom["chat_input"].focus()


def clear_chat():
    chat_log = _dom["chat_log"]
    chat_log.innerHTML = ""
    empty = el("div", cls="chat-empty", text="說點什麼開始對話吧")
    chat_log.appendChild(empty)
    _dom["chat_empty"] = empty


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

def _build_topbar():
    topbar = el("header", cls="topbar")

    brand = el("div", cls="brand")
    brand.appendChild(el("div", cls="brand-mark", html=ICONS["logo"]))
    bt = el("div", cls="brand-text")
    bt.appendChild(el("h1", text="AI Modules"))
    bt.appendChild(el("span", text="本機視覺與對話模型 Playground"))
    brand.appendChild(bt)
    topbar.appendChild(brand)

    pill = el("div", cls="status-pill", id="statusPill")
    pill.setAttribute("data-state", "checking")
    pill.setAttribute("title", "點一下重新檢查連線")
    pill.appendChild(el("span", cls="status-dot"))
    status_text = el("span", id="statusText", text="連線中…")
    pill.appendChild(status_text)
    pill.addEventListener("click", lambda _: asyncio.ensure_future(check_health()))
    _dom["status_pill"] = pill
    _dom["status_text"] = status_text
    topbar.appendChild(pill)

    return topbar


def _build_action_bar():
    bar = el("div", cls="action-bar")

    # ── 相機開關 ──
    toggle_btn = icon_button("camera", "開啟相機", on_click=lambda _: toggle_camera())
    toggle_btn.id = "toggleBtn"
    _dom["toggle_btn"] = toggle_btn
    bar.appendChild(toggle_btn)

    # ── 信心值滑桿 ──
    slider_field = el("div", cls="slider-field")
    slider_field.appendChild(el("span", text="信心值"))
    slider = el("input", id="confSlider")
    slider.setAttribute("type", "range")
    slider.setAttribute("min", "0.1")
    slider.setAttribute("max", "0.9")
    slider.setAttribute("step", "0.05")
    slider.setAttribute("value", "0.35")
    conf_val = el("span", cls="val", id="confValue", text="0.35")
    slider.addEventListener("input", lambda e: setattr(conf_val, "textContent", e.target.value))
    slider_field.appendChild(slider)
    slider_field.appendChild(conf_val)
    _dom["conf_slider"] = slider
    bar.appendChild(slider_field)

    # ── 分隔線 ──
    bar.appendChild(el("div", cls="separator"))

    # ── 模型選擇 ──
    model_wrap = el("div", cls="model-select")
    model_wrap.appendChild(svg_span(ICONS["model"].replace('viewBox', 'style="width:15px;height:15px;stroke:var(--accent);flex-shrink:0" viewBox')))
    model_wrap.appendChild(el("span", text="模型"))
    select = el("select", id="modelSelect")
    for val in ("auto", "sinco", "code", "nvidia"):
        opt = el("option", value=val, text=val)
        if val == "sinco":
            opt.selected = True
        select.appendChild(opt)
    select.addEventListener("change", _on_model_change)
    model_wrap.appendChild(select)
    badge = el("span", cls="model-badge", id="modelBadge", text="sinco")
    model_wrap.appendChild(badge)
    _dom["model_badge"] = badge
    bar.appendChild(model_wrap)

    # ── 分隔線 ──
    bar.appendChild(el("div", cls="separator"))

    # ── 角色 ──
    bar.appendChild(icon_button("character", "角色", on_click=lambda _: _on_character_click()))

    # ── 記憶 ──
    bar.appendChild(icon_button("memory", "記憶", on_click=lambda _: asyncio.ensure_future(_on_memory_click())))

    # ── 學習 ──
    bar.appendChild(icon_button("learn", "學習", on_click=lambda _: asyncio.ensure_future(_on_learn_click())))

    # ── 彈性空間 ──
    spacer = el("div", cls="action-spacer")
    bar.appendChild(spacer)

    # ── 延遲顯示 ──
    latency = el("span", id="latency")
    _dom["latency"] = latency
    bar.appendChild(latency)

    # ── 清除對話 ──
    bar.appendChild(icon_button("clear", "清除對話", on_click=lambda _: clear_chat()))

    return bar


def _build_camera_card():
    card = el("section", cls="card")
    head = el("div", cls="card-head")
    head.appendChild(svg_span(ICONS["camera"].replace('viewBox', 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox')))
    ht = el("div")
    ht.appendChild(el("h2", text="物件偵測"))
    ht.appendChild(el("p", text="即時攝影機影像 + YOLO 偵測"))
    head.appendChild(ht)
    card.appendChild(head)

    body = el("div", cls="card-body")

    stage = el("div", id="stage")
    video = el("video", id="video")
    video.setAttribute("autoplay", "")
    video.setAttribute("playsinline", "")
    video.setAttribute("muted", "")
    overlay = el("canvas", id="overlay")
    stage_empty = el("div", cls="stage-empty", id="stageEmpty")
    stage_empty.appendChild(svg_span(ICONS["cameraOff"].replace('viewBox', 'style="width:34px;height:34px;stroke:var(--text-faint)" viewBox')))
    stage_empty.appendChild(el("span", text="相機尚未開啟"))
    stage.appendChild(video)
    stage.appendChild(overlay)
    stage.appendChild(stage_empty)
    body.appendChild(stage)

    _dom["video"] = video
    _dom["overlay"] = overlay
    _dom["stage_empty"] = stage_empty

    results = el("div", id="results")
    _dom["results"] = results
    body.appendChild(results)

    card.appendChild(body)
    return card


def _build_chat_card():
    card = el("section", cls="card")
    head = el("div", cls="card-head")
    head.appendChild(svg_span(ICONS["chat"].replace('viewBox', 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox')))
    ht = el("div")
    ht.appendChild(el("h2", text="對話"))
    ht.appendChild(el("p", text="自建 seq2seq 模型（無外部 AI API）"))
    head.appendChild(ht)
    card.appendChild(head)

    body = el("div", cls="card-body")

    chat_log = el("div", id="chatLog")
    chat_empty = el("div", cls="chat-empty", id="chatEmpty", text="說點什麼開始對話吧")
    chat_log.appendChild(chat_empty)
    _dom["chat_log"] = chat_log
    _dom["chat_empty"] = chat_empty
    body.appendChild(chat_log)

    form = el("form", id="chatForm")
    chat_input = el("input", id="chatInput")
    chat_input.setAttribute("type", "text")
    chat_input.setAttribute("placeholder", "輸入訊息…")
    chat_input.setAttribute("autocomplete", "off")
    _dom["chat_input"] = chat_input
    form.appendChild(chat_input)

    send_btn = el("button", cls="btn", id="chatSend", html=ICONS["send"])
    send_btn.setAttribute("type", "submit")
    _dom["chat_send"] = send_btn
    form.appendChild(send_btn)

    # 用同步 handler 呼叫 e.preventDefault()（事件觸發當下就要同步呼叫），
    # 實際的非同步工作交給 asyncio.ensure_future() 明確排程——跟這個檔案
    # 其餘事件處理的寫法一致，不依賴「傳一個 async function 給
    # addEventListener 會被自動排程」這種容易因 pyodide 版本而異的隱式行為。
    def _on_submit(e):
        e.preventDefault()
        text = chat_input.value.strip()
        if text:
            asyncio.ensure_future(send_chat(text))

    form.addEventListener("submit", _on_submit)
    body.appendChild(form)

    card.appendChild(body)
    return card


def build_ui():
    root = document.getElementById("root")
    root.innerHTML = ""

    shell = el("div", cls="shell")

    shell.appendChild(_build_topbar())
    shell.appendChild(_build_action_bar())

    grid = el("div", cls="grid")
    grid.appendChild(_build_camera_card())
    grid.appendChild(_build_chat_card())
    shell.appendChild(grid)

    footer = el("footer")
    footer.innerHTML = "&copy; Author: Roy Zeng"
    shell.appendChild(footer)

    root.appendChild(shell)


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════
build_ui()
asyncio.ensure_future(check_health())
