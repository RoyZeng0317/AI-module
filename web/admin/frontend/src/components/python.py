"""
PyScript 按鈕功能列 + 完整前端 UI。

在瀏覽器中由 PyScript (Pyodide) 執行，建構整個頁面的 DOM，
包含頂部功能列、物件偵測區塊、對話區塊，並透過 fetch 呼叫後端 API。
"""

import asyncio
import json as _json
import re as _re
from js import URL as _JsURL
from js import WebSocket, console
from pyscript import document, window

# ═══════════════════════════════════════════════════════════════
# 1. 系統設定 (Configuration)
# ═══════════════════════════════════════════════════════════════

# 留空字串代表「後端跟這個頁面同網域」
BACKEND_URL = ""


def _api_url(path: str) -> str:
    if BACKEND_URL:
        return f"{BACKEND_URL}/{path}"
    return path


HEALTH_URL = _api_url("api/health")
CHAT_URL = _api_url("api/chat")
GPU_INFO_URL = _api_url("api/gpu_info")
DETECT_INTERVAL_MS = 500
CAPTURE_WIDTH = 640


def _detect_ws_url() -> str:
    """算出 /ws/detect 的 ws(s) 網址。

    使用 window.location 的 protocol/host/pathname 進行動態拼接，
    防止在 about:srcdoc 等無效 base URL 環境下拋出 TypeError。
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
# 2. SVG 圖示庫 (SVG Icons)
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
# 3. DOM 輔助工具 (DOM Helpers)
# ═══════════════════════════════════════════════════════════════


def el(tag: str, cls: str | None = None, text: str | None = None, html: str | None = None, **attrs):
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


def svg_span(html_str: str, cls: str | None = None):
    """將 SVG 字串包裝為 DOM span 節點以供 appendChild 使用。"""
    return el("span", cls=cls, html=html_str)


def icon_button(icon_key: str, label: str, on_click=None, btn_id: str | None = None, **extra_attrs):
    """建立帶有圖示與文字的按鈕。"""
    btn = el("button", cls="btn-action", html=f'{ICONS.get(icon_key, "")}<span>{label}</span>')
    if btn_id:
        btn.id = btn_id
    for k, v in extra_attrs.items():
        btn.setAttribute(k.replace("_", "-"), str(v))
    if on_click:
        btn.addEventListener("click", on_click)
    return btn

# ═══════════════════════════════════════════════════════════════
# 4. API 通訊 (API Utilities)
# ═══════════════════════════════════════════════════════════════


async def api_get(url: str):
    resp = await window.fetch(url)
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data


async def api_post(url: str, body: dict):
    resp = await window.fetch(
        url,
        {
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": _json.dumps(body),
        },
    )
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data

# ═══════════════════════════════════════════════════════════════
# 5. Markdown 解析器 (Markdown Parser)
# ═══════════════════════════════════════════════════════════════

_ESCAPE_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


def _escape(s: str) -> str:
    for ch, ent in _ESCAPE_MAP.items():
        s = s.replace(ch, ent)
    return s


def _inline(line: str) -> str:
    h = _escape(line)
    h = _re.sub(
        r"\*\*(.+?)\*\*|__(.+?)__",
        lambda m: f"<strong>{m.group(1) or m.group(2) or ''}</strong>",
        h,
    )
    h = _re.sub(r"`([^`]+)`", lambda m: f'<code class="md-code-inline">{m.group(1) or ""}</code>', h)
    h = _re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", lambda m: f'<em>{m.group(1) or ""}</em>', h)
    return h


_TABLE_ROW_RE = _re.compile(r"^\s*\|(.*)\|\s*$")
_TABLE_SEP_CELL_RE = _re.compile(r"^:?-+:?$")


def _split_table_row(line: str) -> list[str]:
    m = _TABLE_ROW_RE.match(line)
    if not m:
        return []
    inner = m.group(1)
    if not inner:
        return []
    return [cell.strip() for cell in inner.split("|")]


def _is_table_separator(line: str) -> bool:
    if not _TABLE_ROW_RE.match(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_SEP_CELL_RE.match(c) for c in cells)


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    thead = "".join(f"<th>{_inline(c)}</th>" for c in header)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_inline(row[i] if i < len(row) else '')}</td>" for i in range(len(header)))
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="md-table-wrap"><table class="md-table"><thead><tr>'
        + thead
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_markdown(text: str) -> str:
    lines = text.split("\n")
    n_lines = len(lines)
    out = []
    in_code = False
    code_buf = []
    in_list = False

    i = 0
    while i < n_lines:
        line = lines[i]

        # 程式碼區塊
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

        # 表格處理
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

        # 標題 (h1~h6 限制最高至 h6)
        m = _re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            g1, g2 = m.group(1), m.group(2)
            if g1 and g2:
                lvl = min(len(g1) + 2, 6)
                out.append(f'<h{lvl} class="md-heading">{_inline(g2)}</h{lvl}>')
            i += 1
            continue

        # 清單
        m = _re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            if not in_list:
                out.append('<ul class="md-list">')
                in_list = True
            content = m.group(1)
            if content:
                out.append(f"<li>{_inline(content)}</li>")
            i += 1
            continue

        if in_list:
            out.append("</ul>")
            in_list = False

        # 一般段落與換行
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
# 6. 全域狀態與 DOM 快取 (State & Global DOM Container)
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

_dom = {}

# ═══════════════════════════════════════════════════════════════
# 7. 系統服務 (System Services: Health, GPU, Camera)
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


async def check_gpu():
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


def start_camera():
    video = _dom["video"]
    overlay = _dom["overlay"]
    stage_empty = _dom["stage_empty"]
    toggle_btn = _dom["toggle_btn"]
    results_el = _dom["results"]

    async def _go():
        try:
            stream = await window.navigator.mediaDevices.getUserMedia({"video": True, "audio": False})
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
        _state["loop_handle"] = window.setInterval(
            lambda: asyncio.ensure_future(capture_and_send()), DETECT_INTERVAL_MS
        )

    asyncio.ensure_future(_go())


def _open_detect_socket(conf):
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
# 8. 對話邏輯與事件處理 (Chat & Handlers)
# ═══════════════════════════════════════════════════════════════


def _append_chat(role: str, content: str, pending: bool = False):
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


async def send_chat(text: str):
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
# 9. 介面組件建構 (UI Components Construction)
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

    pills_wrap = el("div", cls="pills-wrap")

    # 後端連線狀態
    pill = el("div", cls="status-pill", id="statusPill")
    pill.setAttribute("data-state", "checking")
    pill.setAttribute("title", "點一下重新檢查連線")
    pill.appendChild(el("span", cls="status-dot"))
    status_text = el("span", id="statusText", text="連線中…")
    pill.appendChild(status_text)
    pill.addEventListener("click", lambda _: asyncio.ensure_future(check_health()))
    _dom["status_pill"] = pill
    _dom["status_text"] = status_text
    pills_wrap.appendChild(pill)

    # 顯卡狀態
    gpu_pill = el("div", cls="status-pill", id="gpuPill")
    gpu_pill.setAttribute("data-state", "checking")
    gpu_pill.setAttribute("title", "點一下重新查詢顯卡資訊")
    gpu_pill.appendChild(el("span", cls="status-dot"))
    gpu_text = el("span", id="gpuText", text="顯卡查詢中…")
    gpu_pill.appendChild(gpu_text)
    gpu_pill.addEventListener("click", lambda _: asyncio.ensure_future(check_gpu()))
    _dom["gpu_pill"] = gpu_pill
    _dom["gpu_text"] = gpu_text
    pills_wrap.appendChild(gpu_pill)

    topbar.appendChild(pills_wrap)
    return topbar


def _build_action_bar():
    bar = el("div", cls="action-bar")

    # 相機開關
    toggle_btn = icon_button("camera", "開啟相機", on_click=lambda _: toggle_camera())
    toggle_btn.id = "toggleBtn"
    _dom["toggle_btn"] = toggle_btn
    bar.appendChild(toggle_btn)

    # 信心值滑桿
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

    bar.appendChild(el("div", cls="separator"))

    # 模型選擇
    model_wrap = el("div", cls="model-select")
    model_wrap.appendChild(
        svg_span(ICONS["model"].replace("viewBox", 'style="width:15px;height:15px;stroke:var(--accent);flex-shrink:0" viewBox'))
    )
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

    bar.appendChild(el("div", cls="separator"))

    # 功能按鈕群
    bar.appendChild(icon_button("character", "角色", on_click=lambda _: _on_character_click()))
    bar.appendChild(icon_button("memory", "記憶", on_click=lambda _: asyncio.ensure_future(_on_memory_click())))
    bar.appendChild(icon_button("learn", "學習", on_click=lambda _: asyncio.ensure_future(_on_learn_click())))

    # 彈性間距與資訊
    bar.appendChild(el("div", cls="action-spacer"))
    latency = el("span", id="latency")
    _dom["latency"] = latency
    bar.appendChild(latency)

    # 清除對話
    bar.appendChild(icon_button("clear", "清除對話", on_click=lambda _: clear_chat()))

    return bar


def _build_camera_card():
    card = el("section", cls="card")
    head = el("div", cls="card-head")
    head.appendChild(
        svg_span(ICONS["camera"].replace("viewBox", 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox'))
    )
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
    stage_empty.appendChild(
        svg_span(ICONS["cameraOff"].replace("viewBox", 'style="width:34px;height:34px;stroke:var(--text-faint)" viewBox'))
    )
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
    head.appendChild(
        svg_span(ICONS["chat"].replace("viewBox", 'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox'))
    )
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
# 10. 初始化執行 (App Initialization)
# ═══════════════════════════════════════════════════════════════

build_ui()
asyncio.ensure_future(check_health())
asyncio.ensure_future(check_gpu())