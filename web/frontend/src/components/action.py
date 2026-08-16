"""PyScript 按鈕功能列 + 完整前端 UI。

在瀏覽器中由 PyScript (Pyodide) 執行，建構整個頁面的 DOM，
包含頂部功能列、物件偵測區塊、對話區塊，並透過 fetch 呼叫後端 API。
"""

from pyscript import document, window
from js import console
import json as _json
import re as _re

# ═══════════════════════════════════════════════════════════════
# 設定
# ═══════════════════════════════════════════════════════════════
BACKEND_URL = ""
HEALTH_URL = f"{BACKEND_URL}/api/health"
DETECT_URL = f"{BACKEND_URL}/api/detect"
CHAT_URL = f"{BACKEND_URL}/api/chat"
DETECT_INTERVAL_MS = 500
CAPTURE_WIDTH = 640

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
        results_el.textContent = "偵測中…"
        _state["loop_handle"] = setInterval(capture_and_detect, DETECT_INTERVAL_MS)

    _go()


def stop_camera():
    handle = _state.get("loop_handle")
    if handle is not None:
        window.clearInterval(handle)
        _state["loop_handle"] = None

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


async def capture_and_detect():
    if _state["in_flight"]:
        return
    video = _dom["video"]
    if not video.videoWidth:
        return

    _state["in_flight"] = True
    try:
        conf_slider = _dom["conf_slider"]
        overlay = _dom["overlay"]
        ctx = overlay.getContext("2d")
        latency_el = _dom["latency"]
        results_el = _dom["results"]

        cap_canvas = document.createElement("canvas")
        cap_ctx = cap_canvas.getContext("2d")
        scale = CAPTURE_WIDTH / video.videoWidth
        cap_canvas.width = CAPTURE_WIDTH
        cap_canvas.height = round(video.videoHeight * scale)
        cap_ctx.drawImage(video, 0, 0, cap_canvas.width, cap_canvas.height)

        blob = await new Promise(lambda resolve: cap_canvas.toBlob(resolve, "image/jpeg", 0.7))
        form_data = document.createElement("form")  # placeholder — 用 JS FormData
        from js import FormData
        fd = FormData.new()
        fd.append("frame", blob, "frame.jpg")

        import time
        t0 = time.time() * 1000
        resp = await window.fetch(f"{DETECT_URL}?conf={conf_slider.value}", {
            "method": "POST",
            "body": fd,
        })
        data = await resp.json()
        latency_el.textContent = f"{round(time.time() * 1000 - t0)} ms"
        _draw_detections(data, ctx, overlay, results_el)
    except Exception as exc:
        _dom["results"].textContent = f"偵測錯誤：{exc}"
    finally:
        _state["in_flight"] = False


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
    pill.addEventListener("click", lambda _: check_health())
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
    model_wrap.appendChild(ICONS["model"].replace('viewBox', f'style="width:15px;height:15px;stroke:var(--accent);flex-shrink:0" viewBox'))
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
    bar.appendChild(icon_button("memory", "記憶", on_click=lambda _: _on_memory_click()))

    # ── 學習 ──
    bar.appendChild(icon_button("learn", "學習", on_click=lambda _: _on_learn_click()))

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
    head.appendChild(ICONS["camera"].replace('viewBox', f'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox'))
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
    stage_empty.appendChild(ICONS["cameraOff"].replace('viewBox', f'style="width:34px;height:34px;stroke:var(--text-faint)" viewBox'))
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
    head.appendChild(ICONS["chat"].replace('viewBox', f'style="width:18px;height:18px;stroke:var(--accent);flex-shrink:0" viewBox'))
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

    async def _on_submit(e):
        e.preventDefault()
        text = chat_input.value.strip()
        if text:
            await send_chat(text)

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
check_health()
