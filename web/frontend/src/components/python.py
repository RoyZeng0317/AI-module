"""index.html 的 PyScript 進入點（走 Firebase Hosting 靜態部署，見
web/firebase.json 的 "public": "frontend/src/components"）。

只補齊 index.html 裡兩塊還沒有任何行為的靜態標記，不新增/更動 index.html
或 style.css 的任何內容——全部用 document.querySelector 抓現有元素、在
執行期補行為：
  1. `.user` 的使用者須知——原本整份清單直接攤開擠在畫面上，改成預設收合，
     點一下 <span> 才展開/收合，對齊原本註解「點選連結後要跳出視窗」的
     需求。
  2. `.message-box` 的訊息輸入框——原本只是一個沒有任何行為的 <input>，
     補上 Enter 送出、呼叫後端 /api/chat、把回覆顯示出來。

跟 web/admin/frontend/src/components/action.py 是同一套 PyScript 慣例：
create_proxy() 包 callback、to_js(..., dict_converter=Object.fromEntries)
把 fetch() 的選項轉成 plain object（pyodide 預設把 Python dict 轉成
JS Map，fetch() 用 .method/.headers/.body 這種一般物件屬性讀設定，Map
沒有這些屬性，讀出來全部是 undefined）。
"""

from pyscript import document, window
from js import console, Object
from pyodide.ffi import create_proxy, to_js
import asyncio
import json as _json

# 這份前端跟後端不一定同網域（Firebase Hosting vs FastAPI 後端）——跟
# action.py 的 BACKEND_URL 慣例一樣：同網域部署留空字串即可（相對路徑），
# 不同網域時要填後端完整網址，例如 "https://xxx.a.run.app"。
BACKEND_URL = ""


def _api_url(path):
    return f"{BACKEND_URL}/{path}" if BACKEND_URL else path


CHAT_URL = _api_url("api/chat")


def _js_opts(opts):
    return to_js(opts, dict_converter=Object.fromEntries)


async def _send_chat(text):
    resp = await window.fetch(CHAT_URL, _js_opts({
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": _json.dumps({"message": text}),
    }))
    data = (await resp.json()).to_py()
    if not resp.ok:
        raise Exception(data.get("detail", f"HTTP {resp.status}"))
    return data["reply"]


# ═══════════════════════════════════════════════════════════════
# 使用者須知：預設收合，點 <span> 展開/收合
# ═══════════════════════════════════════════════════════════════

def _setup_notice():
    notice = document.querySelector(".user")
    if notice is None:
        return
    label = notice.querySelector("span")
    body = notice.querySelector("ul")
    if label is None or body is None:
        return

    body.style.display = "none"
    label.style.cursor = "pointer"

    def _toggle(_evt):
        body.style.display = "block" if body.style.display == "none" else "none"

    label.addEventListener("click", create_proxy(_toggle))


# ═══════════════════════════════════════════════════════════════
# 訊息輸入框：Enter 送出、顯示後端回覆
# ═══════════════════════════════════════════════════════════════

def _setup_message_box():
    container = document.querySelector(".message-box")
    input_el = document.querySelector(".message-box input")
    if container is None or input_el is None:
        return

    reply_el = document.createElement("div")
    reply_el.className = "message-reply"
    container.appendChild(reply_el)

    async def _submit(text):
        input_el.disabled = True
        reply_el.textContent = "傳送中…"
        try:
            reply_el.textContent = await _send_chat(text)
        except Exception as exc:
            reply_el.textContent = f"錯誤：{exc}"
            console.error(f"chat request failed: {exc}")
        finally:
            input_el.disabled = False
            input_el.value = ""
            input_el.focus()

    def _on_keydown(evt):
        if evt.key == "Enter":
            evt.preventDefault()
            text = input_el.value.strip()
            if text:
                asyncio.ensure_future(_submit(text))

    input_el.addEventListener("keydown", create_proxy(_on_keydown))


# ═══════════════════════════════════════════════════════════════
# 初始化：index.html 把這支腳本的 <script> 放在 <head>，執行時 <body>
# 可能還沒解析完——保底等 DOMContentLoaded 再抓元素，避免 querySelector
# 抓到 None 而整段功能悄悄失效。
# ═══════════════════════════════════════════════════════════════

def _init():
    _setup_notice()
    _setup_message_box()


if document.readyState == "loading":
    document.addEventListener("DOMContentLoaded", create_proxy(lambda _e: _init()))
else:
    _init()
