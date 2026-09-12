"""FastAPI backend: serves the static frontend and a YOLO object-detection
endpoint that the browser posts webcam frames to.

Run:
    cd backend && python app.py
    # or: uvicorn app:app --reload --port 8000   (from inside backend/)
Then open http://localhost:8080/
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
# 真正會被瀏覽器執行的 PyScript 前端（index.html/style.css/action.py，以及
# 純 Python 的 gpu_info.py）在 admin/frontend/src/components/ 底下——
# web/frontend/src/components/ 是另一份給 Firebase 靜態網頁託管用的
# （見 web/firebase.json 的 "public": "frontend/src/components"），
# 不是這支後端要服務的對象，兩者不能混用。
FRONTEND_DIR = BACKEND_DIR.parent / "admin" / "frontend" / "src" / "components"
TRAINING_DIR = BACKEND_DIR.parent.parent / "tranning"
# memory_store.py：/memory 指令共用的持久記憶儲存（見下方 chat_endpoint 對
# "/memory"、"/learn" 的本機攔截），跟桌面 GUI（command.py）、CLI（cli.py）
# 用的是同一支模組、同一份 memory/memory.json，不是各端各自一份。
COMPONENTS_DIR = BACKEND_DIR.parent.parent / "lib" / "components"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(FRONTEND_DIR))
sys.path.insert(0, str(COMPONENTS_DIR))

from typing import Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import auto_learn
import conversation_store as convo_store
import notify_store
import usage_store
from auth import verify_id_token
from chats import smart_reply
from detector import detect
from gpu_info import get_gpu_info
from memory_store import CATEGORIES as MEMORY_CATEGORIES
from memory_store import add_memory, delete_memory, format_memories, list_memories

# Only these files are served. frontend/src/ (.env, component sources beyond
# these three) and frontend/data/ (basic_data.sql) must never be reachable
# over HTTP.
PUBLIC_FILES = {"index.html", "style.css", "action.py", "script.js", "google-auth-init.js"}

# 部署到子路徑後面（例如 Tailscale Serve 的 `--set-path=/AI-Module`）時，
# 反向代理原樣把完整路徑轉過來，後端這邊的路由也要掛在同一個前綴下才會對得
# 上；同域部署（Render.com、本機 localhost:8000 直接開）維持空字串跑在根目
# 錄，行為跟改之前完全一樣。用環境變數帶入，不寫死在程式碼裡，才能同一份
# app.py 同時給 Render 跟 Pi5 兩種部署方式用。
MOUNT_PREFIX = os.environ.get("MOUNT_PREFIX", "").rstrip("/")


class ChatRequest(BaseModel):
    message: str
    # 沒帶 conversation_id 時維持原本無狀態行為（不寫入任何存檔）——只有
    # action.py 側邊欄「對話紀錄」這條新流程才會帶 id 進來，直接呼叫這支
    # API 的舊用法/其他呼叫端不會被逼著配合這個新功能。
    conversation_id: Optional[str] = None


class ConversationRenameRequest(BaseModel):
    title: str


class NotificationCreateRequest(BaseModel):
    message: str


def _resolve_client_id(request: Request) -> str:
    """有帶 Google 登入的 Authorization header 就用帳號識別（"google:<uid>"），
    沒有（或驗證失敗）就退回連線 IP——舊的、不帶 token 的呼叫端行為不變。"""
    claims = verify_id_token(request.headers.get("authorization"))
    if claims:
        return f"google:{claims['sub']}"
    return request.client.host


# 單一使用者一段時間內能送出的聊天請求數量上限。key 優先用 Google 帳號識別
# （見 _resolve_client_id），沒登入才退回來源 IP。狀態存在記憶體裡（重啟就重
# 置），跟這個專案「單機/私人用」的規模相符，不需要 Redis 之類的外部儲存。
CHAT_RATE_LIMIT = 20
CHAT_RATE_WINDOW_SEC = 60
_chat_request_log: dict[str, list[float]] = defaultdict(list)


def _check_chat_rate_limit(client_ip: str) -> None:
    now = time.time()
    log = _chat_request_log[client_ip]
    log[:] = [t for t in log if now - t < CHAT_RATE_WINDOW_SEC]
    if len(log) >= CHAT_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"請求過於頻繁，請 {CHAT_RATE_WINDOW_SEC} 秒後再試",
        )
    log.append(now)


app = FastAPI(title="AI-module camera object detection")
router = APIRouter(prefix=MOUNT_PREFIX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Starlette's default handler for an uncaught exception returns plain text
# ("Internal Server Error"), not JSON. script.js always calls res.json() on
# API responses, so a plain-text 500 makes fetch throw a confusing
# "Unexpected token 'I'..." (or, if the connection drops mid-crash,
# "Unexpected end of JSON input") instead of a readable error. Force every
# uncaught error on /api/* to still be valid JSON.
@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/gpu_info")
async def gpu_info_endpoint():
    return get_gpu_info()


@router.post("/api/detect")
async def detect_frame(frame: UploadFile = File(...), conf: float = 0.35):
    data = await frame.read()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"detections": [], "width": 0, "height": 0, "error": "could not decode image"}
    h, w = image.shape[:2]
    return {"detections": detect(image, conf=conf), "width": w, "height": h}


@router.websocket("/ws/detect")
async def detect_stream(ws: WebSocket, conf: float = 0.35):
    # action.py 開一條連線就一直用（相機開著就不斷），取代原本 /api/detect
    # 每一格畫面都重開一次 HTTP 連線的做法——部署到 Pi5 這種效能較弱的裝置
    # 上，省掉的 TCP/TLS handshake 開銷對即時串流的延遲影響會比較明顯。
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                await ws.send_json({"detections": [], "width": 0, "height": 0, "error": "could not decode image"})
                continue
            h, w = image.shape[:2]
            await ws.send_json({"detections": detect(image, conf=conf), "width": w, "height": h})
    except WebSocketDisconnect:
        pass


@router.get("/api/conversations")
async def list_conversations_endpoint():
    return convo_store.list_conversations()


@router.post("/api/conversations")
async def create_conversation_endpoint():
    return convo_store.create_conversation()


@router.get("/api/conversations/{conversation_id}")
async def get_conversation_endpoint(conversation_id: str):
    conv = convo_store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@router.patch("/api/conversations/{conversation_id}")
async def rename_conversation_endpoint(conversation_id: str, payload: ConversationRenameRequest):
    conv = convo_store.rename_conversation(conversation_id, payload.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@router.post("/api/conversations/{conversation_id}/clear")
async def clear_conversation_endpoint(conversation_id: str):
    conv = convo_store.clear_messages(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    if not convo_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"status": "deleted"}


# 通知：action.py（admin 端）的「發送通知」彈窗呼叫 POST 寫入一筆，
# python.py（使用者端）的通知面板輪詢 GET 讀出來顯示。
@router.get("/api/notifications")
async def list_notifications_endpoint():
    return notify_store.list_notifications()


@router.post("/api/notifications")
async def create_notification_endpoint(payload: NotificationCreateRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    return notify_store.create_notification(payload.message.strip())


@router.post("/api/notifications/mark_all_read")
async def mark_all_notifications_read_endpoint():
    return notify_store.mark_all_read()


# /memory、/learn 是跟桌面 GUI（command.py 的 CommandPalette._run_memory/
# _run_learn）、CLI（cli.py 的 run_memory/run_learn）同一套本機、非模型指令，
# 邏輯搬過來這裡沿用同一份 memory_store.py / auto_learn.py 存檔。在這之前
# /api/chat 沒有攔截任何 "/" 開頭的訊息，導致 action.py 側邊欄「記憶」「學習」
# 按鈕送出的 "/memory"、"/learn" 字串被當成一般句子直接餵給 smart_reply()，
# 讓還沒訓練好、字元級的 sinco 模型輸出「ho onco he he he he he」這類無意義
# 重複字元（診斷依據：使用者回報按下「記憶」按鈕出現這段亂碼）。
def _run_memory_command(arg: str) -> str:
    sub, _, rest = arg.partition(" ")
    sub, rest = sub.strip().lower(), rest.strip()

    if sub in ("", "list"):
        category = rest.lower() or None
        if category and category not in MEMORY_CATEGORIES:
            return f"未知分類：{category}（可用：{', '.join(MEMORY_CATEGORIES)}）"
        return format_memories(list_memories(category))

    if sub == "add":
        category, _, text = rest.partition(" ")
        try:
            entry = add_memory(category, text)
        except ValueError as exc:
            return str(exc)
        return f"已新增記憶 [{entry['id']}]：{entry['text']}"

    if sub in ("del", "delete"):
        if delete_memory(rest):
            return f"已刪除記憶 [{rest}]"
        return f"找不到記憶 id：{rest}"

    return (
        "用法：\n"
        "  /memory                    列出全部記憶\n"
        "  /memory list <分類>        列出指定分類\n"
        f"  /memory add <分類> <內容>  新增（分類：{', '.join(MEMORY_CATEGORIES)}）\n"
        "  /memory del <id>           刪除"
    )


def _run_learn_command(arg: str) -> str:
    sub, _, rest = arg.partition(" ")
    sub, rest = sub.strip().lower(), rest.strip()

    if sub in ("", "list"):
        return auto_learn.format_candidates(auto_learn.list_candidates())

    if sub == "approve":
        if auto_learn.approve_candidate(rest):
            return f"已核准並寫入 data/pairs.json：[{rest}]（尚未重訓，記得之後手動執行 chats.py）"
        return f"找不到候選 id：{rest}"

    if sub == "reject":
        if auto_learn.reject_candidate(rest):
            return f"已捨棄候選：[{rest}]"
        return f"找不到候選 id：{rest}"

    return (
        "用法：\n"
        "  /learn                 列出目前待審核的候選學習內容\n"
        "  /learn approve <id>    核准並寫入 data/pairs.json\n"
        "  /learn reject <id>     捨棄候選"
    )


def _run_local_command(message: str) -> str | None:
    """message 是本機指令（/memory、/learn）就回傳結果文字，否則回傳 None
    交給呼叫端繼續走 smart_reply()。"""
    if not message.startswith("/"):
        return None
    name, _, arg = message[1:].partition(" ")
    name, arg = name.strip().lower(), arg.strip()
    if name == "memory":
        return _run_memory_command(arg)
    if name == "learn":
        return _run_learn_command(arg)
    return None


@router.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, request: Request):
    client_id = _resolve_client_id(request)
    _check_chat_rate_limit(client_id)

    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if usage_store.get_usage(client_id) >= usage_store.USAGE_CHAR_LIMIT:
        window_hours = usage_store.USAGE_WINDOW_SEC // 3600
        raise HTTPException(
            status_code=429,
            detail=f"已達 {window_hours} 小時內的字數使用上限（{usage_store.USAGE_CHAR_LIMIT} 字），請稍後再試",
        )

    conversation_id = payload.conversation_id
    if conversation_id is not None and convo_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    if conversation_id is not None:
        convo_store.append_message(conversation_id, "user", payload.message)

    local_reply = _run_local_command(payload.message.strip())
    reply = local_reply if local_reply is not None else smart_reply(payload.message)

    # 用量算輸入+輸出的字元總和，比較貼近實際運算量——sinco 是字元級模型，
    # 沒有 subword/BPE token 這種單位（見 usage_store.py 開頭說明）。
    usage_store.add_usage(client_id, len(payload.message) + len(reply))

    if conversation_id is not None:
        convo_store.append_message(conversation_id, "assistant", reply)

    return {"reply": reply, "conversation_id": conversation_id}


@router.get("/api/usage")
async def usage_endpoint(request: Request):
    """查詢呼叫端目前的用量 vs 上限，用來確認 rate limit／用量上限有沒有生效
    （例如手動測試，或之後接進 admin 端畫面）。"""
    used = usage_store.get_usage(_resolve_client_id(request))
    return {
        "chars_used": used,
        "char_limit": usage_store.USAGE_CHAR_LIMIT,
        "window_hours": usage_store.USAGE_WINDOW_SEC // 3600,
    }


# 手機瀏覽器（例如三星瀏覽器）沒有桌面 DevTools 的「Disable cache」選項，
# 改完 action.py/index.html 之後使用者很難手動清快取，所以直接在回應加上
# no-cache：瀏覽器每次都要跟伺服器確認檔案有沒有變過（ETag/Last-Modified
# 條件式請求）才能用快取版本，不會再吃到修正前的舊版程式碼。
_NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}


@router.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@router.get("/{filename}")
async def public_file(filename: str):
    if filename not in PUBLIC_FILES:
        raise HTTPException(status_code=404)
    return FileResponse(FRONTEND_DIR / filename, headers=_NO_CACHE_HEADERS)


app.include_router(router)

if MOUNT_PREFIX:
    # 直接用 Pi5 的區網 IP/hostname 打根目錄（沒有經過 tailscale serve 的
    # /AI-Module 前綴）方便測試時還能導到正確位置，不會只看到 404。
    @app.get("/")
    async def _root_redirect():
        return RedirectResponse(url=f"{MOUNT_PREFIX}/")


if __name__ == "__main__":
    import uvicorn
    # Cloud Run (and most container platforms) inject the port to listen on
    # via $PORT; default to 8000 for local dev where it's unset.
    uvicorn.run(app, host="localhost", port=int(os.environ.get("PORT", 8080)))
