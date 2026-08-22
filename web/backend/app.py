"""FastAPI backend: serves the static frontend and a YOLO object-detection
endpoint that the browser posts webcam frames to.

Run:
    cd backend && python app.py
    # or: uvicorn app:app --reload --port 8000   (from inside backend/)
Then open http://localhost:8000/
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
# 真正會被瀏覽器執行的 PyScript 前端（index.html/style.css/action.py）在
# frontend/src/components/ 底下，不是 frontend/ 這一層——原本指到 frontend/
# 這層是既有 bug（見 ErrorLog.md 第17條，這兩個測試原本就會失敗：那一層
# 目前沒有任何檔案）。
FRONTEND_DIR = BACKEND_DIR.parent / "frontend" / "src" / "components"
TRAINING_DIR = BACKEND_DIR.parent.parent / "tranning"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(TRAINING_DIR))

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

from chats import smart_reply
from detector import detect

# Only these files are served. frontend/src/ (.env, component sources beyond
# these three) and frontend/data/ (basic_data.sql) must never be reachable
# over HTTP.
PUBLIC_FILES = {"index.html", "style.css", "action.py"}

# 部署到子路徑後面（例如 Tailscale Serve 的 `--set-path=/AI-Module`）時，
# 反向代理原樣把完整路徑轉過來，後端這邊的路由也要掛在同一個前綴下才會對得
# 上；同域部署（Render.com、本機 localhost:8000 直接開）維持空字串跑在根目
# 錄，行為跟改之前完全一樣。用環境變數帶入，不寫死在程式碼裡，才能同一份
# app.py 同時給 Render 跟 Pi5 兩種部署方式用。
MOUNT_PREFIX = os.environ.get("MOUNT_PREFIX", "").rstrip("/")


class ChatRequest(BaseModel):
    message: str


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


@router.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    return {"reply": smart_reply(payload.message)}


@router.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/{filename}")
async def public_file(filename: str):
    if filename not in PUBLIC_FILES:
        raise HTTPException(status_code=404)
    return FileResponse(FRONTEND_DIR / filename)


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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
