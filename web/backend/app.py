"""FastAPI backend: serves the static frontend and a YOLO object-detection
endpoint that the browser posts webcam frames to.

Run:
    cd backend && python app.py
    # or: uvicorn app:app --reload --port 8000   (from inside backend/)
Then open http://localhost:8000/
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
TRAINING_DIR = BACKEND_DIR.parent.parent / "tranning"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(TRAINING_DIR))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from chats import smart_reply
from web.backend.detector import detect

# Only these files are served. frontend/src/ (.env, component sources) and
# frontend/data/ (basic_data.sql) must never be reachable over HTTP.
PUBLIC_FILES = {"index.html", "script.js", "style.css"}


class ChatRequest(BaseModel):
    message: str


app = FastAPI(title="AI-module camera object detection")

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect_frame(frame: UploadFile = File(...), conf: float = 0.35):
    data = await frame.read()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"detections": [], "width": 0, "height": 0, "error": "could not decode image"}
    h, w = image.shape[:2]
    return {"detections": detect(image, conf=conf), "width": w, "height": h}


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    return {"reply": smart_reply(payload.message)}


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{filename}")
async def public_file(filename: str):
    if filename not in PUBLIC_FILES:
        raise HTTPException(status_code=404)
    return FileResponse(FRONTEND_DIR / filename)


if __name__ == "__main__":
    import os
    import uvicorn
    # Cloud Run (and most container platforms) inject the port to listen on
    # via $PORT; default to 8000 for local dev where it's unset.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
