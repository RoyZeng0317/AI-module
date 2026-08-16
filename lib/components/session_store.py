"""session_store.py — 對話紀錄的自動存檔，接給 `/resume` 指令用。

跟 memory_store.py（使用者自己選分類手動寫入的長期記憶）是不同機制：這裡是
**每一輪對話都自動、即時寫檔**，不需要使用者主動存檔——目的是即使電腦意外斷電
或直接關機（不是正常關閉程式跑到 exit 那行），已經完成的對話輪次也不會遺失，
`/resume` 才能真的「回到電腦未關機前所記錄下的對話」。GUI（conversation.py 的
Conversation.ask()）跟 CLI（cli.py 的 ask_model()）各自在拿到回覆後呼叫
record_turn()，兩邊共用同一份 session.json、同一套格式。

一條紀錄存成 {"id", "persona", "mode", "user", "reply", "created_at"}，全部
紀錄放在同一個 JSON 陣列檔案 session.json 裡，跟 memory_store.py 的
memory.json 同一層資料夾。只保留最近 MAX_SESSION_TURNS 輪，避免檔案無限長大
——這是「重開機前最後在聊什麼」的還原用途，不是要當成永久聊天紀錄庫。
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller onefile (nova.exe, built to live at the project root): see
    # lib/components/cli.py for why __file__ can't be used here when frozen.
    MEMORY_DIR = Path(sys.executable).resolve().parent / "memory"
else:
    MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
SESSION_PATH = MEMORY_DIR / "session.json"

# 只保留最近這麼多輪——太舊的紀錄對「回到重開機前的對話」這個用途沒有幫助，
# 卻會讓檔案無限長大、/resume 重播畫面洗版洗不完。
MAX_SESSION_TURNS = 200


def _resolve_path(path: Path | None) -> Path:
    # 跟 memory_store.py 同樣的理由：不能把 path=None 綁死成函式預設引數的值，
    # 否則 monkeypatch SESSION_PATH 之後，已經綁定的預設值不會跟著變。
    return path if path is not None else SESSION_PATH


def _load(path: Path | None = None) -> list[dict]:
    path = _resolve_path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(entries: list[dict], path: Path | None = None) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def record_turn(user_text: str, reply: str, persona: str = "sinco", mode: str = "auto",
                 path: Path | None = None) -> dict:
    """記下這一輪對話。在拿到模型回覆的當下就呼叫（不是等使用者離開程式才存），
    才能保證意外斷電/關機時，已經完成的輪次已經落地。"""
    entry = {
        "id": uuid.uuid4().hex[:8],
        "persona": persona,
        "mode": mode,
        "user": user_text,
        "reply": reply,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries = _load(path)
    entries.append(entry)
    del entries[:-MAX_SESSION_TURNS]
    _save(entries, path)
    return entry


def load_session(path: Path | None = None) -> list[dict]:
    return _load(path)


def clear_session(path: Path | None = None) -> None:
    _save([], path)
