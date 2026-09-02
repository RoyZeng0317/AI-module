"""conversation_store.py — 三端（桌面 GUI／終端機 CLI／網頁）共用的多筆對話
持久化儲存，存在 memory/conversations.json，跟 memory/session.json、
memory/memory.json 同一層資料夾。

跟 session_store.py 不是同一件事：session_store.py 是「重開機前最後聊到哪」的
單一連續 rolling window（給 `/resume` 用，只保留最近 MAX_SESSION_TURNS 輪，
該檔案文件裡講明「不是要當成永久聊天紀錄庫」）；這裡才是真正永久、可以開
多筆、各自命名的對話紀錄——GUI（conversation.py）、CLI（cli.py）、網頁
（web/backend/app.py）三端都呼叫這裡的函式，讓同一個人不管從哪個介面聊天，
「對話紀錄」看到的都是同一份資料。

跟 web/backend/conversation_store.py 幾乎是同一份程式碼，故意兩邊各自維護一份、
而不是互相 import 共用同一支模組：web 那邊部署到 Render/樹莓派時（見
web/render.yaml、web/Dockerfile）容器映像檔只會複製 web/backend、web/frontend
兩個資料夾，不會連帶把整個 lib/（含 tkinter、sounddevice 等桌面版才需要的重
依賴）一起打包進去，兩邊沒辦法真的 import 同一支檔案。改成兩份程式碼各自算出
「專案根目錄下的 memory/ 資料夾」這個同一個實體路徑，本機開發時兩邊讀寫的就是
同一個 conversations.json，效果上等於共用同一份紀錄——跟 memory_store.py／
session_store.py／web/backend/conversation_store.py 之間「兩份持久化模組維持
同一套慣例」是同一種設計取捨，只是這次連檔案格式都要完全一致才能真的共用到
同一份資料，不只是風格一致而已。改動任何一邊的 schema 要記得同步改另一邊。

一筆對話存成：
{
    "id": "8 碼 hex",
    "title": str（預設 "新對話"，收到第一則使用者訊息後自動改成訊息開頭）,
    "created_at": ISO8601,
    "updated_at": ISO8601,
    "messages": [{"role": "user"/"assistant", "content": str, "ts": ISO8601,
                  "persona": str（選填，只有 GUI/CLI 會帶，即 /character 目前人格）,
                  "mode": str（選填，只有 GUI/CLI 會帶，即 /model 的 force_mode）}]
}
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller onefile (nova.exe，建置成放在專案根目錄)：見
    # lib/components/session_store.py 同樣的理由，frozen 時 __file__ 不能用。
    MEMORY_DIR = Path(sys.executable).resolve().parent / "memory"
else:
    MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
CONVERSATIONS_PATH = MEMORY_DIR / "conversations.json"

DEFAULT_TITLE = "新對話"
_TITLE_MAX_LEN = 24


def _now() -> str:
    # 微秒精度，讓同一秒內連續建立/更新多筆對話時，updated_at 排序仍然穩定。
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _resolve_path(path: Path | None) -> Path:
    # path=None 時「即時」讀取模組全域的 CONVERSATIONS_PATH：函式預設引數只在
    # 定義當下求值一次，測試 monkeypatch 模組全域變數不會反映到已經綁定的預設值上。
    return path if path is not None else CONVERSATIONS_PATH


def _load(path: Path | None = None) -> list[dict]:
    path = _resolve_path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(conversations: list[dict], path: Path | None = None) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")


def _find(conversations: list[dict], conversation_id: str) -> dict | None:
    for conv in conversations:
        if conv["id"] == conversation_id:
            return conv
    return None


def _summary(conv: dict) -> dict:
    return {
        "id": conv["id"],
        "title": conv["title"],
        "updated_at": conv["updated_at"],
    }


def list_conversations(path: Path | None = None) -> list[dict]:
    """回傳所有對話的摘要（不含訊息內容），依最後更新時間新到舊排序。"""
    conversations = _load(path)
    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return [_summary(c) for c in conversations]


def create_conversation(path: Path | None = None) -> dict:
    conv = {
        "id": uuid.uuid4().hex[:8],
        "title": DEFAULT_TITLE,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    conversations = _load(path)
    conversations.append(conv)
    _save(conversations, path)
    return conv


def get_conversation(conversation_id: str, path: Path | None = None) -> dict | None:
    return _find(_load(path), conversation_id)


def delete_conversation(conversation_id: str, path: Path | None = None) -> bool:
    conversations = _load(path)
    remaining = [c for c in conversations if c["id"] != conversation_id]
    if len(remaining) == len(conversations):
        return False
    _save(remaining, path)
    return True


def rename_conversation(conversation_id: str, title: str, path: Path | None = None) -> dict | None:
    conversations = _load(path)
    conv = _find(conversations, conversation_id)
    if conv is None:
        return None
    conv["title"] = title.strip() or DEFAULT_TITLE
    conv["updated_at"] = _now()
    _save(conversations, path)
    return conv


def clear_messages(conversation_id: str, path: Path | None = None) -> dict | None:
    """清空一筆對話的訊息內容，但保留這個對話 id/在清單中的位置。"""
    conversations = _load(path)
    conv = _find(conversations, conversation_id)
    if conv is None:
        return None
    conv["messages"] = []
    conv["title"] = DEFAULT_TITLE
    conv["updated_at"] = _now()
    _save(conversations, path)
    return conv


def append_message(conversation_id: str, role: str, content: str, persona: str | None = None,
                    mode: str | None = None, path: Path | None = None) -> dict | None:
    """新增一則訊息。persona/mode 只有 GUI/CLI 呼叫時會帶（記錄當時的 /character
    人格、/model 模式），不帶就不寫進訊息物件——網頁端沒有這兩個概念，寫進去的
    訊息維持原本 {"role", "content", "ts"} 的精簡形狀。
    """
    conversations = _load(path)
    conv = _find(conversations, conversation_id)
    if conv is None:
        return None
    message = {"role": role, "content": content, "ts": _now()}
    if persona is not None:
        message["persona"] = persona
    if mode is not None:
        message["mode"] = mode
    conv["messages"].append(message)
    # 對話還沒有自訂標題時，拿第一則使用者訊息的開頭當標題。
    if role == "user" and conv["title"] == DEFAULT_TITLE:
        title = content.strip().splitlines()[0] if content.strip() else DEFAULT_TITLE
        if len(title) > _TITLE_MAX_LEN:
            title = title[:_TITLE_MAX_LEN] + "…"
        conv["title"] = title or DEFAULT_TITLE
    conv["updated_at"] = _now()
    _save(conversations, path)
    return conv
