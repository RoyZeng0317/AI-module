"""conversation_store.py — 對話紀錄的持久化儲存，跟桌面 GUI／終端機 CLI 共用
同一份實體檔案（memory/conversations.json，專案根目錄下，跟 memory/session.json、
memory/memory.json 同一層），不再只是 action.py 側邊欄自己的資料。

action.py 側邊欄的「對話紀錄」清單（對齊 default.html 的側邊欄樣式）需要能
真的新增/切換/刪除多筆對話，不能只是畫面上的假資料。這裡的存檔方式刻意
比照 lib/components/memory_store.py 的寫法（同一份 JSON 陣列檔、每筆有
"id"、_resolve_path() 讓測試可以指定暫存路徑而不動到真正的使用者資料）：
同一個專案內兩份持久化模組維持同一套慣例，之後看到其中一份就知道另一份
大概長怎樣。

跟 lib/components/conversation_store.py 幾乎是同一份程式碼，故意兩邊各自維護
一份、而不是互相 import 共用同一支模組：部署到 Render/樹莓派時（見
web/render.yaml、web/Dockerfile）容器映像檔只會複製 web/backend、web/frontend
兩個資料夾，不會連帶把整個 lib/（含 tkinter、sounddevice 等桌面版才需要的重
依賴）一起打包進去，兩邊沒辦法真的 import 同一支檔案。改成兩份程式碼各自算出
「專案根目錄下的 memory/ 資料夾」這個同一個實體路徑，本機開發時兩邊讀寫的就是
同一個 conversations.json，效果上等於共用同一份紀錄，讓桌面 GUI／CLI／網頁三端
不管從哪邊聊天，「對話紀錄」看到的都是同一份資料。改動任何一邊的 schema 要記得
同步改另一邊。

一筆對話存成：
{
    "id": "8 碼 hex",
    "title": str（預設 "新對話"，收到第一則使用者訊息後自動改成訊息開頭）,
    "created_at": ISO8601,
    "updated_at": ISO8601,
    "messages": [{"role": "user"/"assistant", "content": str, "ts": ISO8601,
                  "persona": str（選填，只有 GUI/CLI 會帶）,
                  "mode": str（選填，只有 GUI/CLI 會帶）}]
}
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if getattr(sys, "frozen", False):
    MEMORY_DIR = Path(sys.executable).resolve().parent / "memory"
else:
    # __file__ = web/backend/conversation_store.py -> parents[2] = 專案根目錄，
    # 跟 lib/components/session_store.py 算出來的是同一個 memory/ 資料夾。
    MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
CONVERSATIONS_PATH = MEMORY_DIR / "conversations.json"

DEFAULT_TITLE = "新對話"
_TITLE_MAX_LEN = 24


def _now() -> str:
    # 微秒精度，不是模仿 memory_store.py 的秒級精度——這裡的時間戳要拿來對
    # 側邊欄清單排序，同一秒內連續建立/更新多筆對話（例如測試、或使用者
    # 手速快）時，秒級精度會讓好幾筆時間戳完全相同，排序結果變得不穩定。
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _resolve_path(path: Path | None) -> Path:
    # path=None 時「即時」讀取模組全域的 CONVERSATIONS_PATH，理由跟
    # memory_store._resolve_path() 完全一樣：函式預設引數只在定義當下求值
    # 一次，測試 monkeypatch 模組全域變數不會反映到已經綁定的預設值上。
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
    """清空一筆對話的訊息內容，但保留這個對話 id/在側邊欄清單中的位置。"""
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
    人格、/model 模式），網頁端不帶，訊息物件維持原本 {"role", "content", "ts"}
    的精簡形狀。
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
    # 對話還沒有自訂標題時，拿第一則使用者訊息的開頭當標題——跟桌面版
    # 檔案總管幫沒命名的檔案挑預設名稱是同一個道理，不需要使用者手動輸入
    # 就能在側邊欄清單裡認出是哪一則對話。
    if role == "user" and conv["title"] == DEFAULT_TITLE:
        title = content.strip().splitlines()[0] if content.strip() else DEFAULT_TITLE
        if len(title) > _TITLE_MAX_LEN:
            title = title[:_TITLE_MAX_LEN] + "…"
        conv["title"] = title or DEFAULT_TITLE
    conv["updated_at"] = _now()
    _save(conversations, path)
    return conv
