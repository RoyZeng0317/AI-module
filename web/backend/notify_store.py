"""notify_store.py — 通知的持久化儲存。admin/frontend/src/components/action.py
的「發送通知」彈窗呼叫 create_notification() 寫入一筆，web/frontend/src/
components/python.py 的通知面板輪詢 list_notifications() 讀出來顯示。寫法
比照 conversation_store.py：同一份 JSON 陣列檔、_resolve_path() 讓測試可以
指定暫存路徑而不動到真正的使用者資料。

一筆通知存成：
{
    "id": "8 碼 hex",
    "message": str,
    "created_at": ISO8601,
    "read": bool
}
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# __file__ = web/backend/notify_store.py -> parents[2] = 專案根目錄，跟
# conversation_store.py 算出來的是同一個 memory/ 資料夾。
MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
NOTIFICATIONS_PATH = MEMORY_DIR / "notifications.json"


def _resolve_path(path: Path | None) -> Path:
    return path if path is not None else NOTIFICATIONS_PATH


def _load(path: Path | None = None) -> list[dict]:
    path = _resolve_path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(notifications: list[dict], path: Path | None = None) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notifications, ensure_ascii=False, indent=2), encoding="utf-8")


def list_notifications(path: Path | None = None) -> list[dict]:
    """依建立時間新到舊排序回傳全部通知。"""
    notifications = _load(path)
    notifications.sort(key=lambda n: n["created_at"], reverse=True)
    return notifications


def create_notification(message: str, path: Path | None = None) -> dict:
    notification = {
        "id": uuid.uuid4().hex[:8],
        "message": message,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "read": False,
    }
    notifications = _load(path)
    notifications.append(notification)
    _save(notifications, path)
    return notification


def mark_all_read(path: Path | None = None) -> list[dict]:
    notifications = _load(path)
    for n in notifications:
        n["read"] = True
    _save(notifications, path)
    return list_notifications(path)
