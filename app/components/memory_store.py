"""memory_store.py — sinco 自己的持久記憶儲存，接給 CLAUDE.md 需求 #01
的 `/memory` 指令用。分類沿用 app/command/memorize.md 定義的優先順序（去掉
"Temporary Conversation" 那一類——memorize.md 自己的 Forget Policy 就講明
暫存內容不該被永久寫入，所以這裡不提供那個分類當選項）。

一條記憶存成 {"id", "category", "text", "created_at"}，全部記憶放在同一個
JSON 陣列檔案 memory.json 裡。這裡刻意不做語意搜尋/embedding——只是給
`/memory` 指令查看、新增、刪除用的結構化清單，跟 app/command/*.md（讀出來
整份塞給模型當提示詞）是不同機制：這些是使用者自己選分類寫入、可以查詢/
刪除的真實資料，不是餵給模型的固定提示詞。
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
MEMORY_PATH = MEMORY_DIR / "memory.json"

CATEGORIES = [
    "project", "architecture", "coding_style", "business_rule",
    "developer_preference", "task_progress",
]
CATEGORY_LABELS = {
    "project": "Project",
    "architecture": "Architecture",
    "coding_style": "Coding Style",
    "business_rule": "Business Rule",
    "developer_preference": "Developer Preference",
    "task_progress": "Task Progress",
}


def _resolve_path(path: Path | None) -> Path:
    # path=None 時「即時」讀取模組全域的 MEMORY_PATH，而不是把它綁死成函式
    # 預設引數的值——預設引數只在函式定義當下被求值一次，之後就算改了
    # memory_store.MEMORY_PATH（測試常見的 monkeypatch 手法），已經綁定
    # 的預設值也不會跟著變，會悄悄寫進真正的專案記憶檔。
    return path if path is not None else MEMORY_PATH


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


def list_memories(category: str | None = None, path: Path | None = None) -> list[dict]:
    entries = _load(path)
    if category:
        entries = [e for e in entries if e["category"] == category]
    return entries


def add_memory(category: str, text: str, path: Path | None = None) -> dict:
    category = category.strip().lower()
    if category not in CATEGORIES:
        raise ValueError(f"未知分類：{category}（可用：{', '.join(CATEGORIES)}）")
    text = text.strip()
    if not text:
        raise ValueError("記憶內容不可為空")

    entry = {
        "id": uuid.uuid4().hex[:8],
        "category": category,
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries = _load(path)
    entries.append(entry)
    _save(entries, path)
    return entry


def delete_memory(memory_id: str, path: Path | None = None) -> bool:
    entries = _load(path)
    remaining = [e for e in entries if e["id"] != memory_id]
    if len(remaining) == len(entries):
        return False
    _save(remaining, path)
    return True


def format_memories(entries: list[dict]) -> str:
    if not entries:
        return "（目前沒有記憶）"
    lines = []
    for e in entries:
        label = CATEGORY_LABELS.get(e["category"], e["category"])
        lines.append(f"[{e['id']}] ({label}) {e['text']}")
    return "\n".join(lines)
