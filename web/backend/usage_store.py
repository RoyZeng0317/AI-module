"""usage_store.py — 字元用量的持久化紀錄與上限判斷。key 由 app.py 的
_resolve_client_id() 決定：有 Google 登入就用帳號識別（"google:<uid>"），
沒有就退回來源 IP，跟 app.py 既有的 _check_chat_rate_limit() 是同一套身分
判斷邏輯，但兩者擋的事情不同：那邊擋「短時間內連續轟炸」（60 秒內幾次請求），
這裡擋「一段時間內累計用掉太多字」（24 小時內幾個字）。

寫法比照 conversation_store.py / notify_store.py：同一份 JSON 檔、
_resolve_path() 讓測試可以指定暫存路徑而不動到真正的使用者資料。

存檔格式：{ip: [{"ts": epoch_seconds, "chars": int}, ...]}——每個 IP 一份
時間序列，讀取時只加總還在時間窗內的筆數；寫入時順便把窗外的舊紀錄丟掉，
檔案不會無限長大。

sinco 是字元級模型（見 CLAUDE.md），這裡算的「用量」就是字元數（使用者
訊息 + 模型回覆的字數相加），不是 subword/BPE token——這個專案沒有那種
分詞器，字元數是最貼近實際運算量的量測方式。
"""

import json
import time
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
USAGE_PATH = MEMORY_DIR / "usage.json"

USAGE_WINDOW_SEC = 24 * 60 * 60
USAGE_CHAR_LIMIT = 20000


def _resolve_path(path: Path | None) -> Path:
    return path if path is not None else USAGE_PATH


def _load(path: Path | None = None) -> dict:
    path = _resolve_path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(data: dict, path: Path | None = None) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prune(entries: list[dict], now: float) -> list[dict]:
    return [e for e in entries if now - e["ts"] < USAGE_WINDOW_SEC]


def get_usage(client_id: str, path: Path | None = None) -> int:
    """回傳 client_id 在時間窗內已累計的字元數。"""
    now = time.time()
    entries = _prune(_load(path).get(client_id, []), now)
    return sum(e["chars"] for e in entries)


def add_usage(client_id: str, chars: int, path: Path | None = None) -> int:
    """記錄一次用量，回傳記錄後的累計字元數。"""
    now = time.time()
    data = _load(path)
    entries = _prune(data.get(client_id, []), now)
    entries.append({"ts": now, "chars": chars})
    data[client_id] = entries
    _save(data, path)
    return sum(e["chars"] for e in entries)
