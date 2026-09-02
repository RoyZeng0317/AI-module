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

這裡也順帶提供兩個從 entries 現算、不額外持久化的衍生功能：
`session_subject()` 幫 `/resume` 的還原內容產出一句主旨（純字詞頻率統計，
見函式內註解，不是呼叫 AI 摘要）——清掉 session.json 後 entries 變空，
下次聊出新內容再 `/resume` 就會照新內容重新產出，不用額外的旗標；
`export_markdown()`／`save_chat_export()` 給 `/chat` 指令把目前記錄的對話
下載成 Markdown 檔案用。
"""

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller onefile (nova.exe, built to live at the project root): see
    # lib/components/cli.py for why __file__ can't be used here when frozen.
    MEMORY_DIR = Path(sys.executable).resolve().parent / "memory"
    OUTPUT_DIR = Path(sys.executable).resolve().parent / "output" / "chats"
else:
    MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
    OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "chats"
SESSION_PATH = MEMORY_DIR / "session.json"

# 只保留最近這麼多輪——太舊的紀錄對「回到重開機前的對話」這個用途沒有幫助，
# 卻會讓檔案無限長大、/resume 重播畫面洗版洗不完。
MAX_SESSION_TURNS = 200

# /resume 主旨抽取用的字詞頻率統計——中文沒有裝 jieba 這類分詞套件，用
# 2-gram 滑動視窗近似分詞（中文常用詞多半兩個字，如「訓練」「模型」），
# 濾掉含有下列這些高頻虛詞/代名詞的詞組。跟 tools.py 的路由判斷同一種
# 「決定性演算法，不呼叫 AI」精神（CLAUDE.md Rule 06）——這不是模型生成
# 的摘要，只是字詞出現次數統計，沒有語意理解能力。
_ZH_STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "們", "這", "那", "也", "就",
    "都", "在", "和", "與", "及", "之", "而", "或", "嗎", "呢", "吧", "啊", "喔",
    "一", "個", "有", "沒", "要", "會", "能", "可", "以", "請", "給", "到", "從",
    "對", "於", "讓", "把", "被", "上", "下", "中", "為", "因", "所", "但", "不",
    "很", "太", "更", "最", "還", "又", "再", "已", "經", "現", "在", "什", "麼",
    "怎", "如", "何", "好", "謝", "麻", "煩", "幫",
}
_EN_STOPWORDS = {
    "the", "is", "a", "an", "to", "and", "of", "in", "for", "that", "this",
    "what", "how", "can", "you", "me", "my", "please", "it", "do", "does",
    "are", "on", "with", "be", "will", "would", "could", "should", "your",
    "not", "but", "just", "get", "have", "has", "had",
}
_HAN_RUN_RE = re.compile(r"[一-鿿]+")
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


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


def _keywords(entries: list[dict], top_n: int = 3) -> list[str]:
    """從對話內容（只看使用者打的那一半，回覆常常混雜太多套話式的固定用語）
    抓出幾個代表性詞組，依出現次數排序。"""
    counts: dict[str, int] = {}
    order: list[str] = []  # 次數相同時，保留第一次出現的先後順序
    for entry in entries:
        text = entry.get("user", "")
        for run in _HAN_RUN_RE.findall(text):
            for i in range(len(run) - 1):
                bigram = run[i:i + 2]
                if any(ch in _ZH_STOPWORDS for ch in bigram):
                    continue
                if bigram not in counts:
                    order.append(bigram)
                counts[bigram] = counts.get(bigram, 0) + 1
        for word in _EN_WORD_RE.findall(text):
            word = word.lower()
            if len(word) < 3 or word in _EN_STOPWORDS:
                continue
            if word not in counts:
                order.append(word)
            counts[word] = counts.get(word, 0) + 1

    ranked = sorted(order, key=lambda w: -counts[w])
    return ranked[:top_n]


def session_subject(entries: list[dict]) -> str:
    """幫這段對話產出一句主旨，給 `/resume`／`/chat` 顯示用。沒有紀錄（例如
    剛 `/resume clear` 過，或還沒聊過）回傳空字串——主旨不額外持久化，每次
    都是從目前的 entries 現算：清掉紀錄後 entries 自然變空，等下次真的聊出
    新內容再 `/resume`，就會照新內容重新產出，不需要另外寫「偵測到被清除
    就重算一次」的旗標邏輯。"""
    if not entries:
        return ""
    words = _keywords(entries)
    if words:
        return "、".join(words)
    # 抓不到夠格的詞組（例如整段都是很短的招呼閒聊）就退回第一句話當主旨
    first_user = entries[0]["user"].strip()
    return first_user[:20] + ("…" if len(first_user) > 20 else "")


def default_export_path(base_dir: Path | None = None) -> Path:
    """`/chat` 沒帶路徑時的預設存檔位置：output/chats/chat_<時間戳>.md，
    跟 memory/session.json 分開資料夾——那份是給程式自己讀寫的內部格式，
    這裡才是特地產生給人下載/保存的檔案。"""
    base = base_dir if base_dir is not None else OUTPUT_DIR
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"chat_{stamp}.md"


def export_markdown(entries: list[dict]) -> str:
    """把 session_store 記錄的對話格式化成 Markdown 逐字稿，給 `/chat`
    指令下載用。跟 `/resume` 重播用的是同一份 entries，確保「畫面上看到
    的對話」跟「存出來的檔案」內容一致，也帶上同一套 session_subject()
    主旨。"""
    lines = ["# 對話紀錄"]
    subject = session_subject(entries)
    if subject:
        lines.append(f"\n**主旨：** {subject}")
    if not entries:
        lines.append("\n（目前沒有記錄下的對話）")
        return "\n".join(lines) + "\n"

    lines.append(f"\n共 {len(entries)} 輪，最後更新於 {entries[-1]['created_at']}\n")
    for entry in entries:
        lines.append(f"### {entry['created_at']}")
        lines.append(f"**你：** {entry['user']}")
        lines.append("")
        lines.append(f"**{entry['persona']}：** {entry['reply']}")
        lines.append("")
    return "\n".join(lines)


def save_chat_export(entries: list[dict], path: Path | None = None) -> Path:
    """把 export_markdown() 的內容寫到磁碟，沒帶路徑就用 default_export_path()。
    回傳實際寫入的路徑，讓呼叫端可以回報給使用者。"""
    target = Path(path) if path is not None else default_export_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(export_markdown(entries), encoding="utf-8")
    return target
