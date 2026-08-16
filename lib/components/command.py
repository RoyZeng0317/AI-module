"""Slash-command palette: `/rules`, `/BugFix`, ... read from the *.md files
in app/command/ and fed to sinco as a prompt, plus the autocomplete UI
(suggestion popup, Tab-complete, Up/Down navigation) for typing them.

`/model`, `/memory`, `/init` (CLAUDE.md 需求 #01) are real Python-side
commands, not .md-prompt commands: they mutate actual state (conversation
mode / memory_store.py 的 JSON / 目標資料夾的 CLAUDE.md)，模型本身完全不
會看到它們的內容。`/character`（需求 #02）只負責開啟一個獨立視窗
（character_browser.py），角色瀏覽/切換邏輯都在那支檔案裡，這裡只是入口。
`/learn` 審核 tools.py 自動網路查詢存下的候選訓練資料（auto_learn.py）——
核准只會把它寫進 data/pairs.json，不會自動重訓，重訓永遠是你自己手動下
`python chats.py --data ...` 的動作。
"""

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import auto_learn
from lib.components.function import read_as_chat_content
from lib.components.memory_store import CATEGORIES as MEMORY_CATEGORIES
from lib.components.memory_store import add_memory, delete_memory, format_memories, list_memories
from lib.components.project_init import IGNORE_DIRS, write_claude_md
from lib.components.session_store import clear_session, load_session

# .md 指令定義檔跟這支程式檔不是同一個資料夾：這支在 app/components/，
# 指令檔在旁邊的 app/command/ 底下
COMMAND_DIR = Path(__file__).resolve().parent.parent / "command"
BUILTIN_COMMANDS = {"clear", "model", "memory", "init", "character", "learn", "resume"}
COMMANDS = sorted(BUILTIN_COMMANDS | {p.stem for p in COMMAND_DIR.glob("*.md")})

# @ 檔案附加：預設候選清單以整個專案資料夾為根，打相對路徑、即使打出 "../"
# 想跳出去也會被夾住當成沒有符合的候選（CLAUDE.md 規則 01：不可「修改」
# 專案資料夾以外的內容）。app/components/ -> app/ -> 專案根目錄，往上兩層。
#
# 但規則 01 管的是「修改」，@ 附加是純讀取（跟既有 /open、輸入框「+」開檔
# 按鈕能讀任意磁碟路徑是同一件事），所以額外開放「磁碟絕對路徑」
# （"C:\..."、"C:/..."、UNC "\\主機\..."）或 "~/" 開頭這兩種寫法可以瀏覽/
# 附加專案資料夾以外的檔案——不是放寬「相對路徑」的限制，"../" 這種仍然擋
# 下來（見 _is_external_path()／resolve_mention()）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_SUGGESTIONS = 12
PREVIEW_MAX_LINES = 30
PREVIEW_MAX_CHARS = 2000
MENTION_PATTERN = re.compile(r"@(\S+)")
_ABS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|~[\\/])")


def _is_external_path(token: str) -> bool:
    """token 是不是「磁碟絕對路徑」寫法（跟專案內相對路徑分開處理，見上方
    PROJECT_ROOT 的說明）。"""
    return bool(_ABS_PATH_RE.match(token))


# /model 手動切換 chats.smart_reply_traced() 的 force_mode（見 conversation.py
# 的 Conversation.force_mode）。放在這裡而不是 chats.py，因為這是「使用者要
# 選哪個選項」的 UI 層清單，跟模型本身的訓練/推論邏輯無關。
MODEL_MODES = ["auto", "sinco", "code", "nvidia"]
MODEL_LABELS = {
    "auto": "自動判斷（預設，程式碼問句自動轉去 code 模型）",
    "sinco": "一般聊天模型（強制，即使問句看起來像程式碼）",
    "code": "程式碼模型（強制，即使問句看起來不像程式碼）",
    "nvidia": "NVIDIA 雲端模型（外部 API，非本專案自訓練，需自行設定環境變數 NVIDIA_API_KEY）",
}

# 輸入「/指令 」（帶一個空格）之後，還能針對特定指令跳出「引數」選項框
# （需求 #01：「選項框顯示，一樣可以 tab 快速鍵入」），用跟指令名稱建議
# 完全相同的 Listbox/Tab/上下鍵機制，不是另外做一套 UI。
ARG_SUGGESTIONS = {"model": MODEL_MODES, "resume": ["clear"]}

# /resume 還原歷史時，補回 self.conversation.history 要跟 conversation.py 的
# MAX_HISTORY_TURNS 保持同一個上限——這裡不直接 import conversation.py 是
# 因為那支檔案會連帶拉進 sounddevice 等音訊套件，屬於 GUI 執行期才需要的重
# 依賴，不該讓純指令邏輯的 command.py 也被迫背上（跟 _run_character() 把
# character_browser 的 import 延後到函式內部是同一個考量）。
RESUME_HISTORY_CAP = 20


def list_files(directory: Path) -> list[Path]:
    """資料夾內容：資料夾排在檔案前面，各自再依檔名做 A-Z/0-9 排序。"""
    entries = list(directory.iterdir())
    folders = sorted((p for p in entries if p.is_dir()), key=lambda p: p.name.lower())
    files = sorted((p for p in entries if p.is_file()), key=lambda p: p.name.lower())
    return folders + files


def mention_matches(token: str) -> list[str]:
    """@token 目前打到的候選清單。token 用最後一個 "/" 拆成「要瀏覽的資料夾」
    跟「檔名前綴」，資料夾內容交給 list_files()（資料夾排前面）列出、依前綴
    篩選——同一套邏輯讓打 "@app/comp" 這種帶子資料夾的路徑也能繼續往下篩，
    不用另外做「點資料夾進去瀏覽」的狀態機。資料夾候選字串結尾補 "/"，讓
    choose_suggestion() 能單純看字串結尾判斷選到的是資料夾還是檔案。

    token 是磁碟絕對路徑（_is_external_path()）時改成瀏覽該路徑所在資料夾，
    不受 PROJECT_ROOT 限制；否則沿用「限制在專案資料夾內」的原本邏輯（"../"
    仍然會被擋下）。normalize 成 "/" 是因為 Windows 路徑習慣用 "\\"，但
    rpartition 只認一種分隔符，兩種輸入都先轉成 "/" 就能共用同一套拆解。
    """
    norm = token.replace("\\", "/")
    dir_part, _, name_query = norm.rpartition("/")

    if _is_external_path(token):
        if not dir_part:
            return []  # 例如只打了 "C:" 或 "~"，還沒打出可以瀏覽的完整資料夾路徑
        browse_dir = Path(dir_part).expanduser()
    else:
        browse_dir = (PROJECT_ROOT / dir_part).resolve() if dir_part else PROJECT_ROOT
        try:
            browse_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            return []  # 想用相對路徑跳出專案資料夾（例如 "../"），視為沒有符合的候選

    if not browse_dir.is_dir():
        return []

    try:
        entries = list_files(browse_dir)
    except OSError:
        return []

    query = name_query.lower()
    prefix = f"{dir_part}/" if dir_part else ""
    matches = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in IGNORE_DIRS:
            continue
        if not entry.name.lower().startswith(query):
            continue
        matches.append(f"{prefix}{entry.name}/" if entry.is_dir() else f"{prefix}{entry.name}")
    return matches[:MAX_FILE_SUGGESTIONS]


def resolve_mention(token: str) -> Path | None:
    """把一個 @提及 字串（使用者自己打的，或從 mention_matches() 候選清單選的
    都一樣）換算成絕對路徑；不是檔案（資料夾、不存在、或用相對路徑想跳出
    專案資料夾）一律回傳 None。file() 用這個決定要不要附加、_update_preview()
    也用同一個函式算出要預覽哪個檔案，兩邊邏輯保持一致。
    """
    if _is_external_path(token):
        candidate = Path(token.replace("\\", "/")).expanduser()
    else:
        candidate = (PROJECT_ROOT / token).resolve()
        try:
            candidate.relative_to(PROJECT_ROOT)
        except ValueError:
            return None
    return candidate if candidate.is_file() else None


def _read_preview(path: Path) -> str:
    """給預覽面板用的簡化讀檔：只讀前 PREVIEW_MAX_CHARS 個字元、只顯示前
    PREVIEW_MAX_LINES 行，避免選到大檔案（例如整段 log）時卡住 UI——跟
    read_as_chat_content() 用途不同，後者是要把「完整內容」送給模型，這裡只
    是給人看的縮圖，不需要也不該讀整份。
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(PREVIEW_MAX_CHARS)
    except OSError:
        return "(無法讀取此檔案)"
    lines = content.splitlines()[:PREVIEW_MAX_LINES]
    return "\n".join(lines) or "(空檔案)"


class CommandPalette:
    def __init__(self, windows: tk.Tk, message: tk.Entry, chat_display: tk.Text, conversation):
        self.windows = windows
        self.message = message
        self.chat_display = chat_display
        self.conversation = conversation
        self.suggestion_win = None
        self.suggestion_listbox = None
        self.current_matches = []
        self.selected_index = -1
        self._suggesting_arg_for: str | None = None  # update_suggestions() 目前是在建議指令名稱、還是某指令的引數
        self._file_mention_span: tuple[int, int] | None = None  # 目前 "@提及" 在 Entry 裡的 (起, 迄) 索引，選好候選後要整段替換掉
        self._match_paths: list[Path | None] | None = None  # current_matches 對應的絕對路徑（給預覽面板用），非 @提及 建議時是 None
        self.preview_text: tk.Text | None = None  # @提及 建議清單旁邊的檔案內容預覽面板（像 Claude Code 打 @ 時那樣）

    # 系統訊息：跟模型回覆無關的本地提示（未知指令、/model /memory /init 的
    # 執行結果），統一走這個 helper，不占用 conversation.ask() 那套「思考中」
    # + 背景執行緒的流程（這些指令是立即、同步、純本機邏輯，不需要那些）。
    def _system_message(self, text: str):
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"{text}\n\n", "system")
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    # 指令：clear/model/memory/init/character 是內建動作；其餘指令對應
    # app/command/ 資料夾同名的 .md 檔，讀出內容交給模型當提示詞
    def run(self, cmd: str):
        name, _, arg = cmd.partition(" ")
        name, arg = name.strip(), arg.strip()

        if name == "clear":
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.configure(state="disabled")
            return
        if name == "model":
            self._run_model(arg)
            return
        if name == "memory":
            self._run_memory(arg)
            return
        if name == "init":
            self._run_init(arg)
            return
        if name == "character":
            self._run_character()
            return
        if name == "learn":
            self._run_learn(arg)
            return
        if name == "resume":
            self._run_resume(arg)
            return

        path = COMMAND_DIR / f"{name}.md"
        if not path.exists():
            self._system_message(f"未知指令：/{name}")
            return

        content = read_as_chat_content(path)
        self.conversation.ask(f"--- /{name} ---", content, header_tag="system")

    # 輸入符號 @ 即可預覽檔案選擇進行訊息附檔，用戶輸入檔名後可以選擇對應的
    # 檔案與 AI 對話，可以減少 AI 花時間與 token 去尋找。另外檔案並未限制
    # 可以掛載用量，因此可以多個檔案進行詢問（見 update_suggestions() 裡
    # @ 開頭時的自動完成，跟這裡送出訊息時的展開，是同一個功能的兩端）。
    # 打相對路徑一律以專案資料夾為根；打磁碟絕對路徑（"C:\..."）或 "~/" 開頭
    # 則可以附加專案資料夾以外的檔案（resolve_mention()），用途是讓 /model
    # nvidia 這種外部模型也能讀到你從檔案總管複製出來的路徑——這兩種指令
    # （這裡的 @ 跟 cli.py 的 /open）現在行為一致了。
    # 由 main.py 的 on_submit() 在非「/」開頭的訊息呼叫，取代直接呼叫
    # conversation.send_message()：沒有 @提及 時行為完全相同，原樣轉呼叫過去。
    def file(self, text: str):
        mentions = list(dict.fromkeys(MENTION_PATTERN.findall(text)))  # 去重、保留第一次出現的順序
        if not mentions:
            self.conversation.send_message(text)
            return

        attachments = []
        for mention in mentions:
            candidate = resolve_mention(mention)
            if candidate is None:
                self._system_message(f"未知檔案：@{mention}")
                return
            attachments.append(f"--- @{mention} ---\n{read_as_chat_content(candidate)}")

        model_input = "\n\n".join(attachments) + f"\n\n{text}"
        self.conversation.ask(text, model_input, record_as=text)

    # /model <auto|sinco|code>：手動覆蓋 chats.smart_reply_traced() 的自動
    # chat/code 判斷。沒帶引數就顯示目前模式＋可用選項（同時也是 /model 後面
    # 打空白時，選項框裡看到的那份清單）。
    def _run_model(self, arg: str):
        if not arg:
            current = self.conversation.force_mode
            options = "\n".join(f"  {m} — {MODEL_LABELS[m]}" for m in MODEL_MODES)
            self._system_message(f"目前模式：{current}（{MODEL_LABELS.get(current, current)}）\n{options}")
            return
        mode = arg.lower()
        if mode not in MODEL_MODES:
            self._system_message(f"未知模式：{arg}（可用：{'、'.join(MODEL_MODES)}）")
            return
        self.conversation.force_mode = mode
        self._system_message(f"已切換模式：{mode}（{MODEL_LABELS[mode]}）")

    # /memory [list [分類] | add <分類> <內容> | del <id>]：真正持久化的記憶
    # 儲存（memory_store.py），分類沿用 app/command/memorize.md 的分類。
    def _run_memory(self, arg: str):
        sub, _, rest = arg.partition(" ")
        sub, rest = sub.strip().lower(), rest.strip()

        if sub in ("", "list"):
            category = rest.lower() or None
            if category and category not in MEMORY_CATEGORIES:
                self._system_message(f"未知分類：{category}（可用：{', '.join(MEMORY_CATEGORIES)}）")
                return
            self._system_message(format_memories(list_memories(category)))
            return

        if sub == "add":
            category, _, text = rest.partition(" ")
            try:
                entry = add_memory(category, text)
            except ValueError as exc:
                self._system_message(str(exc))
                return
            self._system_message(f"已新增記憶 [{entry['id']}]：{entry['text']}")
            return

        if sub in ("del", "delete"):
            if delete_memory(rest):
                self._system_message(f"已刪除記憶 [{rest}]")
            else:
                self._system_message(f"找不到記憶 id：{rest}")
            return

        self._system_message(
            "用法：\n"
            "  /memory                    列出全部記憶\n"
            "  /memory list <分類>        列出指定分類\n"
            f"  /memory add <分類> <內容>  新增（分類：{', '.join(MEMORY_CATEGORIES)}）\n"
            "  /memory del <id>           刪除"
        )

    # /init [資料夾路徑]：沒帶路徑就跳資料夾選擇視窗，掃描結果寫進該資料夾的
    # CLAUDE.md（見 project_init.py，只更新 sinco:init 標記區塊）。
    def _run_init(self, arg: str):
        target = arg
        if not target:
            chosen = filedialog.askdirectory(title="選擇要 /init 的專案資料夾")
            if not chosen:
                return
            target = chosen

        root = Path(target).expanduser()
        if not root.is_dir():
            self._system_message(f"找不到資料夾：{root}")
            return
        path = write_claude_md(root)
        self._system_message(f"已產生/更新：{path}")

    # /character：開啟獨立的角色卡瀏覽器視窗（需求 #02：不是 home_screen.py
    # 這個 GUI 本身），選好角色後回呼切換 self.conversation 的 persona。
    def _run_character(self):
        from lib.components.character_browser import open_character_browser
        open_character_browser(self.windows, self.conversation)

    # /learn [list | approve <id> | reject <id>]：審核 tools.py 自動網路查詢
    # 存下的候選訓練資料（auto_learn.py）。approve 只會把它寫進
    # data/pairs.json，不會自動重訓——重訓永遠是你自己另外手動下
    # `python chats.py --data ...` 的動作，避免在你沒注意到的情況下，用
    # 品質不明的網路摘要悄悄動到已經收斂好的 checkpoint。
    def _run_learn(self, arg: str):
        sub, _, rest = arg.partition(" ")
        sub, rest = sub.strip().lower(), rest.strip()

        if sub in ("", "list"):
            self._system_message(auto_learn.format_candidates(auto_learn.list_candidates()))
            return

        if sub == "approve":
            if auto_learn.approve_candidate(rest):
                self._system_message(f"已核准並寫入 data/pairs.json：[{rest}]（尚未重訓，記得之後手動執行 chats.py）")
            else:
                self._system_message(f"找不到候選 id：{rest}")
            return

        if sub == "reject":
            if auto_learn.reject_candidate(rest):
                self._system_message(f"已捨棄候選：[{rest}]")
            else:
                self._system_message(f"找不到候選 id：{rest}")
            return

        self._system_message(
            "用法：\n"
            "  /learn                 列出目前待審核的候選學習內容\n"
            "  /learn approve <id>    核准，寫入 data/pairs.json（不會自動重訓）\n"
            "  /learn reject <id>     捨棄"
        )

    # /resume [clear]：還原電腦重開機/意外關機前記錄下的對話——每一輪對話在
    # Conversation.ask() 拿到回覆的當下就已經用 session_store.record_turn()
    # 落地到 memory/session.json（不是等程式正常關閉才存），所以就算不是正常
    # 關掉這個視窗，重開機後 /resume 仍讀得到最後聊到哪。重播內容只是唸給你
    # 看的系統訊息，不會重新送進模型；同時把 self.conversation.history 補回
    # 去，維持跟正常聊天時同一種「累積歷史」狀態。跟 CLI（cli.py 的
    # run_resume()）同一套 session_store.py、同一份檔案，兩邊互通。
    def _run_resume(self, arg: str):
        sub = arg.strip().lower()
        if sub in ("clear", "reset"):
            clear_session()
            self._system_message("已清除記錄下的對話（/resume 之後不會再看到目前這些內容）")
            return

        entries = load_session()
        if not entries:
            self._system_message("目前沒有記錄下的對話（還沒聊過，或紀錄已被清除）")
            return

        lines = [f"— 還原對話紀錄（共 {len(entries)} 輪，最後更新於 {entries[-1]['created_at']}）—"]
        for entry in entries:
            lines.append(f"你：{entry['user']}")
            lines.append(f"{entry['persona']}：{entry['reply']}")
        self._system_message("\n".join(lines))

        history = [(e["user"], e["reply"]) for e in entries]
        self.conversation.history = history[-RESUME_HISTORY_CAP:]

    # 隱藏建議
    def hide_suggestions(self):
        if self.suggestion_win is not None:
            self.suggestion_win.destroy()
            self.suggestion_win = None
            self.suggestion_listbox = None
        self.selected_index = -1
        self._file_mention_span = None
        self._match_paths = None
        self.preview_text = None

    # 選擇建議：可能是在補指令名稱（/mo -> /model）、補指令的引數
    # （/model  -> /model code），也可能是在補 @提及 的路徑——由
    # _file_mention_span/_suggesting_arg_for 判斷是哪一種。@提及 選到資料夾
    # 就只補到那層路徑（結尾留 "/"，不加空白），讓使用者可以繼續往下打/篩選
    # 子資料夾；選到檔案才補一個空白，代表這個 @提及 已經完整、可以接著打
    # 別的字或再打一個 @ 附加下一個檔案。
    def choose_suggestion(self, value: str):
        if self._file_mention_span is not None:
            start, end = self._file_mention_span
            insertion = f"@{value}" if value.endswith("/") else f"@{value} "
            self.message.delete(start, end)
            self.message.insert(start, insertion)
            self.hide_suggestions()
            self.message.icursor(start + len(insertion))
            self.message.focus_set()
            return

        self.message.delete(0, tk.END)
        if self._suggesting_arg_for is not None:
            self.message.insert(0, f"/{self._suggesting_arg_for} {value}")
        else:
            self.message.insert(0, f"/{value}")
        self.hide_suggestions()
        self.message.focus_set()

    # 目前游標左邊，正在打的「@提及」是什麼：從游標往回找最近一個 "@"，中間
    # 只要出現空白就代表那個 "@" 已經跟目前這段文字無關（是先前打完的提及，
    # 或單純訊息裡的一個符號），回傳 None。跟 "/" 指令名稱建議不同的地方是
    # @提及 可以出現在句子中間，不限定要在整個訊息的開頭。
    def _current_mention(self) -> tuple[int, int, str] | None:
        cursor = self.message.index(tk.INSERT)
        prefix = self.message.get()[:cursor]
        at = prefix.rfind("@")
        if at == -1:
            return None
        token = prefix[at + 1:]
        if any(ch.isspace() for ch in token):
            return None
        return at, cursor, token

    # 顯示建議。match_paths 只有 @提及 建議會帶（跟 matches 一一對應，資料夾
    # /找不到是 None）——帶了才會多開一塊預覽面板，仿照 Claude Code 打 @ 時
    # 會即時顯示候選檔案內容的體驗；/指令 或引數建議不帶這個參數，維持原本
    # 只有清單的樣子。
    def show_suggestions(self, matches: list[str], match_paths: list[Path | None] | None = None):
        self.hide_suggestions()
        self.suggestion_win = tk.Toplevel(self.windows)
        self.suggestion_win.overrideredirect(True)
        row_height = 20
        win_height = row_height * len(matches) + 4
        x = self.message.winfo_rootx()
        y = self.message.winfo_rooty() - win_height - 4
        self.suggestion_win.geometry(f"+{x}+{y}")

        container = tk.Frame(self.suggestion_win)
        container.pack()

        # 綁定時抓區域變數 listbox（一定是實例，不是 Optional），不要在 lambda
        # 裡引用 self.suggestion_listbox，避免型別檢查器誤判它可能是 None
        listbox = tk.Listbox(container, height=len(matches), width=36, bd=1, relief="solid", exportselection=False)
        listbox.pack(side="left", fill="y")
        for cmd in matches:
            listbox.insert(tk.END, cmd)
        listbox.bind("<<ListboxSelect>>", lambda e: self.choose_suggestion(matches[listbox.curselection()[0]]))
        self.suggestion_listbox = listbox

        self._match_paths = match_paths
        if match_paths is not None:
            preview = tk.Text(
                container, width=64, height=len(matches), bd=1, relief="solid",
                wrap="none", state="disabled", bg="#1e1e1e", fg="#d4d4d4",
            )
            preview.pack(side="left", fill="both")
            self.preview_text = preview
            self._update_preview(0)  # 一開始沒按過上下鍵，先預覽第一個候選（Tab 也是預設補這個）

    # 更新指令建議：輸入「/」開頭時建議指令名稱；輸入「/指令 」（帶空格）之後，
    # 如果那個指令在 ARG_SUGGESTIONS 有登記引數選項（目前只有 /model），改成
    # 建議引數值——同一套 Listbox/Tab/上下鍵機制，不是另外做一個選項框元件。
    def update_suggestions(self, event=None):
        # 上下鍵／Tab／Enter／Esc 是在操作已經顯示出來的建議清單，不該讓清單重新整個重建
        if event is not None and event.keysym in ("Up", "Down", "Tab", "Return", "Escape"):
            return

        # @提及 優先判斷：跟「/」指令不同，@ 可以出現在一般訊息句子中間，
        # 不限定要在整個輸入框的開頭，所以不能只看 text.startswith(...)。
        mention = self._current_mention()
        if mention is not None:
            start, end, token = mention
            self._suggesting_arg_for = None
            self.current_matches = mention_matches(token)
            self.selected_index = -1
            if self.current_matches:
                # show_suggestions() 內部一開始會呼叫 hide_suggestions()（清掉
                # 上一次的建議視窗），而 hide_suggestions() 也會把
                # _file_mention_span 重置成 None——所以一定要等 show_suggestions()
                # 呼叫完才能設定 _file_mention_span，不然會被馬上清掉。
                match_paths = [resolve_mention(m) for m in self.current_matches]
                self.show_suggestions(self.current_matches, match_paths)
                self._file_mention_span = (start, end)
            else:
                self.hide_suggestions()  # 沒有符合的候選：沒東西可選，_file_mention_span 清掉也無妨
            return
        self._file_mention_span = None

        text = self.message.get()
        if not text.startswith("/"):
            self._suggesting_arg_for = None
            self.current_matches = []
            self.hide_suggestions()
            return

        if " " in text:
            name, _, partial_arg = text[1:].partition(" ")
            options = ARG_SUGGESTIONS.get(name.strip().lower())
            if options is None:
                self._suggesting_arg_for = None
                self.current_matches = []
                self.hide_suggestions()
                return
            self._suggesting_arg_for = name.strip().lower()
            query = partial_arg.strip().lower()
            self.current_matches = [o for o in options if o.lower().startswith(query)]
        else:
            self._suggesting_arg_for = None
            query = text[1:].lower()
            self.current_matches = [c for c in COMMANDS if c.lower().startswith(query)]

        self.selected_index = -1
        if self.current_matches:
            self.show_suggestions(self.current_matches)
        else:
            self.hide_suggestions()

    # @提及 預覽面板：把目前反白的候選換算成絕對路徑、讀出內容摘要顯示出來，
    # 不是檔案（資料夾、找不到）就顯示提示文字而不是報錯——這只是預覽，選錯
    # 也不影響 file() 送出時的正式判斷（resolve_mention() 是同一套邏輯）。
    def _update_preview(self, idx: int):
        if self.preview_text is None or self._match_paths is None:
            return
        if idx < 0 or idx >= len(self._match_paths):
            return
        path = self._match_paths[idx]
        text = "(資料夾，無法預覽)" if path is None else _read_preview(path)
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    # 上下鍵：在目前顯示的建議清單裡移動反白選項
    def move_selection(self, delta: int):
        if not self.current_matches or self.suggestion_listbox is None:
            return "break"
        self.selected_index = (self.selected_index + delta) % len(self.current_matches)
        self.suggestion_listbox.selection_clear(0, tk.END)
        self.suggestion_listbox.selection_set(self.selected_index)
        self.suggestion_listbox.activate(self.selected_index)
        self.suggestion_listbox.see(self.selected_index)
        self._update_preview(self.selected_index)
        return "break"

    def on_arrow_up(self, event):
        return self.move_selection(-1)

    def on_arrow_down(self, event):
        return self.move_selection(1)

    # Tab 快速鍵：補成目前反白的指令（沒有用上下鍵選過就用第一個），不用打完整指令名稱
    def complete(self, event=None):
        if not self.current_matches:
            return None
        idx = self.selected_index if self.selected_index >= 0 else 0
        self.choose_suggestion(self.current_matches[idx])
        return "break"  # 阻止 Tab 預設把焦點跳到下一個元件
