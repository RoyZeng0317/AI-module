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

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import auto_learn
from function import read_as_chat_content
from memory_store import CATEGORIES as MEMORY_CATEGORIES
from memory_store import add_memory, delete_memory, format_memories, list_memories
from project_init import write_claude_md

# .md 指令定義檔跟這支程式檔不是同一個資料夾：這支在 app/components/，
# 指令檔在旁邊的 app/command/ 底下
COMMAND_DIR = Path(__file__).resolve().parent.parent / "command"
BUILTIN_COMMANDS = {"clear", "model", "memory", "init", "character", "learn"}
COMMANDS = sorted(BUILTIN_COMMANDS | {p.stem for p in COMMAND_DIR.glob("*.md")})

# /model 手動切換 chats.smart_reply_traced() 的 force_mode（見 conversation.py
# 的 Conversation.force_mode）。放在這裡而不是 chats.py，因為這是「使用者要
# 選哪個選項」的 UI 層清單，跟模型本身的訓練/推論邏輯無關。
MODEL_MODES = ["auto", "sinco", "code"]
MODEL_LABELS = {
    "auto": "自動判斷（預設，程式碼問句自動轉去 code 模型）",
    "sinco": "一般聊天模型（強制，即使問句看起來像程式碼）",
    "code": "程式碼模型（強制，即使問句看起來不像程式碼）",
}

# 輸入「/指令 」（帶一個空格）之後，還能針對特定指令跳出「引數」選項框
# （需求 #01：「選項框顯示，一樣可以 tab 快速鍵入」），用跟指令名稱建議
# 完全相同的 Listbox/Tab/上下鍵機制，不是另外做一套 UI。
ARG_SUGGESTIONS = {"model": MODEL_MODES}


def list_files(directory: Path) -> list[Path]:
    """資料夾內容：資料夾排在檔案前面，各自再依檔名做 A-Z/0-9 排序。"""
    entries = list(directory.iterdir())
    folders = sorted((p for p in entries if p.is_dir()), key=lambda p: p.name.lower())
    files = sorted((p for p in entries if p.is_file()), key=lambda p: p.name.lower())
    return folders + files


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

        path = COMMAND_DIR / f"{name}.md"
        if not path.exists():
            self._system_message(f"未知指令：/{name}")
            return

        content = read_as_chat_content(path)
        self.conversation.ask(f"--- /{name} ---", content, header_tag="system")

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
        from character_browser import open_character_browser
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

    # 隱藏建議
    def hide_suggestions(self):
        if self.suggestion_win is not None:
            self.suggestion_win.destroy()
            self.suggestion_win = None
            self.suggestion_listbox = None
        self.selected_index = -1

    # 選擇建議：可能是在補指令名稱（/mo -> /model），也可能是在補指令的引數
    # （/model  -> /model code），由 _suggesting_arg_for 判斷是哪一種
    def choose_suggestion(self, value: str):
        self.message.delete(0, tk.END)
        if self._suggesting_arg_for is not None:
            self.message.insert(0, f"/{self._suggesting_arg_for} {value}")
        else:
            self.message.insert(0, f"/{value}")
        self.hide_suggestions()
        self.message.focus_set()

    # 顯示建議
    def show_suggestions(self, matches: list[str]):
        self.hide_suggestions()
        self.suggestion_win = tk.Toplevel(self.windows)
        self.suggestion_win.overrideredirect(True)
        row_height = 20
        x = self.message.winfo_rootx()
        y = self.message.winfo_rooty() - (row_height * len(matches)) - 4
        self.suggestion_win.geometry(f"+{x}+{y}")
        # 綁定時抓區域變數 listbox（一定是實例，不是 Optional），不要在 lambda
        # 裡引用 self.suggestion_listbox，避免型別檢查器誤判它可能是 None
        listbox = tk.Listbox(self.suggestion_win, height=len(matches), bd=1, relief="solid", exportselection=False)
        listbox.pack()
        for cmd in matches:
            listbox.insert(tk.END, cmd)
        listbox.bind("<<ListboxSelect>>", lambda e: self.choose_suggestion(matches[listbox.curselection()[0]]))
        self.suggestion_listbox = listbox

    # 更新指令建議：輸入「/」開頭時建議指令名稱；輸入「/指令 」（帶空格）之後，
    # 如果那個指令在 ARG_SUGGESTIONS 有登記引數選項（目前只有 /model），改成
    # 建議引數值——同一套 Listbox/Tab/上下鍵機制，不是另外做一個選項框元件。
    def update_suggestions(self, event=None):
        # 上下鍵／Tab／Enter／Esc 是在操作已經顯示出來的建議清單，不該讓清單重新整個重建
        if event is not None and event.keysym in ("Up", "Down", "Tab", "Return", "Escape"):
            return
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

    # 上下鍵：在目前顯示的建議清單裡移動反白選項
    def move_selection(self, delta: int):
        if not self.current_matches or self.suggestion_listbox is None:
            return "break"
        self.selected_index = (self.selected_index + delta) % len(self.current_matches)
        self.suggestion_listbox.selection_clear(0, tk.END)
        self.suggestion_listbox.selection_set(self.selected_index)
        self.suggestion_listbox.activate(self.selected_index)
        self.suggestion_listbox.see(self.selected_index)
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
