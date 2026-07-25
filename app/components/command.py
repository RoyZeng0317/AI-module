"""Slash-command palette: `/rules`, `/BugFix`, ... read from the *.md files
in app/command/ and fed to sinco as a prompt, plus the autocomplete UI
(suggestion popup, Tab-complete, Up/Down navigation) for typing them.
"""

import tkinter as tk
from pathlib import Path

from function import read_as_chat_content

# .md 指令定義檔跟這支程式檔不是同一個資料夾：這支在 app/components/，
# 指令檔在旁邊的 app/command/ 底下
COMMAND_DIR = Path(__file__).resolve().parent.parent / "command"
BUILTIN_COMMANDS = {"clear"}
COMMANDS = sorted(BUILTIN_COMMANDS | {p.stem for p in COMMAND_DIR.glob("*.md")})


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

    # 指令：clear 是內建動作；其餘指令對應 app/command/ 資料夾同名的 .md 檔，
    # 讀出內容交給模型當提示詞，不再用 if/elif 逐一寫死每個指令的行為
    def run(self, cmd: str):
        if cmd == "clear":
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.configure(state="disabled")
            return

        path = COMMAND_DIR / f"{cmd}.md"
        if not path.exists():
            self.chat_display.configure(state="normal")
            self.chat_display.insert(tk.END, f"未知指令：/{cmd}\n\n")
            self.chat_display.configure(state="disabled")
            self.chat_display.see(tk.END)
            return

        content = read_as_chat_content(path)
        self.conversation.ask(f"--- /{cmd} ---", content)

    # 隱藏建議
    def hide_suggestions(self):
        if self.suggestion_win is not None:
            self.suggestion_win.destroy()
            self.suggestion_win = None
            self.suggestion_listbox = None
        self.selected_index = -1

    # 選擇建議
    def choose_suggestion(self, cmd: str):
        self.message.delete(0, tk.END)
        self.message.insert(0, f"/{cmd}")
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

    # 更新指令建議
    def update_suggestions(self, event=None):
        # 上下鍵／Tab／Enter／Esc 是在操作已經顯示出來的建議清單，不該讓清單重新整個重建
        if event is not None and event.keysym in ("Up", "Down", "Tab", "Return", "Escape"):
            return
        text = self.message.get()
        if text.startswith("/"):
            query = text[1:].lower()
            self.current_matches = [c for c in COMMANDS if c.lower().startswith(query)]
            self.selected_index = -1
            if self.current_matches:
                self.show_suggestions(self.current_matches)
            else:
                self.hide_suggestions()
        else:
            self.current_matches = []
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
