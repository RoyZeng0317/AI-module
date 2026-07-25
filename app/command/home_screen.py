import tkinter as tk
from tkinter import colorchooser, filedialog
import ctypes
import os
import re
import sys
import threading
from pathlib import Path
import markdown

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "FileConvert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))

from chats import smart_reply_traced
# 模型
MODULE = "sinco 1.5"
# 建立視窗
windows = tk.Tk()
# 視窗大小
windows.geometry("800x600")
# 視窗名稱
windows.title("AI Module")
# 預設背景顏色
DEFAULT_BG = "#191a1b"
# 
bot_reply = {
    "Hello": "Hello, I'm AI module, what kind of things need to help?",
    "": ""
}
# 支援所有檔案格式；markdown 檔案先轉成純文字再交給模型
MARKDOWN_EXTENSIONS = {".md", ".markdown"}

def read_as_chat_content(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    if path.suffix.lower() in MARKDOWN_EXTENSIONS:
        html = markdown.markdown(raw_text)
        return re.sub(r"<[^>]+>", "", html).strip()
    return raw_text.strip()

# 指令集：clear 是內建動作，其餘指令名稱直接取自本檔案所在資料夾內的 .md 檔名
COMMAND_DIR = Path(__file__).resolve().parent
BUILTIN_COMMANDS = {"clear"}
COMMANDS = sorted(BUILTIN_COMMANDS | {p.stem for p in COMMAND_DIR.glob("*.md")})

# 問模型：smart_reply_traced() 跑天氣/搜尋查詢或神經網路推論時可能要花幾秒，
# 同步呼叫會把整個 Tkinter 視窗凍住，所以先插入一行「思考中」佔位文字，實際
# 運算丟到背景執行緒跑，跑完用 windows.after() 排回主執行緒把佔位文字換成
# 「思考過程」（走的是哪條路徑：即時查詢／程式碼模型／聊天模型，都是真實
# 的路由結果，不是裝飾用的假動畫）加上最終回覆
busy = False

def ask_model(header: str, model_input: str):
    global busy
    if busy:
        return
    busy = True

    chat_display.configure(state="normal")
    chat_display.insert(tk.END, f"{header}\n")
    # 用 tag 記錄「思考中」佔位文字的範圍，而不是存一個固定的 index 字串——
    # 實測發現 chat_display.index(tk.END) 抓到的位置，跟接下來 insert(tk.END, ...)
    # 實際落下去的位置會差一行（Tk Text 內部永遠多一個看不見的結尾空行），拿
    # 那個位置去 delete 會少刪一行，「思考中...」殘留、新內容直接接在後面。
    # tag_ranges() 會自動追蹤標記過的文字範圍，不受這個位置誤差影響。
    chat_display.insert(tk.END, "思考中...\n\n", "pending")
    chat_display.configure(state="disabled")
    chat_display.see(tk.END)

    def worker():
        trace, reply = smart_reply_traced(model_input)

        def show_result():
            global busy
            chat_display.configure(state="normal")
            start, end = chat_display.tag_ranges("pending")
            chat_display.delete(start, end)
            chat_display.insert(start, f"[思考過程：{trace}]\nAI: {reply}\n\n")
            chat_display.configure(state="disabled")
            chat_display.see(tk.END)
            busy = False

        windows.after(0, show_result)

    threading.Thread(target=worker, daemon=True).start()
# 訊息框
def messagebox(user_text: str):
    ask_model(f"You: {user_text}", user_text)
# 打開檔案
def open_file():
    path = filedialog.askopenfilename(
        title="Open File",
        filetypes=[("All files", "*.*")],
    )
    if not path:
        return

    content = read_as_chat_content(Path(path))
    ask_model(f"--- {os.path.basename(path)} ---\n{content}", content)
# 指令：clear 是內建動作；其餘指令對應 /command 資料夾同名的 .md 檔，
# 讀出內容交給模型當提示詞，不再用 if/elif 逐一寫死每個指令的行為
def command(cmd: str):
    if cmd == "clear":
        chat_display.configure(state="normal")
        chat_display.delete("1.0", tk.END)
        chat_display.configure(state="disabled")
        return

    path = COMMAND_DIR / f"{cmd}.md"
    if not path.exists():
        chat_display.configure(state="normal")
        chat_display.insert(tk.END, f"未知指令：/{cmd}\n\n")
        chat_display.configure(state="disabled")
        chat_display.see(tk.END)
        return

    content = read_as_chat_content(path)
    ask_model(f"--- /{cmd} ---", content)

suggestion_win = None
suggestion_listbox = None
current_matches = []
selected_index = -1
# 隱藏建議
def hide_suggestions():
    global suggestion_win, suggestion_listbox, selected_index
    if suggestion_win is not None:
        suggestion_win.destroy()
        suggestion_win = None
        suggestion_listbox = None
    selected_index = -1
# 選擇建議
def choose_suggestion(cmd):
    message.delete(0, tk.END)
    message.insert(0, f"/{cmd}")
    hide_suggestions()
    message.focus_set()
# 顯示建議
def show_suggestions(matches):
    global suggestion_win, suggestion_listbox
    hide_suggestions()
    suggestion_win = tk.Toplevel(windows)
    suggestion_win.overrideredirect(True)
    row_height = 20
    x = message.winfo_rootx()
    y = message.winfo_rooty() - (row_height * len(matches)) - 4
    suggestion_win.geometry(f"+{x}+{y}")
    listbox = tk.Listbox(suggestion_win, height=len(matches), bd=1, relief="solid", exportselection=False)
    listbox.pack()
    for cmd in matches:
        listbox.insert(tk.END, cmd)
    # 綁定時抓區域變數 listbox（一定是實例，不是 Optional），不要在 lambda
    # 裡引用模組全域的 suggestion_listbox，避免型別檢查器誤判它可能是 None
    listbox.bind("<<ListboxSelect>>", lambda e: choose_suggestion(matches[listbox.curselection()[0]]))
    suggestion_listbox = listbox
# 更新指令建議
def update_command_suggestions(event=None):
    global current_matches, selected_index
    # 上下鍵／Tab／Enter／Esc 是在操作已經顯示出來的建議清單，不該讓清單重新整個重建
    if event is not None and event.keysym in ("Up", "Down", "Tab", "Return", "Escape"):
        return
    text = message.get()
    if text.startswith("/"):
        query = text[1:].lower()
        current_matches = [c for c in COMMANDS if c.lower().startswith(query)]
        selected_index = -1
        if current_matches:
            show_suggestions(current_matches)
        else:
            hide_suggestions()
    else:
        current_matches = []
        hide_suggestions()
# 上下鍵：在目前顯示的建議清單裡移動反白選項
def move_selection(delta):
    global selected_index
    if not current_matches or suggestion_listbox is None:
        return "break"
    selected_index = (selected_index + delta) % len(current_matches)
    suggestion_listbox.selection_clear(0, tk.END)
    suggestion_listbox.selection_set(selected_index)
    suggestion_listbox.activate(selected_index)
    suggestion_listbox.see(selected_index)
    return "break"

def on_arrow_up(event):
    return move_selection(-1)

def on_arrow_down(event):
    return move_selection(1)
# Tab 快速鍵：補成目前反白的指令（沒有用上下鍵選過就用第一個），不用打完整指令名稱
def complete_command(event=None):
    if not current_matches:
        return None
    idx = selected_index if selected_index >= 0 else 0
    choose_suggestion(current_matches[idx])
    return "break"  # 阻止 Tab 預設把焦點跳到下一個元件
# 提交按鈕
def on_submit():
    if busy:
        return
    text = message.get().strip()
    if text:
        if text.startswith("/") and selected_index >= 0 and current_matches:
            command(current_matches[selected_index])
        elif text.startswith("/"):
            command(text[1:])
        else:
            messagebox(text)
    message.delete(0, tk.END)
    hide_suggestions()
# 背景顏色
def bg():
    color = colorchooser.askcolor(color=windows["bg"], title="Choose Background Color")[1]
    if color:
        windows.configure(bg=color)
def voice():
    tk.Button("Microphone")
    # 與 AI 進行對話
    # 要能夠輸出生波看起來像是有在聆聽以及回應
    
input_frame = tk.Frame(windows, relief="sunken", bd=1)
input_frame.pack(side="bottom", pady=(0, 8))

chat_frame = tk.Frame(windows)
chat_frame.pack(side="top", fill="both", expand=True, padx=8, pady=8)

chat_scrollbar = tk.Scrollbar(chat_frame)
chat_scrollbar.pack(side="right", fill="y")

chat_display = tk.Text(
    chat_frame, state="disabled", wrap="word", bg=DEFAULT_BG, fg="white",
    relief="flat", bd=0, yscrollcommand=chat_scrollbar.set,
)
chat_display.pack(side="left", fill="both", expand=True)
chat_scrollbar.config(command=chat_display.yview)

tk.Button(input_frame, text="+", relief="flat", bd=0, command=open_file).pack(side="left", padx=(4, 0))

message = tk.Entry(input_frame, width=30, relief="flat", bd=0)
message.pack(side="left", fill="y", ipady=4)
message.bind("<Return>", lambda e: on_submit())
message.bind("<KeyRelease>", update_command_suggestions)
message.bind("<Escape>", lambda e: hide_suggestions())
message.bind("<Tab>", complete_command)
message.bind("<Up>", on_arrow_up)
message.bind("<Down>", on_arrow_down)

tk.Button(input_frame, text="➤", relief="flat", bd=0, command=on_submit).pack(side="left")

menubar = tk.Menu(windows)
windows.config(menu=menubar)
windows.configure(bg=DEFAULT_BG)
chat_menu = tk.Menu(menubar, tearoff=0)
chat_menu.add_command(label="Open Chats", command=on_submit)
menubar.add_cascade(label="Chats", menu=chat_menu)

windows.mainloop()
