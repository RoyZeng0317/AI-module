import tkinter as tk
from tkinter import colorchooser, filedialog
import ctypes
import math
import os
import re
import sys
import threading
from pathlib import Path
import markdown

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "FileConvert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))

from chats import smart_reply_traced
from conversation import is_recording, start_recording, stop_recording
# 模型
MODULE = "sinco 1.5"
# 建立視窗
windows = tk.Tk()
# 視窗大小
windows.geometry("800x600")
# 視窗名稱
windows.title("AI Module")
# 預設背景顏色（Claude 風格深色主題）
DEFAULT_BG = "#191a1b"
BUBBLE_BG_AI = "#2a2b2e"       # sinco 訊息氣泡
BUBBLE_BG_USER = "#33415c"    # 使用者訊息氣泡（帶藍灰色調做出區隔）
TEXT_FG = "#e6e6e6"
SENDER_FG = "#9a9da3"
TRACE_FG = "#6f7378"
BUBBLE_FONT = ("Segoe UI", 10)
SENDER_FONT = ("Segoe UI", 8)
TRACE_FONT = ("Segoe UI", 8, "italic")
BUBBLE_TEXT_WIDTH = 50  # 字元數，決定氣泡最大寬度（換行點）
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

# ---------------------------------------------------------------------------
# 訊息氣泡：取代原本單一 Text 元件的整段對話紀錄，改成可捲動的 Canvas+Frame，
# 每則訊息各自是一個獨立的（唯讀）Text 元件（維持可選取/複製文字，跟原本
# Text 版一樣，不會因為改用 Label 而失去這個功能），依寄件者靠左/靠右對齊、
# 套上不同底色，做出類似 Claude 網頁版的訊息氣泡外觀。
# ---------------------------------------------------------------------------

def _autosize_bubble_text(widget: tk.Text):
    """把只顯示固定 width（字元數）的 Text 元件高度，依實際換行後的行數
    自動調整，讓每則氣泡剛好包住文字、不留多餘空白也不會被裁切。用
    "end-1c" 而不是 "end" 排除 Tk Text 元件內部永遠多出來的那個看不見的
    結尾空行（同樣的坑，ask_model() 原本處理「思考中」佔位文字時就踩過
    一次，這裡沿用同一個教訓）。
    """
    widget.update_idletasks()
    lines = widget.count("1.0", "end-1c", "displaylines")
    if isinstance(lines, tuple):
        lines = lines[0] if lines else 1
    widget.configure(height=max(1, lines or 1))


def add_bubble(sender: str, text: str, *, align: str = "left"):
    """在訊息串尾端加入一則新的氣泡，回傳 (body, trace_label) 兩個元件
    參照，讓呼叫端（ask_model 的背景執行緒完成時）可以直接改內容，取代
    原本靠 chat_display.tag_ranges("pending") 去找範圍再刪除重插的做法。
    """
    is_user = align == "right"
    bubble_bg = BUBBLE_BG_USER if is_user else BUBBLE_BG_AI
    anchor = "e" if is_user else "w"

    row = tk.Frame(messages_frame, bg=DEFAULT_BG)
    row.pack(fill="x", padx=8, pady=(4, 0))

    column = tk.Frame(row, bg=DEFAULT_BG)
    column.pack(side="right" if is_user else "left")

    tk.Label(column, text=sender, bg=DEFAULT_BG, fg=SENDER_FG, font=SENDER_FONT).pack(anchor=anchor)

    bubble = tk.Frame(column, bg=bubble_bg)
    bubble.pack(anchor=anchor)

    trace_label = tk.Label(
        bubble, text="", bg=bubble_bg, fg=TRACE_FG, font=TRACE_FONT,
        wraplength=BUBBLE_TEXT_WIDTH * 7, justify="left",
    )
    # 「思考過程」只有 AI 回覆才有，先不 pack；有內容時（finish_bubble）
    # 才用 pack(before=body) 插到內文上方，避免沒有內容時留一段空白。

    body = tk.Text(
        bubble, wrap="word", width=BUBBLE_TEXT_WIDTH, bg=bubble_bg, fg=TEXT_FG,
        relief="flat", bd=0, padx=10, pady=8, highlightthickness=0, font=BUBBLE_FONT,
    )
    body.insert("1.0", text)
    body.configure(state="disabled")
    body.pack()
    _autosize_bubble_text(body)

    scroll_to_bottom()
    return body, trace_label


def update_bubble(body: tk.Text, trace_label: tk.Label, text: str, trace: str | None = None):
    """就地更新一則已經存在的氣泡內容——給「思考中...」佔位文字換成真正
    回覆用，不用整個重建。
    """
    if trace:
        trace_label.configure(text=f"思考過程：{trace}")
        trace_label.pack(anchor="w", padx=10, pady=(8, 0), before=body)

    body.configure(state="normal")
    body.delete("1.0", "end")
    body.insert("1.0", text)
    body.configure(state="disabled")
    _autosize_bubble_text(body)
    scroll_to_bottom()


def add_system_bubble(text: str):
    """語音辨識通知、未知指令等沒有真正模型回覆的系統訊息，沿用 AI 側的
    靠左氣泡樣式，但寄件者標籤留白（不是 sinco 說的話）。
    """
    add_bubble("", text, align="left")


def clear_bubbles():
    for child in messages_frame.winfo_children():
        child.destroy()

# 問模型：smart_reply_traced() 跑天氣/搜尋查詢或神經網路推論時可能要花幾秒，
# 同步呼叫會把整個 Tkinter 視窗凍住，所以先插入一個「思考中」佔位氣泡，實際
# 運算丟到背景執行緒跑，跑完用 windows.after() 排回主執行緒把佔位內容換成
# 「思考過程」（走的是哪條路徑：即時查詢／程式碼模型／聊天模型，都是真實
# 的路由結果，不是裝飾用的假動畫）加上最終回覆
busy = False

def ask_model(header: str, model_input: str):
    global busy
    if busy:
        return
    busy = True

    add_bubble("You", header, align="right")
    pending_body, pending_trace = add_bubble("sinco", "思考中...", align="left")
    animate_responding()

    def worker():
        trace, reply = smart_reply_traced(model_input)

        def show_result():
            global busy
            update_bubble(pending_body, pending_trace, reply, trace)
            busy = False

        windows.after(0, show_result)

    threading.Thread(target=worker, daemon=True).start()
# 訊息框
def messagebox(user_text: str):
    ask_model(user_text, user_text)
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
        clear_bubbles()
        return

    path = COMMAND_DIR / f"{cmd}.md"
    if not path.exists():
        add_system_bubble(f"未知指令：/{cmd}")
        return

    content = read_as_chat_content(path)
    ask_model(f"/{cmd}", content)

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
# 語音波形：錄音中顯示麥克風即時音量，AI 回覆中顯示脈動動畫；兩者共用同一組
# bar 高度陣列與畫布，靠 draw_waveform() 統一畫出來
WAVEFORM_BARS = 16
waveform_levels = [0.0] * WAVEFORM_BARS
_pulse_phase = 0

def draw_waveform():
    voice_canvas.delete("all")
    width = voice_canvas.winfo_width() or int(voice_canvas["width"])
    height = voice_canvas.winfo_height() or int(voice_canvas["height"])
    bar_width = width / WAVEFORM_BARS
    for i, level in enumerate(waveform_levels):
        bar_height = max(2, level * height)
        x0 = i * bar_width + 1
        x1 = x0 + bar_width - 2
        y0 = (height - bar_height) / 2
        y1 = y0 + bar_height
        voice_canvas.create_rectangle(x0, y0, x1, y1, fill="#3ddc84", outline="")

def reset_waveform():
    global waveform_levels
    waveform_levels = [0.0] * WAVEFORM_BARS
    draw_waveform()
# 錄音中：麥克風回呼跑在音訊執行緒，不能直接動 Tk 元件，一定要透過 windows.after() 轉回主執行緒
def _on_mic_level(level: float):
    def apply():
        waveform_levels.pop(0)
        waveform_levels.append(level)
        draw_waveform()
    windows.after(0, apply)
# AI 回覆中：沒有真的音訊可以畫，用一個會晃動的假波形讓使用者知道「正在回應」
def animate_responding():
    if not busy:
        reset_waveform()
        return
    global _pulse_phase
    _pulse_phase += 1
    for i in range(WAVEFORM_BARS):
        waveform_levels[i] = 0.25 + 0.65 * abs(math.sin(_pulse_phase * 0.3 + i * 0.5))
    draw_waveform()
    windows.after(80, animate_responding)
# 麥克風按鈕：click 開始錄音、再 click 一次結束錄音並送去辨識
def voice():
    if busy:
        return
    if is_recording():
        text = stop_recording()
        voice_button.configure(text="🎤", fg="white")
        reset_waveform()
        if not text:
            return
        if "尚未訓練" in text:
            add_system_bubble(f"[語音] {text}")
            return
        messagebox(text)
    else:
        try:
            start_recording(on_level=_on_mic_level)
        except Exception as exc:
            add_system_bubble(f"[語音] 無法開啟麥克風：{exc}")
            return
        voice_button.configure(text="■", fg="#e74c3c")

input_frame = tk.Frame(windows, relief="sunken", bd=1)
input_frame.pack(side="bottom", pady=(0, 8))

chat_frame = tk.Frame(windows, bg=DEFAULT_BG)
chat_frame.pack(side="top", fill="both", expand=True, padx=8, pady=8)

chat_canvas = tk.Canvas(chat_frame, bg=DEFAULT_BG, highlightthickness=0)
chat_scrollbar = tk.Scrollbar(chat_frame, orient="vertical", command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=chat_scrollbar.set)
chat_scrollbar.pack(side="right", fill="y")
chat_canvas.pack(side="left", fill="both", expand=True)

# 可捲動的訊息容器：Tkinter 沒有原生的 scrollable frame，標準做法是把一個
# Frame 放進 Canvas 的 create_window，再靠 Canvas 的 scrollregion + Scrollbar
# 捲動；<Configure> 綁定讓內容變化時自動更新可捲動範圍，也讓容器寬度跟著
# 視窗縮放同步（否則氣泡對齊在視窗變寬/變窄後會跟原本的寬度對不齊）。
messages_frame = tk.Frame(chat_canvas, bg=DEFAULT_BG)
messages_window = chat_canvas.create_window((0, 0), window=messages_frame, anchor="nw")

def _on_messages_configure(event=None):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
messages_frame.bind("<Configure>", _on_messages_configure)

def _on_canvas_configure(event):
    chat_canvas.itemconfigure(messages_window, width=event.width)
chat_canvas.bind("<Configure>", _on_canvas_configure)

def scroll_to_bottom():
    windows.update_idletasks()
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)

def _on_mousewheel(event):
    chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    return "break"
# 只在滑鼠停留在聊天區域時才攔截滾輪事件，離開就解除，避免搶走其他元件
# （例如指令建議清單）原本的捲動行為
chat_canvas.bind("<Enter>", lambda e: chat_canvas.bind_all("<MouseWheel>", _on_mousewheel))
chat_canvas.bind("<Leave>", lambda e: chat_canvas.unbind_all("<MouseWheel>"))

tk.Button(input_frame, text="+", relief="flat", bd=0, command=open_file).pack(side="left", padx=(4, 0))

voice_button = tk.Button(input_frame, text="🎤", relief="flat", bd=0, fg="white", bg=DEFAULT_BG, command=voice)
voice_button.pack(side="left", padx=(4, 0))

voice_canvas = tk.Canvas(input_frame, width=90, height=24, bg=DEFAULT_BG, highlightthickness=0)
voice_canvas.pack(side="left", padx=(4, 0))

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
