import math
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "FileConvert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web", "backend"))

from camera import CameraPanel
from command import CommandPalette
from conversation import Conversation, is_recording, start_recording, stop_recording

MODULE = "sinco 1.5"

windows = tk.Tk()
windows.geometry("800x600")
windows.title(f"{MODULE}")
windows.iconbitmap()
DEFAULT_BG = "#191a1b"

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

message = tk.Entry(
    input_frame, width=30, relief="flat", bd=0,
    fg="white", bg=DEFAULT_BG, insertbackground="white",
)

# 輸入框提示文字：Tkinter Entry 沒有原生 placeholder，用「灰色提示字 + 手動清除/還原」
# 模擬。清除要同時掛在 FocusIn（滑鼠點進來）跟 KeyPress（送出後游標留在框內，
# 沒有觸發新的 FocusIn，靠這個抓住下一次打字）兩個事件上，不然直接打字會把
# 提示文字當成真的輸入內容夾在游標後面。
PLACEHOLDER_TEXT = "Enter message to sinco"
_showing_placeholder = False


def _show_placeholder():
    global _showing_placeholder
    message.delete(0, tk.END)
    message.insert(0, PLACEHOLDER_TEXT)
    message.configure(fg="grey")
    _showing_placeholder = True


def _clear_placeholder(event=None):
    global _showing_placeholder
    if _showing_placeholder:
        message.delete(0, tk.END)
        message.configure(fg="white")
        _showing_placeholder = False


def _restore_placeholder_if_empty(event=None):
    if not message.get():
        _show_placeholder()


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


# AI 回覆中：沒有真的音訊可以畫，用一個會晃動的假波形讓使用者知道「正在回應」；
# 掛在 Conversation 的 on_ask_start callback 上，一送出問題就開始跑
def animate_responding():
    if not conversation.busy:
        reset_waveform()
        return
    global _pulse_phase
    _pulse_phase += 1
    for i in range(WAVEFORM_BARS):
        waveform_levels[i] = 0.25 + 0.65 * abs(math.sin(_pulse_phase * 0.3 + i * 0.5))
    draw_waveform()
    windows.after(80, animate_responding)


conversation = Conversation(windows, chat_display, on_ask_start=animate_responding)
palette = CommandPalette(windows, message, chat_display, conversation)
camera_panel = CameraPanel(windows)


# 提交按鈕
def on_submit():
    if conversation.busy or _showing_placeholder:
        return
    text = message.get().strip()
    if text:
        if text.startswith("/") and palette.selected_index >= 0 and palette.current_matches:
            palette.run(palette.current_matches[palette.selected_index])
        elif text.startswith("/"):
            palette.run(text[1:])
        else:
            conversation.send_message(text)
    message.delete(0, tk.END)
    palette.hide_suggestions()
    _show_placeholder()


# 麥克風按鈕：click 開始錄音、再 click 一次結束錄音並送去辨識
def voice():
    if conversation.busy:
        return
    if is_recording():
        text = stop_recording()
        voice_button.configure(text="🎤", fg="white")
        reset_waveform()
        tk.Misc.lower(voice_canvas, message)
        if not text:
            return
        if "尚未訓練" in text:
            chat_display.configure(state="normal")
            chat_display.insert(tk.END, f"[語音] {text}\n\n")
            chat_display.configure(state="disabled")
            chat_display.see(tk.END)
            return
        conversation.send_message(text)
    else:
        try:
            start_recording(on_level=_on_mic_level)
        except Exception as exc:
            chat_display.configure(state="normal")
            chat_display.insert(tk.END, f"[語音] 無法開啟麥克風：{exc}\n\n")
            chat_display.configure(state="disabled")
            chat_display.see(tk.END)
            return
        voice_button.configure(text="■", fg="#e74c3c")
        tk.Misc.lift(voice_canvas, message)


tk.Button(input_frame, text="+", relief="flat", bd=0, command=conversation.open_file).pack(side="left", padx=(4, 0))

voice_button = tk.Button(input_frame, text="🎤", relief="flat", bd=0, fg="white", bg=DEFAULT_BG, command=voice)
voice_button.pack(side="left", padx=(4, 0))

tk.Button(
    input_frame, text="◉", relief="flat", bd=0, fg="white", bg=DEFAULT_BG,
    command=camera_panel.open,
).pack(side="left", padx=(4, 0))

message.pack(side="left", fill="y", ipady=4)

# 錄音時的聲波疊在輸入框「裡面」，不要另外開一個聲波框：跟 message 疊在同一個
# 位置/同一塊大小（in_=message 讓座標以 message 為基準），預設沉到 message
# 底下（打字時完全看不到、看到的是正常輸入框），開始錄音才 lift 上來蓋住輸入
# 框、改顯示波形，停止錄音再 lower 回去，輸入框就恢復原狀。
voice_canvas = tk.Canvas(input_frame, bg=DEFAULT_BG, highlightthickness=0)
voice_canvas.place(in_=message, x=0, y=0, relwidth=1, relheight=1)
tk.Misc.lower(voice_canvas, message)
message.bind("<Return>", lambda e: on_submit())
message.bind("<KeyRelease>", palette.update_suggestions)
message.bind("<KeyPress>", _clear_placeholder)
message.bind("<FocusIn>", _clear_placeholder)
message.bind("<FocusOut>", _restore_placeholder_if_empty)
message.bind("<Escape>", lambda e: palette.hide_suggestions())
message.bind("<Tab>", palette.complete)
message.bind("<Up>", palette.on_arrow_up)
message.bind("<Down>", palette.on_arrow_down)
_show_placeholder()

tk.Button(input_frame, text="➤", relief="flat", bd=0, command=on_submit).pack(side="left")

# AI 免責聲明：跟一般 AI 產品同樣的位置跟語氣，提醒使用者 sinco 可能出錯
disclaimer = tk.Label(
    windows, text="sinco is AI and can make mistakes. Please double-check response.",
    font=("Segoe UI", 8), fg="grey", bg=DEFAULT_BG,
)
disclaimer.pack(side="bottom", pady=(0, 4))

menubar = tk.Menu(windows)
windows.config(menu=menubar)
windows.configure(bg=DEFAULT_BG)
chat_menu = tk.Menu(menubar, tearoff=0)
chat_menu.add_command(label="Open Chats", command=on_submit)
menubar.add_cascade(label="Chats", menu=chat_menu)

windows.mainloop()
