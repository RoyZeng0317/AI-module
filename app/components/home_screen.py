import math
import os
import re
import sys
import tkinter as tk
from typing import Callable

import fitz  # PyMuPDF：kicad_dataset_convert.py 已經在用它把 PDF/SVG 光柵化成點陣圖，
             # 這裡沿用同一顆套件把下面的 FontAwesome SVG 轉成 Tkinter 看得懂的點陣圖示，
             # 不必再另外裝 cairosvg（在 Windows 上常常要另外裝原生 Cairo 函式庫才能用）。
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "FileConvert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web", "backend"))

from camera import CameraPanel
from command import CommandPalette
from conversation import Conversation, is_recording, start_recording, stop_recording

MODULE = "sinco 1.5"

# SVG code
send = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M322.5 351.7L523.4 150.9L391 520.3L322.5 351.7zM489.4 117L288.6 317.8L120 249.3L489.4 117zM70.1 280.8L275.9 364.4L359.5 570.2C364.8 583.3 377.6 591.9 391.8 591.9C406.5 591.9 419.6 582.7 424.6 568.8L602.6 72C606.1 62.2 603.6 51.4 596.3 44C589 36.6 578.1 34.2 568.3 37.7L71.4 215.7C57.5 220.7 48.3 233.8 48.3 248.5C48.3 262.7 56.9 275.5 70 280.8z"/></svg>'
microphone = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M320 64C267 64 224 107 224 160L224 288C224 341 267 384 320 384C373 384 416 341 416 288L416 160C416 107 373 64 320 64zM176 248C176 234.7 165.3 224 152 224C138.7 224 128 234.7 128 248L128 288C128 385.9 201.3 466.7 296 478.5L296 528L248 528C234.7 528 224 538.7 224 552C224 565.3 234.7 576 248 576L392 576C405.3 576 416 565.3 416 552C416 538.7 405.3 528 392 528L344 528L344 478.5C438.7 466.7 512 385.9 512 288L512 248C512 234.7 501.3 224 488 224C474.7 224 464 234.7 464 248L464 288C464 367.5 399.5 432 320 432C240.5 432 176 367.5 176 288L176 248z"/></svg>'

_SVG_ICON_CACHE: dict[tuple[str, int, str], "ImageTk.PhotoImage"] = {}


def _tint_svg(svg_source: str, color: str) -> str:
    """把 fill 直接寫在 <svg> 根節點上，靠 SVG fill 屬性的繼承性套用到底下所有
    <path>（FontAwesome 匯出的 path 本身沒寫 fill，預設繼承父層），這樣同一份
    SVG 原始碼就能依需要重繪成 MUTED_FG／ACCENT／BG 等不同顏色。"""
    return re.sub(r"<svg ", f'<svg fill="{color}" ', svg_source, count=1)


def render_svg_icon(svg_source: str, size: int, color: str) -> "ImageTk.PhotoImage":
    """把 FontAwesome 的 SVG 原始碼（send / microphone）用 PyMuPDF 光柵化成指定
    大小/顏色的點陣圖，再包成 Tkinter 認得的 PhotoImage。用 (svg_source, size,
    color) 當快取 key：同一個圖示同一個顏色只會渲染一次，順便讓這些 PhotoImage
    物件一直被這個全域字典參照著——Tkinter 的 PhotoImage 沒有其他地方引用時，
    下一次垃圾回收就會被清掉、Canvas 上會突然變成空白，這是常見的坑。
    """
    cache_key = (svg_source, size, color)
    cached = _SVG_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tinted = _tint_svg(svg_source, color)
    doc = fitz.open(stream=tinted.encode("utf-8"), filetype="svg")
    page = doc[0]
    scale = size / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    photo = ImageTk.PhotoImage(image)
    _SVG_ICON_CACHE[cache_key] = photo
    return photo

# 仿 Claude.ai 深色主題的暖色調色盤（肉眼比對抓出來的近似值，不是官方色票）。
BG = "#262624"          # 視窗主背景
INPUT_BG = "#30302E"     # 輸入列（圓角膠囊）背景
BUBBLE_BG = "#3A3936"    # 使用者訊息「氣泡」背景
BORDER = "#3E3D39"       # 分隔線 / 邊框
TEXT_FG = "#F3F1EA"      # 主要文字（暖白）
MUTED_FG = "#9C9890"     # 次要/說明文字（暖灰）
ACCENT = "#D97757"       # Claude 品牌強調色（陶土橘）：送出鍵、AI 名稱、標題重點
ACCENT_HOVER = "#C56A4C"  # 送出鍵 hover 時用的深一階陶土橘
FONT_FAMILY = "Segoe UI"


def rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius, **kwargs):
    """在 canvas 上畫一個圓角矩形（Tk 原生沒有這個圖元，用 12 個控制點 + smooth
    多邊形近似出圓角，這是 Tkinter 畫圓角矩形的標準做法）。"""
    radius = max(2, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class VectorIconCanvas(tk.Canvas):
    """帶 icon_type/redraw 的 Canvas 子類別，讓 make_claude_icon_button() 畫出的
    向量圖示按鈕可以事後換圖案（例如麥克風錄音中變成停止方塊）。單純繼承 tk.Canvas
    加型別註記，不改變任何行為，只是讓這兩個屬性是「宣告過的」，而不是動態塞到
    不知情的 tk.Canvas 實例上。
    """
    icon_type: str
    redraw: Callable[..., None]


def make_claude_icon_button(parent, icon_type, command, size=34, icon_color=None, hover_bg=None):
    """Claude 風格圖示按鈕：不用 Emoji/文字符號，改以 Canvas 繪製簡約幾何線條
    (Vector Icon)，跟 tk.Button 在 Windows 上 bd=0/relief="flat" 仍會畫出一圈
    系統亮框的問題無關——這裡本來就是純 Canvas 畫，highlightthickness=0 天生乾淨。
    icon_type 存成 canvas.icon_type（而不是閉包常數），讓外部可以事後動態換圖示
    （例如麥克風錄音中變成停止方塊），呼叫 canvas.redraw() 重繪即可。
    """
    icon_color = icon_color or MUTED_FG
    hover_bg = hover_bg or BORDER
    canvas = VectorIconCanvas(parent, width=size, height=size, bg=INPUT_BG, highlightthickness=0, cursor="hand2")
    canvas.icon_type = icon_type

    # 繪製 Icon 線條
    def draw_icon(color=None):
        color = color or icon_color
        canvas.delete("icon")
        cx, cy = size / 2, size / 2
        t = canvas.icon_type

        if t == "plus":  # 「+」新增附件
            canvas.create_line(cx - 5, cy, cx + 5, cy, fill=color, width=1.8, tags="icon")
            canvas.create_line(cx, cy - 5, cx, cy + 5, fill=color, width=1.8, tags="icon")

        elif t == "mic":  # 麥克風：改用真的 FontAwesome SVG（line 19），不再手繪幾何形狀
            icon = render_svg_icon(microphone, int(size * 0.5), color)
            canvas.create_image(cx, cy, image=icon, tags="icon")

        elif t == "camera":  # 相機/鏡頭 (雙重圓環)
            canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline=color, width=1.5, tags="icon")
            canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=color, outline="", tags="icon")

        elif t == "stop":  # 錄音中：停止方塊
            canvas.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill=color, outline="", tags="icon")

    # 滑鼠 Hover 效果 (浮現圓形灰底)
    def on_enter(e):
        canvas.delete("bg")
        canvas.create_oval(2, 2, size - 2, size - 2, fill=hover_bg, outline="", tags="bg")
        canvas.tag_lower("bg")

    def on_leave(e):
        canvas.delete("bg")

    canvas.redraw = draw_icon
    draw_icon()
    canvas.bind("<Enter>", on_enter)
    canvas.bind("<Leave>", on_leave)
    canvas.bind("<Button-1>", lambda e: command())

    return canvas


windows = tk.Tk()
windows.geometry("800x600")
windows.title(f"{MODULE}")
windows.iconbitmap()
windows.configure(bg=BG)

# 頂部標題列：強調色圓點 + 名稱，仿 Claude 網頁版左上角的品牌識別，跟下面聊天區
# 之間用一條細分隔線隔開
header_frame = tk.Frame(windows, bg=BG)
header_frame.pack(side="top", fill="x")

header_dot = tk.Canvas(header_frame, width=10, height=10, bg=BG, highlightthickness=0)
header_dot.pack(side="left", padx=(16, 8), pady=12)
header_dot.create_oval(1, 1, 9, 9, fill=ACCENT, outline="")

tk.Label(
    header_frame, text=MODULE, bg=BG, fg=TEXT_FG, font=(FONT_FAMILY, 11, "bold"),
).pack(side="left", pady=12)

tk.Frame(windows, bg=BORDER, height=1).pack(side="top", fill="x")

# 輸入列：外層 input_outer 只負責跟視窗邊緣留白，input_canvas 畫圓角膠囊背景，
# input_frame（embed 進 canvas 裡的一般 Frame）才是真正放按鈕/輸入框的地方——
# 之後的按鈕/Entry 建立方式跟改版前完全一樣，只是 parent 從「直接掛在視窗上的
# 方形 Frame」換成「掛在圓角 canvas 裡的 Frame」，其餘程式碼不用跟著改。
INPUT_HEIGHT = 52

input_outer = tk.Frame(windows, bg=BG)
input_outer.pack(side="bottom", fill="x", padx=18, pady=(4, 14))

input_canvas = tk.Canvas(input_outer, bg=BG, height=INPUT_HEIGHT, highlightthickness=0)
input_canvas.pack(fill="x")

input_frame = tk.Frame(input_canvas, bg=INPUT_BG)
_pill_id = None
_pill_window = input_canvas.create_window(0, 0, anchor="nw", window=input_frame)


def _redraw_pill(event=None):
    global _pill_id
    w, h = input_canvas.winfo_width(), input_canvas.winfo_height()
    if w < 4 or h < 4:
        return
    if _pill_id is not None:
        input_canvas.delete(_pill_id)
    _pill_id = rounded_rect(input_canvas, 1, 1, w - 1, h - 1, radius=h / 2, fill=INPUT_BG, outline=BORDER)
    input_canvas.tag_lower(_pill_id)
    input_canvas.itemconfig(_pill_window, width=w - 2, height=h - 2)
    input_canvas.coords(_pill_window, 1, 1)


input_canvas.bind("<Configure>", _redraw_pill)

chat_frame = tk.Frame(windows, bg=BG)
chat_frame.pack(side="top", fill="both", expand=True, padx=8, pady=8)

chat_scrollbar = tk.Scrollbar(chat_frame, bg=BG, troughcolor=BG, activebackground=ACCENT, bd=0)
chat_scrollbar.pack(side="right", fill="y")

chat_display = tk.Text(
    chat_frame, state="disabled", wrap="word", bg=BG, fg=TEXT_FG,
    relief="flat", bd=0, yscrollcommand=chat_scrollbar.set,
    padx=18, pady=14, spacing1=2, spacing3=10, font=(FONT_FAMILY, 10),
)
chat_display.pack(side="left", fill="both", expand=True)
chat_scrollbar.config(command=chat_display.yview)

# 訊息樣式：user_bubble 靠左邊界(lmargin)推遠 + justify=right，讓上色的「氣泡」
# 只出現在整行的右側一小段，視覺上接近 Claude 右對齊的使用者訊息氣泡；AI 這邊
# 是 sinco 名稱(ai_label) + 判斷依據(trace，小字/灰階，呼應 smart_reply_traced()
# 「誠實展示真正的路由決策」這個設計) + 實際回覆(ai_text)三段組成，不用氣泡框，
# 貼近 Claude 回覆本身不加框、靠純排版跟顏色區分身份的作法。
chat_display.tag_configure(
    "user_bubble", justify="right", background=BUBBLE_BG, foreground=TEXT_FG,
    lmargin1=90, lmargin2=90, rmargin=6, spacing1=10, spacing3=2, font=(FONT_FAMILY, 10),
)
chat_display.tag_configure("pending", foreground=MUTED_FG, font=(FONT_FAMILY, 9, "italic"))
chat_display.tag_configure("ai_label", foreground=ACCENT, font=(FONT_FAMILY, 10, "bold"), spacing1=10)
chat_display.tag_configure("trace", foreground=MUTED_FG, font=(FONT_FAMILY, 8))
chat_display.tag_configure("ai_text", foreground=TEXT_FG, font=(FONT_FAMILY, 10), spacing3=4)
chat_display.tag_configure("system", foreground=MUTED_FG, font=(FONT_FAMILY, 9), spacing1=6, spacing3=6)

message = tk.Entry(
    input_frame, relief="flat", bd=0,
    fg=TEXT_FG, bg=INPUT_BG, insertbackground=TEXT_FG,
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
    message.configure(fg=MUTED_FG)
    _showing_placeholder = True


def _clear_placeholder(event=None):
    global _showing_placeholder
    if _showing_placeholder:
        message.delete(0, tk.END)
        message.configure(fg=TEXT_FG)
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
        voice_canvas.create_rectangle(x0, y0, x1, y1, fill=ACCENT, outline="")


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
        voice_button.icon_type = "mic"
        voice_button.redraw(MUTED_FG)
        reset_waveform()
        tk.Misc.lower(voice_canvas, message)
        if not text:
            return
        if "尚未訓練" in text:
            chat_display.configure(state="normal")
            chat_display.insert(tk.END, f"[語音] {text}\n\n", "system")
            chat_display.configure(state="disabled")
            chat_display.see(tk.END)
            return
        conversation.send_message(text)
    else:
        try:
            start_recording(on_level=_on_mic_level)
        except Exception as exc:
            chat_display.configure(state="normal")
            chat_display.insert(tk.END, f"[語音] 無法開啟麥克風：{exc}\n\n", "system")
            chat_display.configure(state="disabled")
            chat_display.see(tk.END)
            return
        voice_button.icon_type = "stop"
        voice_button.redraw("#e74c3c")
        tk.Misc.lift(voice_canvas, message)


# 四顆按鈕都是 make_claude_icon_button() 畫出來的向量線條圖示鍵：「+」「mic」
# 「camera」平常完全沒有背景圓圈（純線條，跟膠囊背景融為一體），滑鼠移上去才浮現
# 一圈灰底，屬於次要動作；送出鍵固定是強調色實心圓，是唯一的主要動作，跟 Claude
# 輸入列「次要動作皆為 ghost icon、只有送出鍵是實色」的視覺層級一致。
# 送出鍵尺寸固定，不受視窗縮放影響；要先 pack(side="right") 讓它卡在膠囊右端，
# 再 pack 輸入框 fill="both" expand=True 去吃剩下的空間，兩者順序不能反過來。
make_claude_icon_button(input_frame, "plus", conversation.open_file).pack(side="left", padx=(8, 0), pady=9)

voice_button = make_claude_icon_button(input_frame, "mic", voice)
voice_button.pack(side="left", padx=(2, 0), pady=9)

make_claude_icon_button(input_frame, "camera", camera_panel.open).pack(side="left", padx=(2, 0), pady=9)

# 送出鍵不走 make_claude_icon_button()：它是唯一實心圓底的主要動作，圖示改用真的
# FontAwesome 紙飛機 SVG（line 18），不再手繪多邊形。圖示顏色固定用 BG（深色）
# 襯在 ACCENT 橘色圓底上，跟原本文字版 fg=BG 的慣例一致。
send_button = tk.Canvas(input_frame, width=34, height=34, bg=INPUT_BG, highlightthickness=0, cursor="hand2")


def draw_send_btn(bg_color):
    send_button.delete("all")
    send_button.create_oval(2, 2, 32, 32, fill=bg_color, outline="")
    icon = render_svg_icon(send, 17, BG)
    send_button.create_image(17, 17, image=icon, tags="icon")


draw_send_btn(ACCENT)
send_button.bind("<Enter>", lambda e: draw_send_btn(ACCENT_HOVER))
send_button.bind("<Leave>", lambda e: draw_send_btn(ACCENT))
send_button.bind("<Button-1>", lambda e: on_submit())
send_button.pack(side="right", padx=(4, 10), pady=9)

message.pack(side="left", fill="both", expand=True, ipady=6, padx=(8, 6))

# 錄音時的聲波疊在輸入框「裡面」，不要另外開一個聲波框：跟 message 疊在同一個
# 位置/同一塊大小（in_=message 讓座標以 message 為基準），預設沉到 message
# 底下（打字時完全看不到、看到的是正常輸入框），開始錄音才 lift 上來蓋住輸入
# 框、改顯示波形，停止錄音再 lower 回去，輸入框就恢復原狀。
voice_canvas = tk.Canvas(input_frame, bg=INPUT_BG, highlightthickness=0)
voice_canvas.place(in_=message, x=0, y=0, relwidth=1, relheight=1)
tk.Misc.lower(voice_canvas, message)

# ==================== 事件綁定 ====================
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

# AI 免責聲明：跟一般 AI 產品同樣的位置跟語氣，提醒使用者 sinco 可能出錯
disclaimer = tk.Label(
    windows, text="sinco is AI and can make mistakes. Please double-check response.",
    font=(FONT_FAMILY, 8), fg=MUTED_FG, bg=BG,
)
disclaimer.pack(side="bottom", pady=(0, 4))

menubar = tk.Menu(windows)
windows.config(menu=menubar)
chat_menu = tk.Menu(menubar, tearoff=0)
chat_menu.add_command(label="Open Chats", command=on_submit)
menubar.add_cascade(label="Chats", menu=chat_menu)

windows.mainloop()
