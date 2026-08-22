# GUI 呈現層：只放視窗/元件建構與純視覺繪製（顏色、字型、SVG 圖示光柵化、
# 圓角膠囊、聊天氣泡樣式…），刻意不 import lib/main.py 或任何商業邏輯
# （Conversation / CommandPalette / CameraPanel），避免這支檔案跟 main.py
# 互相 import 造成循環匯入。main.py 是唯一的執行入口，由它 `import GUI as gui`
# 取用這裡的元件/常數，再把對話邏輯、指令面板、事件綁定掛上去。
import re
import tkinter as tk
from typing import Callable

import fitz  # PyMuPDF：把下面的 FontAwesome SVG 光柵化成 Tkinter 看得懂的點陣圖示，
             # 不必再另外裝 cairosvg（在 Windows 上常常要另外裝原生 Cairo 函式庫才能用）。
from PIL import Image, ImageTk

MODULE = "Sinco 1.5"  # 應用程式/品牌名稱（視窗標題/標題列用）；聊天時的助理人格名稱仍是 "sinco"（訓練資料/回覆內容都沒有改，見 CLAUDE.md to-do #15/#16）

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
    command 由呼叫端（main.py）傳入實際的商業邏輯 callback，這支函式本身不知道
    也不需要知道 conversation/camera_panel 等物件的存在。
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
# input_frame（embed 進 canvas 裡的一般 Frame）才是真正放按鈕/輸入框的地方。
# 按鈕/Entry 本身在這裡只建構元件、不 pack——實際要放哪些按鈕（含商業邏輯
# callback）跟 pack 的先後順序，由 main.py 統一決定，因為 pack 在同一個 side
# 上是照呼叫順序排版的，跨檔案拆開建立容易讓左右順序跑掉。
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

# md_* 系列：markdown_view.insert_markdown() 解析 sinco 回覆時套用的渲染 tag，
# 讓 ai_text 顯示的是排版後的「預覽」（標題變大字粗體、清單變項目符號、程式碼
# 區塊變等寬字），跟 CLI 版用 rich.markdown.Markdown 是同一個目的、不同技術棧
# （Tkinter Text 沒有內建 markdown 渲染能力，只能手動 tag_configure + insert）。
# 這些 tag 都是在 ai_text 之後才建立，Tk 的 tag 優先權預設「後建立的蓋過先建立
# 的」，所以疊加 ai_text 一起用時，字型/顏色會照這裡的設定顯示，不會被 ai_text
# 蓋掉。
CODE_FONT_FAMILY = "Consolas"
chat_display.tag_configure("md_h1", font=(FONT_FAMILY, 15, "bold"), foreground=TEXT_FG, spacing1=10, spacing3=4)
chat_display.tag_configure("md_h2", font=(FONT_FAMILY, 13, "bold"), foreground=TEXT_FG, spacing1=8, spacing3=3)
chat_display.tag_configure("md_h3", font=(FONT_FAMILY, 11, "bold"), foreground=TEXT_FG, spacing1=6, spacing3=2)
chat_display.tag_configure("md_bold", font=(FONT_FAMILY, 10, "bold"), foreground=TEXT_FG)
chat_display.tag_configure("md_italic", font=(FONT_FAMILY, 10, "italic"), foreground=TEXT_FG)
chat_display.tag_configure("md_code_inline", font=(CODE_FONT_FAMILY, 9), foreground=ACCENT, background=INPUT_BG)
chat_display.tag_configure(
    "md_code_block", font=(CODE_FONT_FAMILY, 9), foreground=TEXT_FG, background=INPUT_BG,
    lmargin1=24, lmargin2=24, spacing1=4, spacing3=4,
)
chat_display.tag_configure("md_quote", font=(FONT_FAMILY, 10, "italic"), foreground=MUTED_FG, lmargin1=24, lmargin2=24)
chat_display.tag_configure("md_bullet", font=(FONT_FAMILY, 10, "bold"), foreground=ACCENT)

# message 只在這裡「建構」，特意不 pack——要等 main.py 把 plus/mic/camera 等
# 按鈕都 pack(side="left") 完，最後才能 pack message(fill="both", expand=True)
# 去吃剩餘空間，順序反了三顆圖示鍵會被擠到看不見（Tk pack 同一 side 依呼叫
# 順序排版），所以 pack 動作留給 main.py 統一處理。
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


_show_placeholder()

# 語音波形：錄音中顯示麥克風即時音量，AI 回覆中顯示脈動動畫；兩者共用同一組
# bar 高度陣列與畫布，靠 draw_waveform() 統一畫出來。動畫本身（麥克風音量回呼、
# AI 回應中的假脈動）屬於商業邏輯，留在 main.py，這裡只負責把 waveform_levels
# 目前的值畫到 voice_canvas 上。
WAVEFORM_BARS = 16
waveform_levels = [0.0] * WAVEFORM_BARS


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


# 送出鍵不走 make_claude_icon_button()：它是唯一實心圓底的主要動作，圖示改用真的
# FontAwesome 紙飛機 SVG，不再手繪多邊形。圖示顏色固定用 BG（深色）襯在 ACCENT
# 橘色圓底上，跟原本文字版 fg=BG 的慣例一致。外觀/hover 在這裡就畫好，真正
# 「送出」的 <Button-1> commit 動作由 main.py 綁（那是 on_submit()，屬於商業
# 邏輯），pack 順序理由同 message。
send_button = tk.Canvas(input_frame, width=34, height=34, bg=INPUT_BG, highlightthickness=0, cursor="hand2")


def draw_send_btn(bg_color):
    send_button.delete("all")
    send_button.create_oval(2, 2, 32, 32, fill=bg_color, outline="")
    icon = render_svg_icon(send, 17, BG)
    send_button.create_image(17, 17, image=icon, tags="icon")


draw_send_btn(ACCENT)
send_button.bind("<Enter>", lambda e: draw_send_btn(ACCENT_HOVER))
send_button.bind("<Leave>", lambda e: draw_send_btn(ACCENT))

# 錄音時的聲波疊在輸入框「裡面」，不要另外開一個聲波框：跟 message 疊在同一個
# 位置/同一塊大小（in_=message 讓座標以 message 為基準），預設沉到 message
# 底下（打字時完全看不到、看到的是正常輸入框），開始錄音才 lift 上來蓋住輸入
# 框、改顯示波形，停止錄音再 lower 回去，輸入框就恢復原狀。這兩行只靠 place()
# 定位（跟 pack() 是獨立的兩套幾何管理器），不受上面 message 尚未 pack 影響。
voice_canvas = tk.Canvas(input_frame, bg=INPUT_BG, highlightthickness=0)
voice_canvas.place(in_=message, x=0, y=0, relwidth=1, relheight=1)
tk.Misc.lower(voice_canvas, message)

# AI 免責聲明：跟一般 AI 產品同樣的位置跟語氣，提醒使用者 sinco 可能出錯
disclaimer = tk.Label(
    windows, text="sinco is AI and can make mistakes. Please double-check response.",
    font=(FONT_FAMILY, 8), fg=MUTED_FG, bg=BG,
)
disclaimer.pack(side="bottom", pady=(0, 4))

# 選單殼在這裡建好，但「Open Chats」要接的 on_submit 是商業邏輯，
# add_command(...) 留給 main.py 呼叫 chat_menu.add_command(...)。
menubar = tk.Menu(windows)
windows.config(menu=menubar)
chat_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Chats", menu=chat_menu)
