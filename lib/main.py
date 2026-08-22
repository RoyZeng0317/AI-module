# 下方庫不要變更(如: import math)保持現在這樣
import math, os, sys
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FileConvert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tranning"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "backend"))

if not __package__:
    # 直接用完整路徑執行這支檔案時（例如 VS Code「執行 Python 檔案」按鈕），
    # sys.path[0] 只會是 lib/ 這層目錄，專案根目錄不在 sys.path 裡，下面
    # `from lib.components...` 這種絕對匯入就會 ModuleNotFoundError: No module
    # named 'lib'。正規跑法是在專案根目錄下 `python -m lib.main`（見 CLAUDE.md），
    # 這裡補上保險：偵測到不是用 -m 執行時，把專案根目錄塞進 sys.path。
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GUI 呈現層（視窗/元件建構、顏色、字型、SVG 圖示繪製）都搬到 lib/GUI.py，
# 這支檔案只負責商業邏輯：把 Conversation/CommandPalette/CameraPanel 接上
# GUI 元件、處理事件、驅動 mainloop。GUI.py 不 import 這支檔案，避免循環匯入。
from lib import GUI as gui
from lib.components.camera import CameraPanel
from lib.components.command import CommandPalette
from lib.components.conversation import Conversation, is_recording, start_recording, stop_recording

# 語音波形動畫（麥克風即時音量 / AI 回覆中的假脈動）屬於商業邏輯，畫面本身
# （voice_canvas、waveform_levels、draw_waveform()）留在 gui。
_pulse_phase = 0


def reset_waveform():
    gui.waveform_levels[:] = [0.0] * gui.WAVEFORM_BARS
    gui.draw_waveform()


# 錄音中：麥克風回呼跑在音訊執行緒，不能直接動 Tk 元件，一定要透過 windows.after() 轉回主執行緒
def _on_mic_level(level: float):
    def apply():
        gui.waveform_levels.pop(0)
        gui.waveform_levels.append(level)
        gui.draw_waveform()
    gui.windows.after(0, apply)


# AI 回覆中：沒有真的音訊可以畫，用一個會晃動的假波形讓使用者知道「正在回應」；
# 掛在 Conversation 的 on_ask_start callback 上，一送出問題就開始跑
def animate_responding():
    if not conversation.busy:
        reset_waveform()
        return
    global _pulse_phase
    _pulse_phase += 1
    for i in range(gui.WAVEFORM_BARS):
        gui.waveform_levels[i] = 0.25 + 0.65 * abs(math.sin(_pulse_phase * 0.3 + i * 0.5))
    gui.draw_waveform()
    gui.windows.after(80, animate_responding)


conversation = Conversation(gui.windows, gui.chat_display, on_ask_start=animate_responding)
palette = CommandPalette(gui.windows, gui.message, gui.chat_display, conversation)
camera_panel = CameraPanel(gui.windows)


# 提交按鈕
def on_submit():
    if conversation.busy or gui._showing_placeholder:
        return
    text = gui.message.get().strip()
    if text:
        if text.startswith("/") and palette.selected_index >= 0 and palette.current_matches:
            palette.run(palette.current_matches[palette.selected_index])
        elif text.startswith("/"):
            palette.run(text[1:])
        else:
            palette.file(text)
    gui.message.delete(0, tk.END)
    palette.hide_suggestions()
    gui._show_placeholder()


# 麥克風按鈕：click 開始錄音、再 click 一次結束錄音並送去辨識
def voice():
    if conversation.busy:
        return
    if is_recording():
        text = stop_recording()
        voice_button.icon_type = "mic"
        voice_button.redraw(gui.MUTED_FG)
        reset_waveform()
        tk.Misc.lower(gui.voice_canvas, gui.message)
        if not text:
            return
        if "尚未訓練" in text:
            gui.chat_display.configure(state="normal")
            gui.chat_display.insert(tk.END, f"[語音] {text}\n\n", "system")
            gui.chat_display.configure(state="disabled")
            gui.chat_display.see(tk.END)
            return
        conversation.send_message(text)
    else:
        try:
            start_recording(on_level=_on_mic_level)
        except Exception as exc:
            gui.chat_display.configure(state="normal")
            gui.chat_display.insert(tk.END, f"[語音] 無法開啟麥克風：{exc}\n\n", "system")
            gui.chat_display.configure(state="disabled")
            gui.chat_display.see(tk.END)
            return
        voice_button.icon_type = "stop"
        voice_button.redraw("#e74c3c")
        tk.Misc.lift(gui.voice_canvas, gui.message)


# 四顆按鈕都是 make_claude_icon_button() 畫出來的向量線條圖示鍵：「+」「mic」
# 「camera」平常完全沒有背景圓圈（純線條，跟膠囊背景融為一體），滑鼠移上去才浮現
# 一圈灰底，屬於次要動作；送出鍵固定是強調色實心圓，是唯一的主要動作，跟 Claude
# 輸入列「次要動作皆為 ghost icon、只有送出鍵是實色」的視覺層級一致。
# 送出鍵尺寸固定，不受視窗縮放影響；要先 pack(side="right") 讓它卡在膠囊右端，
# 再 pack 輸入框 fill="both" expand=True 去吃剩下的空間，兩者順序不能反過來
# （gui.message/gui.send_button 在 GUI.py 裡刻意沒 pack，就是為了讓這個順序
# 由這裡統一控制）。
gui.make_claude_icon_button(gui.input_frame, "plus", conversation.open_file).pack(side="left", padx=(8, 0), pady=9)

voice_button = gui.make_claude_icon_button(gui.input_frame, "mic", voice)
voice_button.pack(side="left", padx=(2, 0), pady=9)

gui.make_claude_icon_button(gui.input_frame, "camera", camera_panel.open).pack(side="left", padx=(2, 0), pady=9)

gui.send_button.bind("<Button-1>", lambda e: on_submit())
gui.send_button.pack(side="right", padx=(4, 10), pady=9)

gui.message.pack(side="left", fill="both", expand=True, ipady=6, padx=(8, 6))

# ==================== 事件綁定 ====================
gui.message.bind("<Return>", lambda e: on_submit())
gui.message.bind("<KeyRelease>", palette.update_suggestions)
gui.message.bind("<KeyPress>", gui._clear_placeholder)
gui.message.bind("<FocusIn>", gui._clear_placeholder)
gui.message.bind("<FocusOut>", gui._restore_placeholder_if_empty)
gui.message.bind("<Escape>", lambda e: palette.hide_suggestions())
gui.message.bind("<Tab>", palette.complete)
gui.message.bind("<Up>", palette.on_arrow_up)
gui.message.bind("<Down>", palette.on_arrow_down)

gui.chat_menu.add_command(label="Open Chats", command=on_submit)

gui.windows.mainloop()
