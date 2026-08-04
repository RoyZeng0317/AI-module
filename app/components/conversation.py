"""Everything that gets a user's input to sinco and the reply back — whether
typed (Conversation class) or spoken (the voice functions below).

Conversation talks to sinco: shows a "思考中" placeholder immediately, runs
smart_reply_traced() on a background thread so the Tk mainloop never freezes
(weather/search lookups and the code-model forward pass can take a couple of
seconds), then swaps the placeholder for the real trace + reply.

is_recording()/start_recording()/stop_recording() are the mic-capture side:
record from the default microphone, then run the from-scratch STT model
(tranning/speech_to_text.py, a CRNN+CTC model — same architecture as
tranning/OCR.py, just fed a spectrogram instead of a text-line image) to
turn the recording into text. No cloud speech API (Rule 06). Kept free of
any Tkinter drawing logic on purpose — home_screen.py owns the waveform
canvas and decides how recording state gets shown; this module only knows
about audio in/out and text.

home_screen.py must add tranning/ to sys.path before importing this module
(that's where chats.py and speech_to_text.py live).
"""

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))

from chats import DEFAULT_OUT_DIR, smart_reply_traced  # noqa: E402
from function import read_as_chat_content  # noqa: E402
from speech_to_text import SAMPLE_RATE, recognize_waveform  # noqa: E402

# shorter than this is almost certainly an accidental click, not real speech
MIN_UTTERANCE_SECONDS = 0.2

_stream: Optional[sd.InputStream] = None
_frames: list = []
_recording = False


def is_recording() -> bool:
    return _recording


def start_recording(on_level: Optional[Callable[[float], None]] = None) -> None:
    """Start capturing from the default microphone.

    `on_level` is invoked from sounddevice's own audio callback thread (not
    the caller's thread) with each block's amplitude scaled to roughly
    [0, 1], meant for a live "listening" waveform — a Tk caller must marshal
    it back to the UI thread itself (e.g. via `windows.after`).
    """
    global _stream, _frames, _recording
    if _recording:
        return

    _frames = []
    _recording = True

    def _callback(indata, frames, time_info, status):
        _frames.append(indata.copy())
        if on_level is not None:
            on_level(min(1.0, float(np.abs(indata).mean()) * 8))

    _stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_callback)
    _stream.start()


def stop_recording() -> str:
    """Stop capturing and run the self-trained STT model on what was just
    recorded.

    Returns "" if nothing usable was recorded (e.g. an accidental
    click-and-release), or speech_to_text.recognize_waveform()'s own
    placeholder message if no checkpoint has been trained yet.
    """
    global _stream, _recording
    _recording = False
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

    if not _frames:
        return ""
    waveform = np.concatenate(_frames, axis=0).flatten()
    if waveform.size < SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
        return ""

    return recognize_waveform(waveform)


# 「記住對話內容」原本試著把上一輪使用者訊息當純文字接在這一輪前面一起送進
# 模型，但實測發現這樣做會讓 sinco 的回覆被「拉走」——例如 "你是誰｜再說一次"
# 會答成「我是 sinco...」，"hi｜再見" 會答出兩句混在一起的亂碼——不管新問題是
# 什麼，都被字串裡先出現的那段內容主導。根因跟 chat checkpoint 的 max_len=40
# 有關（chats.encode() 會直接砍掉超過 39 字元的部分，塞歷史的空間本來就很
# 少），但更根本的問題是模型從沒被訓練過要怎麼解讀「歷史｜這句話」這種格式，
# 對它來說只是一段沒看過的亂碼輸入，會直接觸發最接近的已訓練樣式、而不是真的
# 理解上下文。**所以目前只保留歷史紀錄本身（供之後真的要重訓多輪對話用），
# 不會把歷史文字塞進模型的輸入**——這是唯一能在不重訓的前提下，做到「記得住」
# 又不會讓現有回覆變差的做法。想要 sinco 真的懂上下文，需要重新設計多輪對話
# 的訓練資料並重訓（見 chats.py 的討論）。
MAX_HISTORY_TURNS = 20


class Conversation:
    def __init__(self, windows: tk.Tk, chat_display: tk.Text, on_ask_start: Optional[Callable[[], None]] = None):
        self.windows = windows
        self.chat_display = chat_display
        self.on_ask_start = on_ask_start
        self.busy = False
        self.history: list[tuple[str, str]] = []  # [(使用者訊息, sinco回覆), ...]，只保留最近 MAX_HISTORY_TURNS 輪

        # /model（CLAUDE.md 需求 #01）：手動覆蓋 chats.smart_reply_traced() 的
        # chat/code 自動判斷，"auto" = 維持既有行為。目前 tranning/chats.py 的
        # smart_reply_traced() 還沒有 force_mode 參數（second 分支有一版 Bayesian
        # MC-Dropout 升級才加了這個參數，main 尚未合併那份升級），所以這裡先只是
        # 記錄使用者選了哪個模式、讓 /model 指令本身能正常運作，實際還不會影響
        # 路由——跟這個專案其他「先接上 UI，模型端功能之後再補」的慣例一致。
        self.force_mode = "auto"
        # /character（需求 #02）：目前對話用的人格 checkpoint，預設 sinco。
        # character_browser.py 選好角色後會呼叫 set_persona() 改這兩個值。
        self.out_dir = DEFAULT_OUT_DIR
        self.persona = "sinco"

    def set_persona(self, out_dir, name: str):
        self.out_dir = out_dir
        self.persona = name

    def ask(self, header: str, model_input: str, record_as: Optional[str] = None,
            header_tag: str = "user_bubble"):
        if self.busy:
            return
        self.busy = True

        chat_display = self.chat_display
        chat_display.configure(state="normal")
        chat_display.insert(tk.END, f"{header}\n", header_tag)
        chat_display.insert(tk.END, "\n")
        # 用 tag 記錄「思考中」佔位文字的範圍，而不是存一個固定的 index 字串——
        # 實測發現 chat_display.index(tk.END) 抓到的位置，跟接下來 insert(tk.END, ...)
        # 實際落下去的位置會差一行（Tk Text 內部永遠多一個看不見的結尾空行），拿
        # 那個位置去 delete 會少刪一行，「思考中...」殘留、新內容直接接在後面。
        # tag_ranges() 會自動追蹤標記過的文字範圍，不受這個位置誤差影響。
        chat_display.insert(tk.END, f"{self.persona} 思考中...\n\n", "pending")
        chat_display.configure(state="disabled")
        chat_display.see(tk.END)
        if self.on_ask_start is not None:
            self.on_ask_start()

        def worker():
            trace, reply = smart_reply_traced(model_input, out_dir=self.out_dir)

            def show_result():
                chat_display.configure(state="normal")
                start, end = chat_display.tag_ranges("pending")
                chat_display.delete(start, end)
                # 用 tk.INSERT 這個「活」標記依序插入三段各自不同 tag 的文字——
                # 每次 insert(tk.INSERT, ...) 都會插在標記目前所在位置、再把標記
                # 推到新插入內容的後面，才能讓三段文字照順序接起來；如果直接對
                # start 這個「當下那一刻」的固定索引重複 insert，後面插入的內容
                # 會被塞到前面插入內容的前面（索引沒有跟著移動）。
                chat_display.mark_set(tk.INSERT, start)
                chat_display.insert(tk.INSERT, f"{self.persona}\n", "ai_label")
                chat_display.insert(tk.INSERT, f"{trace}\n", "trace")
                chat_display.insert(tk.INSERT, f"{reply}\n\n", "ai_text")
                chat_display.configure(state="disabled")
                chat_display.see(tk.END)
                self.busy = False
                if record_as is not None:
                    self.history.append((record_as, reply))
                    del self.history[:-MAX_HISTORY_TURNS]

            self.windows.after(0, show_result)

        threading.Thread(target=worker, daemon=True).start()

    def send_message(self, user_text: str):
        self.ask(user_text, user_text, record_as=user_text)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open File",
            filetypes=[("All files", "*.*")],
        )
        if not path:
            return

        content = read_as_chat_content(Path(path))
        self.ask(f"--- {os.path.basename(path)} ---\n{content}", content, header_tag="system")
