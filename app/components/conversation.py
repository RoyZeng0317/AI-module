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
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from chats import smart_reply_traced
from function import read_as_chat_content
from speech_to_text import SAMPLE_RATE, recognize_waveform

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


class Conversation:
    def __init__(self, windows: tk.Tk, chat_display: tk.Text, on_ask_start: Optional[Callable[[], None]] = None):
        self.windows = windows
        self.chat_display = chat_display
        self.on_ask_start = on_ask_start
        self.busy = False

    def ask(self, header: str, model_input: str):
        if self.busy:
            return
        self.busy = True

        chat_display = self.chat_display
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
        if self.on_ask_start is not None:
            self.on_ask_start()

        def worker():
            trace, reply = smart_reply_traced(model_input)

            def show_result():
                chat_display.configure(state="normal")
                start, end = chat_display.tag_ranges("pending")
                chat_display.delete(start, end)
                chat_display.insert(start, f"[思考過程：{trace}]\nAI: {reply}\n\n")
                chat_display.configure(state="disabled")
                chat_display.see(tk.END)
                self.busy = False

            self.windows.after(0, show_result)

        threading.Thread(target=worker, daemon=True).start()

    def send_message(self, user_text: str):
        self.ask(f"You: {user_text}", user_text)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open File",
            filetypes=[("All files", "*.*")],
        )
        if not path:
            return

        content = read_as_chat_content(Path(path))
        self.ask(f"--- {os.path.basename(path)} ---\n{content}", content)
