"""Voice conversation glue: mic capture -> self-built STT (speech_to_text.py)
-> text ready to hand to the chat model. No cloud speech API (Rule 06) —
recognize_waveform() below is the from-scratch CRNN+CTC model trained in
tranning/speech_to_text.py, not a third-party speech service.

Kept free of any tkinter import on purpose: this module only knows about
audio in/out, not about how a caller chooses to render it (waveform canvas,
plain text, or anything else) — home_screen.py owns that part.
"""

import os
import sys
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tranning"))

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
