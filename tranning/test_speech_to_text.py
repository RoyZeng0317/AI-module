"""Pipeline smoke test for speech_to_text.py.

Does NOT claim any recognition accuracy — there is no real speech dataset
yet. Only proves the CRNN+CTC training loop (WAV loading, spectrogram
extraction, dataset, CTC loss, early stopping, checkpoint + charset/config
output) and both recognize() entry points run end-to-end without crashing,
using a handful of tiny synthetic sine-wave "utterances" (each text label
mapped to its own fixed tone+duration so the mapping is at least
deterministic, not real speech).
"""

import json
import wave
from pathlib import Path

import numpy as np

from speech_to_text import SAMPLE_RATE, recognize, recognize_waveform, train

TEXTS = ["12", "AB", "9X"]
TONES = {"12": 220.0, "AB": 440.0, "9X": 660.0}


def _make_tone(text: str, duration: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * TONES[text] * t)).astype(np.float32)


def _write_wav(path: Path, waveform: np.ndarray):
    pcm = (waveform * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _make_manifest(audio_dir: Path, prefix: str, repeats: int) -> list[dict]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for text in TEXTS:
        for i in range(repeats):
            name = f"{prefix}_{text}_{i}.wav"
            _write_wav(audio_dir / name, _make_tone(text))
            records.append({"audio": name, "text": text})
    return records


def test_training_and_recognize_run_end_to_end(tmp_path):
    audio_dir = tmp_path / "audio"
    train_records = _make_manifest(audio_dir, "train", repeats=6)
    val_records = _make_manifest(audio_dir, "val", repeats=2)

    train_manifest = tmp_path / "train.json"
    val_manifest = tmp_path / "val.json"
    train_manifest.write_text(json.dumps(train_records), encoding="utf-8")
    val_manifest.write_text(json.dumps(val_records), encoding="utf-8")

    out_dir = tmp_path / "runs"
    model, charset, history = train(
        train_manifest=train_manifest, val_manifest=val_manifest, audio_root=audio_dir,
        out_dir=out_dir, epochs=2, batch_size=4, lr=1e-3, patience=5,
        max_frames=40, hidden_size=32, dropout=0.1, weight_decay=1e-4,
    )

    assert set(charset) == set("".join(TEXTS))
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "charset.json").read_text(encoding="utf-8")) == charset
    assert all(h["val_cer"] >= 0.0 for h in history)

    sample_audio = audio_dir / f"train_{TEXTS[0]}_0.wav"
    result = recognize(sample_audio, out_dir=out_dir)
    assert isinstance(result, str)

    live_result = recognize_waveform(_make_tone(TEXTS[0]), out_dir=out_dir)
    assert isinstance(live_result, str)


def test_recognize_without_checkpoint_returns_placeholder(tmp_path):
    message = recognize_waveform(np.zeros(SAMPLE_RATE, dtype=np.float32), out_dir=tmp_path / "no_checkpoint_here")
    assert "尚未訓練" in message
