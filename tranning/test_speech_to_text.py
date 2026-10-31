"""Pipeline smoke test for speech_to_text.py (syllable CTC recognizer).

Does NOT claim any recognition accuracy -- there is no real speech dataset
in this repo. It only proves the training loop (WAV loading, log-mel
features, syllable targets, length-bucketed batches, CTC loss with real
lengths, early stopping, checkpoint + units/config output) and the
recognize() entry points run end-to-end, using tiny synthetic tone clips:
each Chinese label gets its own fixed tone pattern (one tone per syllable),
so the mapping is deterministic but is not real speech.
"""

import json
import wave
from pathlib import Path

import numpy as np

from speech_to_text import (SAMPLE_RATE, recognize, recognize_waveform, remove_fillers, subsampled_length,
                            train, waveform_to_logmel)

TEXTS = ["你好", "謝謝", "再見"]
TONES = {"你好": (220.0, 330.0), "謝謝": (440.0, 440.0), "再見": (550.0, 660.0)}


def _make_tone(text: str, seconds_per_syllable: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds_per_syllable, int(SAMPLE_RATE * seconds_per_syllable), endpoint=False)
    return np.concatenate([0.5 * np.sin(2 * np.pi * f * t) for f in TONES[text]]).astype(np.float32)


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
    for k, text in enumerate(TEXTS):
        for i in range(repeats):
            name = f"{prefix}_{k}_{i}.wav"
            _write_wav(audio_dir / name, _make_tone(text))
            records.append({"audio": name, "text": text})
    return records


def test_logmel_keeps_the_whole_clip():
    # the old model cropped every clip to its first 1.6 s; features must now cover all of it
    feats = waveform_to_logmel(np.zeros(SAMPLE_RATE * 5, dtype=np.float32))
    assert feats.shape[1] == 80
    assert feats.shape[0] >= 5 * 100 - 2              # one frame per 10 ms
    assert subsampled_length(feats.shape[0]) >= 120   # ~40 ms per output frame


def test_training_and_recognize_run_end_to_end(tmp_path):
    audio_dir = tmp_path / "audio"
    train_records = _make_manifest(audio_dir, "train", repeats=6)
    val_records = _make_manifest(audio_dir, "val", repeats=2)
    # a clip longer than max_seconds is skipped, not cropped
    _write_wav(audio_dir / "long.wav", np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
    train_records.append({"audio": "long.wav", "text": "很長的句子"})

    train_manifest = tmp_path / "train.json"
    val_manifest = tmp_path / "val.json"
    train_manifest.write_text(json.dumps(train_records, ensure_ascii=False), encoding="utf-8")
    val_manifest.write_text(json.dumps(val_records, ensure_ascii=False), encoding="utf-8")

    out_dir = tmp_path / "runs"
    model, units, history = train(
        train_manifest=train_manifest, val_manifest=val_manifest, audio_root=audio_dir,
        out_dir=out_dir, epochs=2, batch_size=4, lr=1e-3, patience=5,
        max_seconds=2.0, hidden_size=32, num_layers=1, dropout=0.1, weight_decay=1e-4, device="cpu",
    )

    assert set(units) == {"ni", "hao", "xie", "zai", "jian"}   # toneless syllables, not characters
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "units.json").read_text(encoding="utf-8")) == units
    assert json.loads((out_dir / "config.json").read_text(encoding="utf-8"))["kind"] == "syllable_ctc"
    assert all(h["val_ser"] >= 0.0 for h in history)

    result = recognize(audio_dir / "train_0_0.wav", out_dir=out_dir)
    assert isinstance(result, str) and "尚未訓練" not in result

    live_result = recognize_waveform(_make_tone(TEXTS[0]), out_dir=out_dir, lm_dir=tmp_path / "no_lm")
    assert isinstance(live_result, str)


def test_recognize_without_checkpoint_returns_placeholder(tmp_path):
    message = recognize_waveform(np.zeros(SAMPLE_RATE, dtype=np.float32), out_dir=tmp_path / "no_checkpoint_here")
    assert "尚未訓練" in message


def test_remove_fillers_strips_standalone_chinese_interjections():
    assert remove_fillers("嗯我覺得啊這樣可以") == "我覺得這樣可以"
    assert remove_fillers("呃就是那個東西") == "就是那個東西"


def test_remove_fillers_strips_repeated_filler_runs():
    assert remove_fillers("嗯嗯嗯你說得對") == "你說得對"


def test_remove_fillers_strips_english_filler_words():
    assert remove_fillers("um I think uh this works") == "I think this works"


def test_remove_fillers_leaves_clean_text_unchanged():
    assert remove_fillers("我覺得這樣可以") == "我覺得這樣可以"
    assert remove_fillers("this works fine") == "this works fine"
