"""Tests for syllable_tts.py with synthetic "recordings" (sine tones with
silence around them) -- checks the cutting/gluing logic, not voice quality."""

import numpy as np

import syllable_tts as tts
from speech_to_text import SAMPLE_RATE


def _clip(freq: float, seconds: float = 0.3, silence: float = 0.2) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    gap = np.zeros(int(SAMPLE_RATE * silence), dtype=np.float32)
    return np.concatenate([gap, (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), gap])


def _voice_dir(tmp_path, names):
    d = tmp_path / "voice"
    for i, name in enumerate(names):
        tts.save_wav(_clip(200 + 50 * i), d / f"{name}.wav")
    return d


def test_recording_list_covers_every_base_syllable_first():
    order = tts.recording_list(["你好", "你好嗎", "好的"])
    bases = [tts.strip_tone(s) for s in order]
    first_pass = bases[: len(set(bases))]
    assert len(set(first_pass)) == len(first_pass)      # one tone per syllable before any second tone
    assert order[0] == "hao3"                          # most frequent first


def test_synthesize_trims_glues_and_pauses(tmp_path):
    d = _voice_dir(tmp_path, ["ni3", "hao3"])
    wave_ = tts.synthesize("你好", d)
    # two 0.3 s syllables, silence trimmed, crossfaded, 50 ms pad each side
    expected = 2 * 0.3 * SAMPLE_RATE - tts.CROSSFADE + 0.1 * SAMPLE_RATE
    assert abs(len(wave_) - expected) < 0.03 * SAMPLE_RATE
    with_pause = tts.synthesize("你，好", d)
    assert len(with_pause) > len(wave_) + 0.2 * SAMPLE_RATE


def test_missing_tone_falls_back_and_missing_syllable_is_reported(tmp_path):
    d = _voice_dir(tmp_path, ["hao3"])
    voice = tts.SyllableVoice(d)
    assert voice.lookup("hao4") is voice.units["hao3"]
    _, missing = voice.synthesize("你好")
    assert missing == ["ni3"]


def test_coverage_and_placeholder(tmp_path):
    assert tts.synthesize("你好", tmp_path / "empty") == tts.NOT_RECORDED
    d = _voice_dir(tmp_path, ["ni3"])
    cov = tts.coverage(["你好"], d)
    assert cov["exact"] == 0.5 and cov["missing"] == ["hao3"]


def test_import_dir_only_takes_syllable_named_wavs(tmp_path):
    src = tmp_path / "src"
    tts.save_wav(_clip(300), src / "Ni3.wav")
    tts.save_wav(_clip(300), src / "hello.wav")
    assert tts.import_dir(src, tmp_path / "voice") == ["ni3"]
    # the cached voice reloads when new recordings appear
    assert not isinstance(tts.synthesize("你", tmp_path / "voice"), str)
