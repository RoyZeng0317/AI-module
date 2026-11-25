"""speech_to_text.py 的 --recognize CLI 入口測試：只測檔案檢查，不訓練、不讀權重。
recognize() 一律換成假函式，所以不會載入或執行任何模型。
執行：python -m pytest tranning/test_speech_cli.py
"""
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_to_text as stt  # noqa: E402


@pytest.fixture
def fake_recognize(monkeypatch):
    calls = []

    def fake(audio_path, out_dir=stt.DEFAULT_OUT_DIR):
        calls.append((audio_path, out_dir))
        return "假辨識結果"

    monkeypatch.setattr(stt, "recognize", fake)
    return calls


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["speech_to_text.py", *map(str, args)])
    stt.main()


@pytest.mark.parametrize("make", ["missing", "folder"])
def test_missing_file_or_folder_exits_2_without_traceback(tmp_path, monkeypatch, capsys, fake_recognize, make):
    target = tmp_path / "錄音檔完整路徑.wav"
    if make == "folder":
        target = tmp_path / "一個 資料夾"
        target.mkdir()
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "--recognize", target)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(target) in err
    assert "real WAV file" in err
    assert ("is a folder" in err) == (make == "folder")
    assert "Traceback" not in err
    assert fake_recognize == []  # 檔案不對就不呼叫模型


def test_real_wav_with_chinese_and_spaces_reaches_recognize(tmp_path, monkeypatch, capsys, fake_recognize):
    folder = tmp_path / "我的 錄音"
    folder.mkdir()
    wav = folder / "測試 檔案.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    run_cli(monkeypatch, "--recognize", wav)
    assert fake_recognize == [(wav, stt.DEFAULT_OUT_DIR)]
    assert capsys.readouterr().out.strip() == "假辨識結果"  # 回傳值原樣印出


def test_recognize_errors_are_not_swallowed(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"not really a wav")

    def broken(audio_path, out_dir=stt.DEFAULT_OUT_DIR):
        raise RuntimeError("model failed")

    monkeypatch.setattr(stt, "recognize", broken)
    with pytest.raises(RuntimeError, match="model failed"):
        run_cli(monkeypatch, "--recognize", wav)
