"""Tests for aishell_import.py on a tiny fake data_aishell tree (no real
AISHELL-1 data in the repo): split mapping, space removal, skip counting,
.tar.gz handling (message / safe --extract), and that the train manifest is
accepted by speech_to_text._usable_records."""

import io
import json
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest

import aishell_import
from speech_to_text import SAMPLE_RATE, _usable_records

UTTS = {
    "train": [("S0002", "BAC009S0002W0122", "而 对 楼市 成交"), ("S0002", "BAC009S0002W0123", "限 购")],
    "dev": [("S0724", "BAC009S0724W0121", "今天 天气 很 好")],
    "test": [("S0764", "BAC009S0764W0121", "你好 世界")],
}


def _wav_bytes(seconds: float = 1.5) -> bytes:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _make_tree(tmp_path: Path, write_wavs: bool = True) -> Path:
    root = tmp_path / "data_aishell"
    (root / "transcript").mkdir(parents=True)
    lines = [f"{utt} {text}" for split in UTTS.values() for _, utt, text in split]
    lines.append("BAC009S0002W9999 这 条 没有 音频")          # transcript without wav
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    (root / "wav").mkdir()
    if write_wavs:
        for split, items in UTTS.items():
            for spk, utt, _ in items:
                d = root / "wav" / split / spk
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{utt}.wav").write_bytes(_wav_bytes())
        # wav without transcript
        (root / "wav" / "train" / "S0002" / "BAC009S0002W8888.wav").write_bytes(_wav_bytes())
    return root


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifests_split_mapping_and_skips(tmp_path, capsys):
    root = _make_tree(tmp_path)
    out = tmp_path / "out"
    summary = aishell_import.main(["--aishell-dir", str(tmp_path), "--out-dir", str(out)])  # parent dir accepted

    train, val, test = _load(out / "train.json"), _load(out / "val.json"), _load(out / "test.json")
    assert train == [
        {"audio": "wav/train/S0002/BAC009S0002W0122.wav", "text": "而对楼市成交"},
        {"audio": "wav/train/S0002/BAC009S0002W0123.wav", "text": "限购"},
    ]
    assert val == [{"audio": "wav/dev/S0724/BAC009S0724W0121.wav", "text": "今天天气很好"}]
    assert test == [{"audio": "wav/test/S0764/BAC009S0764W0121.wav", "text": "你好世界"}]
    assert summary["no_transcript"] == 1 and summary["no_wav"] == 1
    assert summary["splits"] == {"train": 2, "dev": 1, "test": 1}
    for rec in train + val + test:
        assert (root / rec["audio"]).is_file()      # relative to data_aishell, wavs not copied
    assert not list(out.rglob("*.wav"))

    printed = capsys.readouterr().out
    assert f'--audio-root "{root}"' in printed
    assert "Simplified" in printed and "--extra-text" in printed

    kept = _usable_records(train, root, max_seconds=12.0)
    assert len(kept) == 2 and all(r["syllables"] for r in kept)


def test_max_per_split(tmp_path):
    root = _make_tree(tmp_path)
    out = tmp_path / "out"
    aishell_import.main(["--aishell-dir", str(root), "--out-dir", str(out), "--max-per-split", "1"])
    assert len(_load(out / "train.json")) == 1
    assert len(_load(out / "val.json")) == 1


def _make_speaker_archive(root: Path, split: str, spk: str, utts: list[str]):
    arc = root / "wav" / f"{spk}.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:
        for utt in utts:
            data = _wav_bytes()
            info = tarfile.TarInfo(f"{split}/{spk}/{utt}.wav")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_archives_without_wavs_give_clear_message(tmp_path):
    root = _make_tree(tmp_path, write_wavs=False)
    _make_speaker_archive(root, "train", "S0002", ["BAC009S0002W0122"])
    with pytest.raises(SystemExit) as exc:
        aishell_import.main(["--aishell-dir", str(root), "--out-dir", str(tmp_path / "out")])
    assert "--extract" in str(exc.value) and "tar" in str(exc.value)
    assert not (root / "wav" / "train").exists()      # nothing extracted without --extract


def test_extract_flag(tmp_path):
    root = _make_tree(tmp_path, write_wavs=False)
    _make_speaker_archive(root, "train", "S0002", ["BAC009S0002W0122", "BAC009S0002W0123"])
    _make_speaker_archive(root, "dev", "S0724", ["BAC009S0724W0121"])
    out = tmp_path / "out"
    summary = aishell_import.main(["--aishell-dir", str(root), "--out-dir", str(out), "--extract"])
    assert summary["splits"] == {"train": 2, "dev": 1, "test": 0}
    assert (root / "wav" / "train" / "S0002" / "BAC009S0002W0122.wav").is_file()


@pytest.mark.parametrize("bad_name", ["../evil.wav", "/abs/evil.wav", "train/../../evil.wav"])
def test_extract_rejects_unsafe_members(tmp_path, bad_name):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    with tarfile.open(wav_dir / "S9999.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo(bad_name)
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(ValueError):
        aishell_import.extract_archives(wav_dir)
    assert not (tmp_path / "evil.wav").exists()


def test_extract_rejects_symlink(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    with tarfile.open(wav_dir / "S9999.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("train/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError):
        aishell_import.extract_archives(wav_dir)


def test_to_traditional(tmp_path):
    pytest.importorskip("opencc")
    root = _make_tree(tmp_path)
    out = tmp_path / "out"
    aishell_import.main(["--aishell-dir", str(root), "--out-dir", str(out), "--to-traditional"])
    train = _load(out / "train.json")
    assert train[0]["text"] == "而對樓市成交"
    assert train[1]["text"] == "限購"
    assert _load(out / "val.json")[0]["text"] == "今天天氣很好"
