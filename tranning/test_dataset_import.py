"""Unit tests for dataset_import.py — building chats.py/OCR.py/
speech_to_text.py manifests (or road_sign/image_classifier image folders)
straight from raw files (paired .txt, image+.txt, audio+.txt, video),
instead of requiring a hand-written JSON manifest up front (CLAUDE.md
需求 #04).
"""

import json
import wave

import numpy as np
import pytest

from dataset_import import (
    collect_pairs,
    collect_sidecar,
    extract_video_frames,
    is_16bit_pcm_wav,
    split_prompt_reply,
    write_manifest,
)


def _write_wav(path, sampwidth=2, n_frames=1600):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * n_frames * sampwidth)


def test_split_prompt_reply_blank_line_format():
    assert split_prompt_reply("你好嗎\n\n我很好，謝謝關心") == ("你好嗎", "我很好，謝謝關心")


def test_split_prompt_reply_two_line_format():
    assert split_prompt_reply("hello\nhi there") == ("hello", "hi there")


def test_split_prompt_reply_rejects_single_line():
    assert split_prompt_reply("只有一行沒有下文") is None
    assert split_prompt_reply("   ") is None


def test_collect_pairs_reads_txt_folder_and_skips_bad_files(tmp_path):
    (tmp_path / "a.txt").write_text("你好嗎\n\n我很好", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\nhi there", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("只有一行", encoding="utf-8")

    entries, skipped = collect_pairs(tmp_path)

    assert {"prompt": "你好嗎", "reply": "我很好"} in entries
    assert {"prompt": "hello", "reply": "hi there"} in entries
    assert len(entries) == 2
    assert len(skipped) == 1


def test_collect_sidecar_image_pairs_matching_txt(tmp_path):
    (tmp_path / "sign01.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "sign01.txt").write_text("停止", encoding="utf-8")
    (tmp_path / "sign02.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # no sidecar -> skipped

    entries, skipped = collect_sidecar(tmp_path, "image")

    assert entries == [{"image": "sign01.png", "text": "停止"}]
    assert len(skipped) == 1
    assert "sign02.png" in skipped[0]


def test_collect_sidecar_audio_rejects_non_16bit_wav(tmp_path):
    _write_wav(tmp_path / "good.wav", sampwidth=2)
    (tmp_path / "good.txt").write_text("你好", encoding="utf-8")
    _write_wav(tmp_path / "bad.wav", sampwidth=1)
    (tmp_path / "bad.txt").write_text("哈囉", encoding="utf-8")

    entries, skipped = collect_sidecar(tmp_path, "audio")

    assert entries == [{"audio": "good.wav", "text": "你好"}]
    assert len(skipped) == 1
    assert "bad.wav" in skipped[0]


def test_is_16bit_pcm_wav(tmp_path):
    path_16 = tmp_path / "sixteen.wav"
    _write_wav(path_16, sampwidth=2)
    assert is_16bit_pcm_wav(path_16) is True

    path_8 = tmp_path / "eight.wav"
    _write_wav(path_8, sampwidth=1)
    assert is_16bit_pcm_wav(path_8) is False

    assert is_16bit_pcm_wav(tmp_path / "missing.wav") is False


def test_write_manifest_single_file_when_no_val_ratio(tmp_path):
    out = tmp_path / "pairs.json"
    entries = [{"prompt": "hi", "reply": "hello"}]

    summary = write_manifest(entries, out, val_ratio=0.0)

    assert summary == {"total": 1, "train_path": str(out), "val_path": None,
                        "train_count": 1, "val_count": 0}
    assert json.loads(out.read_text(encoding="utf-8")) == entries


def test_write_manifest_falls_back_to_single_file_when_too_few_entries(tmp_path):
    out = tmp_path / "pairs.json"
    entries = [{"prompt": "hi", "reply": "hello"}]

    summary = write_manifest(entries, out, val_ratio=0.3)

    assert summary["val_path"] is None
    assert summary["train_count"] == 1


def test_write_manifest_splits_train_val(tmp_path):
    out = tmp_path / "pairs.json"
    entries = [{"prompt": f"p{i}", "reply": f"r{i}"} for i in range(10)]

    summary = write_manifest(entries, out, val_ratio=0.2, seed=42)

    train_path = tmp_path / "pairs_train.json"
    val_path = tmp_path / "pairs_val.json"
    assert summary["train_path"] == str(train_path)
    assert summary["val_path"] == str(val_path)
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    val_data = json.loads(val_path.read_text(encoding="utf-8"))
    assert len(train_data) == 8
    assert len(val_data) == 2
    # no overlap between the two splits
    assert {e["prompt"] for e in train_data}.isdisjoint({e["prompt"] for e in val_data})


def test_write_manifest_merges_and_dedupes_existing_file(tmp_path):
    out = tmp_path / "pairs.json"
    out.write_text(json.dumps([{"prompt": "old", "reply": "既有資料"}], ensure_ascii=False), encoding="utf-8")

    write_manifest([{"prompt": "new", "reply": "新資料"}], out, val_ratio=0.0)
    # re-importing the exact same new folder contents shouldn't duplicate it
    write_manifest([{"prompt": "new", "reply": "新資料"}], out, val_ratio=0.0)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.count({"prompt": "new", "reply": "新資料"}) == 1
    assert {"prompt": "old", "reply": "既有資料"} in data


def test_extract_video_frames_writes_per_class_images(tmp_path):
    cv2 = pytest.importorskip("cv2")

    source = tmp_path / "source"
    (source / "stop_sign").mkdir(parents=True)
    video_path = source / "stop_sign" / "clip.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (32, 32))
    for _ in range(20):  # 2 秒份的影格 @ 10fps
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    out = tmp_path / "frames"
    counts = extract_video_frames(source, out, interval=1.0)

    assert counts["stop_sign"] >= 1
    saved = list((out / "stop_sign").glob("*.jpg"))
    assert len(saved) == counts["stop_sign"]


def test_extract_video_frames_writes_files_for_non_ascii_class_names(tmp_path):
    """迴歸測試：cv2.imwrite() 在 Windows 上對非 ASCII 路徑（這個專案的類別
    名稱常常是中文，例如「停止」）會靜默寫入失敗——counts 回報有存到，
    但資料夾底下其實是空的。這裡直接斷言檔案真的落在磁碟上，不能只看
    counts 的回傳值。
    """
    cv2 = pytest.importorskip("cv2")

    source = tmp_path / "來源"
    class_dir = source / "停止"
    class_dir.mkdir(parents=True)
    video_path = class_dir / "clip.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (32, 32))
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    out = tmp_path / "輸出"
    counts = extract_video_frames(source, out, interval=1.0)

    assert counts["停止"] >= 1
    saved = list((out / "停止").glob("*.jpg"))
    assert len(saved) == counts["停止"] > 0  # 不能只是 counts>0 但磁碟上其實是空的


def test_extract_video_frames_empty_source_returns_empty_counts(tmp_path):
    pytest.importorskip("cv2")
    source = tmp_path / "empty"
    source.mkdir()

    counts = extract_video_frames(source, tmp_path / "out", interval=1.0)

    assert counts == {}
