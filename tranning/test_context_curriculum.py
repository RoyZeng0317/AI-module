import hashlib, json
from pathlib import Path

import pytest

from context_curriculum import make_curriculum
from context_dialogues import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_curriculum_reproducible_and_preserves_source(tmp_path):
    source = ROOT / "data/context_dialogues/train.json"
    before = source.read_bytes()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    report = make_curriculum(source, a, ROOT / "tranning/gpt_chat_runs", count=12)
    make_curriculum(source, b, ROOT / "tranning/gpt_chat_runs", count=12)
    assert a.read_bytes() == b.read_bytes()
    assert source.read_bytes() == before
    assert report["output_sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    rows = json.loads(a.read_text(encoding="utf-8"))
    assert len(rows) == len(json.loads(before)) + 12
    assert all(row["task"] == "span_copy" for row in rows[-12:])
    with pytest.raises(ValueError):
        make_curriculum(source, a, ROOT / "tranning/gpt_chat_runs", count=12)


def test_dataset_hash_is_actual_written_bytes(tmp_path):
    folder = tmp_path / "dataset"
    manifest = make_dataset(folder)
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == hashlib.sha256((folder / name).read_bytes()).hexdigest()
