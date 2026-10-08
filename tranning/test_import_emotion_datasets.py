"""Source-role, split and integrity regressions; no network or model calls."""
import json
import hashlib
from pathlib import Path

import pytest

from import_emotion_datasets import (acceptable_pair, cped_records, dailydialog_pairs,
                                    download_sources, empathetic_pairs, esconv_pairs,
                                    goemotions_records, hash_split, isolate, pair)


def test_esconv_merges_roles_and_excludes_suggestion_blocks():
    conversation = {"emotion_type": "sadness", "dialog": [
        {"speaker": "seeker", "content": "I have felt very lonely all week", "annotation": {}},
        {"speaker": "seeker", "content": "and miss my friends", "annotation": {}},
        {"speaker": "supporter", "content": "That sounds difficult.", "annotation": {"strategy": "Reflection of feelings"}},
        {"speaker": "supporter", "content": "Would you like to tell me more?", "annotation": {"strategy": "Question"}},
        {"speaker": "seeker", "content": "I am struggling with decisions about work", "annotation": {}},
        {"speaker": "supporter", "content": "You should quit your job.", "annotation": {"strategy": "Providing Suggestions"}},
    ]}
    rows = esconv_pairs([conversation])
    assert len(rows) == 1
    assert rows[0]["prompt"] == "I have felt very lonely all week and miss my friends"
    assert rows[0]["reply"] == "That sounds difficult. Would you like to tell me more?"
    assert esconv_pairs([conversation])[0]["split"] == rows[0]["split"]


def test_empathetic_only_first_listener_response_and_decodes_commas():
    rows = [dict(conv_id="c1", utterance_idx="1", speaker_idx="9", context="sad", utterance="I feel sad_comma_ today"),
            dict(conv_id="c1", utterance_idx="2", speaker_idx="7", context="sad", utterance="What happened?"),
            dict(conv_id="c1", utterance_idx="3", speaker_idx="9", context="sad", utterance="I lost a friend"),
            dict(conv_id="c1", utterance_idx="4", speaker_idx="7", context="sad", utterance="Tell me more")]
    result = empathetic_pairs(rows[::-1], "val")
    assert len(result) == 1
    assert result[0]["prompt"] == "I feel sad, today"
    assert result[0]["reply"] == "What happened?" and result[0]["split"] == "val"


def test_daily_user_system_direction_and_emotion_filter():
    dialogue = {"data_split": "validation", "dialogue_id": "1", "turns": [
        dict(speaker="system", utterance="Please tell me more", emotion="no emotion", utt_idx=0),
        dict(speaker="user", utterance="I feel sad", emotion="sadness", utt_idx=1),
        dict(speaker="system", utterance="What happened?", emotion="no emotion", utt_idx=2),
        dict(speaker="user", utterance="Nothing special", emotion="no emotion", utt_idx=3),
        dict(speaker="system", utterance="How rude!", emotion="anger", utt_idx=4)]}
    rows = dailydialog_pairs([dialogue])
    assert len(rows) == 1 and rows[0]["split"] == "val" and rows[0]["reply"] == "What happened?"


def test_isolation_prefers_held_out_text_and_keeps_distinct_training_replies():
    rows = [pair("HELLO THERE", "train leakage", "x", "1", "train"),
            pair("hello there", "held-out reply", "x", "2", "test"),
            pair("another question", "one answer", "x", "3", "train"),
            pair("another question", "one answer", "x", "3", "train"),
            pair("another question", "another answer", "x", "3", "train")]
    output, stats = isolate(rows, "prompt")
    assert len(output["test"]) == 1 and len(output["train"]) == 2
    assert stats == {"cross_split_texts_removed": 1, "duplicates_removed": 1}
    assert not {r["prompt"].casefold() for r in output["test"]} & {r["prompt"].casefold() for r in output["train"]}


def test_isolation_rejects_cross_split_conversation():
    with pytest.raises(ValueError, match="crosses"):
        isolate([pair("first question", "first reply", "x", "same", "train"),
                 pair("other question", "other reply", "x", "same", "test")], "prompt")


def test_classification_preserves_label_spaces_and_multiple_goemotions_labels():
    rows = cped_records([dict(Utterance="简体话", Emotion="worried", Sentiment="negative",
                              DA="comfort", TV_ID="1", Dialogue_ID="2", Utterance_ID="3", Speaker="x")],
                        "train", lambda text: text.replace("简体话", "簡體話"))
    assert rows[0]["text"] == "簡體話" and rows[0]["labels"] == ["worried"]
    assert rows[0]["label_space"] == "cped_emotion" and "reply" not in rows[0]
    go = goemotions_records([["mixed feelings", "0,2", "abc"]], ["sadness", "neutral", "joy"], "test")
    assert go[0]["labels"] == ["sadness", "joy"] and go[0]["label_space"] == "goemotions"


def test_length_limit_rejects_instead_of_truncating():
    row = pair("long prompt", "long answer", "x", "1", "train")
    assert not acceptable_pair(row, 10)
    assert row["reply"] == "long answer"
    assert acceptable_pair(row, 480)


def test_download_reuses_only_checksum_verified_cache(tmp_path, monkeypatch):
    data = b"verified source"
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "x.json").write_bytes(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"filename": "x.json", "sha256": hashlib.sha256(data).hexdigest(),
                                             "bytes": len(data), "url": "https://example.com/x"}]}))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("valid cache must not download"))
    download_sources(manifest, raw)


def test_hash_split_is_deterministic_and_never_depends_on_row_order():
    assert hash_split("conversation") == hash_split("conversation")
    assert {hash_split(str(i)) for i in range(1000)} == {"train", "val", "test"}


def test_build_and_validate_complete_archives(tmp_path):
    import csv
    import io
    import tarfile
    import zipfile
    from import_emotion_datasets import build, validate_outputs
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "esconv.json").write_text(json.dumps([{"emotion_type": "sadness", "dialog": [
        {"speaker": "seeker", "content": "I feel lonely and miss my friends", "annotation": {}},
        {"speaker": "supporter", "content": "Would you like to tell me more?", "annotation": {"strategy": "Question"}}]}]))
    with tarfile.open(raw / "empathetic.tar.gz", "w:gz") as archive:
        for split in ("train", "valid", "test"):
            buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=["conv_id", "utterance_idx", "speaker_idx", "context", "utterance"])
            writer.writeheader()
            for i, utterance in enumerate([f"I miss my friend from {split}", "What happened to your friend?"]):
                writer.writerow(dict(conv_id=split, utterance_idx=i+1, speaker_idx=i, context="sadness", utterance=utterance))
            data = buf.getvalue().encode(); info = tarfile.TarInfo(f"empatheticdialogues/{split}.csv"); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(raw / "dailydialog.zip", "w") as archive:
        archive.writestr("data/dialogues.json", "[]")
    for split in ("train", "valid", "test"):
        row = dict(Utterance=f"我的话{split}", Emotion="sadness", Sentiment="negative", DA="comfort",
                   TV_ID=split, Dialogue_ID=split, Utterance_ID="0", Speaker="x")
        with (raw / f"cped_{split}.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    (raw / "goemotions_labels.txt").write_text("sadness\nneutral\n")
    for split in ("train", "dev", "test"):
        (raw / f"goemotions_{split}.tsv").write_text(f"Unique comment for {split}\t0,1\t{split}\n")
    legacy = tmp_path / "pairs.json"; original = '[{"prompt": "我今天心情不好", "reply": "你怎麼了？願意聊聊嗎？"}]'
    legacy.write_text(original)
    out = tmp_path / "output"
    report = build(raw, out, legacy)
    checked = validate_outputs(out)
    assert report["pairs"]["train"]["count"] >= 1
    assert checked["classification"]["cped"] == {"train": 1, "val": 1, "test": 1}
    assert legacy.read_text() == original
    # A regression that accidentally copies a training pair into validation
    # must fail actual output validation, not only the converter's unit tests.
    train_rows = json.loads((out / "pairs_train.json").read_text())
    injected = dict(train_rows[0], split="val")
    (out / "pairs_val.json").write_text(json.dumps([injected]))
    with pytest.raises(ValueError, match="leakage"):
        validate_outputs(out)
