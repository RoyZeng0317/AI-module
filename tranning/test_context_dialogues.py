import json
from pathlib import Path

import pytest

from context_dialogues import MISSING, ROOT, generate_records, make_dataset, validate_splits
from context_evaluate import evaluate_file, evaluate_records, normalized, score


def test_reproducible_grouped_splits_and_counterfactuals():
    splits = generate_records(42)
    assert splits == generate_records(42)
    assert splits["train"] != generate_records(43)["train"]
    validate_splits(splits)
    assert len(splits["test"]) >= 100
    assert {row["task"] for row in splits["test"]} == {"name", "object", "preference", "format"}
    for rows in splits.values():
        groups = {}
        for row in rows:
            groups.setdefault(row["dialogue_id"], []).append(row)
            if row["turn_index"] == 2:
                assert row["reply"] != row["history"][1][1]
                assert len(row["history"]) == 4
            if not row["history"]:
                assert row["reply"] == MISSING[row["task"]]
        assert all({row["turn_index"] for row in group} == {1, 2} for group in groups.values() if group[0]["turn_index"])
        unique = {(json.dumps(row["history"], ensure_ascii=False), row["prompt"]) for row in rows}
        assert len(unique) >= 100


@pytest.mark.parametrize("key", ["dialogue_id", "template_id", "entity_group"])
def test_split_leaks_are_rejected(key):
    splits = generate_records()
    splits["val"][0][key] = splits["train"][0][key]
    with pytest.raises(ValueError, match="洩漏"):
        validate_splits(splits)


def test_manifest_hashes_unknown_coverage_and_frozen_output(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    a, b = make_dataset(first), make_dataset(second)
    assert a == b
    assert a["unk_tokens"] == 0 and a["max_reply_tokens_with_eos"] <= 60
    assert a["files"]["test.json"]["records"] == 200
    assert a["files"]["regression.json"]["records"] == 20
    assert (first / "test.json").read_bytes() == (second / "test.json").read_bytes()
    original = (first / "test.json").read_bytes()
    with pytest.raises(ValueError, match="非空"):
        make_dataset(first)
    assert (first / "test.json").read_bytes() == original


def _records():
    return [
        {"dialogue_id": "a", "task": "name", "turn_index": 1, "history": [["我叫小明。", "知道了。"]], "prompt": "我叫什麼名字？", "reply": "小明", "no_history_reply": "你還沒有告訴我名字。"},
        {"dialogue_id": "b", "task": "name", "turn_index": 1, "history": [["我叫小安。", "知道了。"]], "prompt": "我叫什麼名字？", "reply": "小安", "no_history_reply": "你還沒有告訴我名字。"},
    ]


def test_exact_scoring_rejects_extra_wrong_answers_and_format_changes():
    row = _records()[0]
    assert score(row, "小明。") == {"exact": True, "task_success": True}
    assert score(row, "你叫小明。") == {"exact": False, "task_success": True}
    assert not score(row, "你不是小明，是小安。拉拉拉")["task_success"]
    assert not score(dict(row, task="format", reply="答案：小明"), "名稱：小明")["task_success"]
    assert normalized("  答案：小明。 ") == "答案:小明"


def test_counterfactual_expected_changes_with_history_and_cache():
    rows = _records()
    calls = []
    expected = {json.dumps(row["history"], ensure_ascii=False): row["reply"] for row in rows}
    expected["[]"] = rows[0]["no_history_reply"]
    def predictor(prompt, history):
        calls.append(history)
        return expected[json.dumps(history, ensure_ascii=False)]
    result = evaluate_records(rows, predictor)
    assert result["summary"]["macro_task_success"] == 1
    assert result["summary"]["contrastive_success"] == 1
    assert result["summary"]["missing_history_success"] == 1
    assert len(calls) == 3
    assert result["rows"][0]["history_swapped"]["expected"] == "小安"


def test_always_same_answer_does_not_pass_context_dependency():
    result = evaluate_records(_records(), lambda prompt, history: "小明")
    assert result["summary"]["macro_exact"] == .5
    assert result["summary"]["contrastive_success"] == 0
    assert result["summary"]["missing_history_success"] == 0


def test_duplicate_count_and_conflicting_expected_answer():
    rows = _records()
    result = evaluate_records(rows + [rows[0]], lambda prompt, history: "小明", counterfactual=False)
    assert result["summary"]["unique_inputs"] == 2
    assert result["summary"]["repeated_inputs"] == 1
    assert result["unique_input_summary"]["macro_exact"] == .5
    with pytest.raises(ValueError, match="衝突"):
        evaluate_records([rows[0], dict(rows[0], reply="錯誤")], lambda prompt, history: "小明", counterfactual=False)


def test_regression_metadata_uses_checkpoint_format(tmp_path, monkeypatch):
    import transformer_chat
    folder = tmp_path / "checkpoint"; folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"context_format": "dialogue_v1"}))
    for name in ("model.pt", "bpe_vocab.json", "bpe_merges.json"):
        (folder / name).write_bytes(b"fake")
    data = tmp_path / "regression.json"
    data.write_text(json.dumps([dict(_records()[0], history=[])]), encoding="utf-8")
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs)
        return "小明"
    monkeypatch.setattr(transformer_chat, "reply", reply)
    result = evaluate_file(data, folder, tmp_path / "result.json", legacy=True)
    assert result["metadata"]["mode"] == "single_turn_regression"
    assert result["metadata"]["format"] == "dialogue_v1"
    assert calls[0]["max_new_tokens"] == 60 and "context_format" not in calls[0]


def test_output_cannot_leave_project():
    with pytest.raises(ValueError, match="專案內"):
        make_dataset(ROOT.parent / "outside-context-data")
