"""to-do #50 v2 資料、訓練參數與驗收判定的契約測試；不以這些測試主張模型語意正確。"""
import json
from pathlib import Path

import pytest

from context_dialogues import validate_splits
from context_dialogues_v2 import (FORMS, PREFIXES, SPLIT_NAMES, TASKS, check_entity_isolation,
                                  generate_records, make_dataset)
from context_evaluate import evaluate_records
from context_validate import CRITERIA, format_summary, judge

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = next((p for p in (ROOT / "tranning/gpt_chat_runs", ROOT / "tranning/gpt_context_runs/pilot_v2")
                  if (p / "bpe_vocab.json").exists()), None)


@pytest.fixture(scope="module")
def splits():
    return generate_records(42)


def test_reproducible_and_no_group_or_input_leak(splits):
    assert splits == generate_records(42)
    assert splits["train"] != generate_records(7)["train"]
    validate_splits(splits)  # dialogue/template/entity_group 與完全相同輸入都不跨 split


def test_every_split_covers_every_entity_form(splits):
    """v1 的缺陷：test 只含訓練沒見過的字形。v2 每個 split 都要有全部字形。"""
    coverage = check_entity_isolation(splits)
    for task in TASKS:
        for split in SPLIT_NAMES:
            assert coverage[task][split] == sorted(FORMS[task])


def test_answer_entities_never_cross_splits(splits):
    def entities(split):
        out = set()
        for row in splits[split]:
            if row["kind"] in ("recall", "corrected"):
                reply = row["reply"]
                out.add(next((reply[len(p):] for p in PREFIXES if reply.startswith(p)), reply))
        return out
    train, val, test = (entities(s) for s in SPLIT_NAMES)
    assert not (train & val) and not (train & test) and not (val & test)


def test_isolation_check_rejects_entity_leak(splits):
    leaked = {k: [dict(r) for r in v] for k, v in splits.items()}
    train_row = next(r for r in leaked["train"] if r["task"] == "name" and r["kind"] == "recall")
    test_row = next(r for r in leaked["test"] if r["task"] == "name" and r["kind"] == "recall")
    test_row["reply"] = train_row["reply"]
    with pytest.raises(ValueError, match="洩漏"):
        check_entity_isolation(leaked)


def test_kinds_and_answers_are_consistent(splits):
    for rows in splits.values():
        kinds = {r["kind"] for r in rows}
        assert kinds == {"recall", "corrected", "missing_irrelevant", "missing_empty"}
        for row in rows:
            text = "".join(u + a for u, a in row["history"])
            if row["kind"].startswith("missing"):
                assert row["reply"] == row["no_history_reply"]
            elif row["kind"] == "recall":
                core = next((row["reply"][len(p):] for p in PREFIXES if row["reply"].startswith(p)), row["reply"])
                assert core in text  # 答案一定在前文出現過，是「抄回」而不是背
            else:
                core = next((row["reply"][len(p):] for p in PREFIXES if row["reply"].startswith(p)), row["reply"])
                earlier = next(a for u, a in row["history"] if u == row["prompt"])  # 更正前模型的回答
                assert core in text and row["reply"] != earlier and row["turn_index"] == 2
        assert len(rows) >= 380


def test_test_split_size_is_balanced(splits):
    from collections import Counter
    counts = Counter(r["task"] for r in splits["test"])
    assert all(95 <= counts[t] <= 100 for t in TASKS)


@pytest.mark.skipif(TOKENIZER is None, reason="沒有可用的 BPE tokenizer")
def test_make_dataset_has_no_unk_and_refuses_overwrite(tmp_path, monkeypatch):
    import context_dialogues_v2 as v2
    monkeypatch.setattr(v2, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/pairs.json").write_text((ROOT / "data/pairs.json").read_text(encoding="utf-8"), encoding="utf-8")
    manifest = make_dataset(tmp_path / "out", 42, TOKENIZER)
    assert manifest["unk_tokens"] == 0 and manifest["max_reply_tokens_with_eos"] <= 60
    assert manifest["files"]["regression.json"]["records"] == 20
    with pytest.raises(ValueError, match="非空"):
        make_dataset(tmp_path / "out", 42, TOKENIZER)


def _fake_v2(perfect=True):
    rows = generate_records(42)["test"][:120]
    def predictor(prompt, history):
        if not perfect:
            return "不知道"
        for row in rows:
            if row["prompt"] == prompt and row["history"] == history:
                return row["reply"]
        return next(r["no_history_reply"] for r in rows if r["prompt"] == prompt)
    return evaluate_records(rows, predictor)


def test_judge_passes_perfect_and_fails_constant_model():
    regression = {"summary": {"macro_exact": 0.5}}
    good = judge(_fake_v2(True), regression, {"summary": {"macro_exact": 0.52}})
    assert good["passed"], good["checks"]
    bad = judge(_fake_v2(False), regression, {"summary": {"macro_exact": 0.52}})
    assert not bad["passed"]
    assert not bad["checks"]["v2_min_form_success"]["passed"]


def test_judge_flags_single_turn_regression_and_missing_baseline():
    v2 = _fake_v2(True)
    dropped = judge(v2, {"summary": {"macro_exact": 0.3}}, {"summary": {"macro_exact": 0.6}})
    assert dropped["checks"]["regression_max_drop"]["passed"] is False and not dropped["passed"]
    no_base = judge(v2, {"summary": {"macro_exact": 0.3}}, None)
    assert no_base["checks"]["regression_max_drop"]["passed"] is None and not no_base["passed"]
    assert "未判定" in format_summary(no_base, "x")


def test_criteria_are_declared_up_front():
    assert set(CRITERIA) == {"v2_macro_task_success", "v2_min_task_success", "v2_min_form_success",
                             "v2_missing_success", "v2_contrastive_success", "v2_missing_history_success",
                             "regression_max_drop"}


def test_replay_repeat_upweights_legacy_data(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import context_training
    import transformer_chat as transformer
    from bpe_tokenizer import BPETokenizer
    monkeypatch.setattr(context_training, "ROOT", tmp_path)
    source = tmp_path / "source"; source.mkdir()
    tokenizer = BPETokenizer.train(["使用者：我叫小明\n助手：你好\n我叫什麼名字小明小安早安"], vocab_size=80)
    tokenizer.save(source)
    (source / "config.json").write_text(json.dumps({"block_size": 96, "n_layer": 1, "n_embd": 16, "n_head": 2, "dropout": 0.0}))
    torch.save(transformer.GPT(tokenizer.vocab_size, 96, 1, 16, 2, 0).state_dict(), source / "model.pt")
    rec = {"dialogue_id": "t", "history": [["我叫小明", "你好"]], "prompt": "我叫什麼", "reply": "小明"}
    files = {}
    for name, rows in {"train": [rec], "val": [dict(rec, dialogue_id="v", history=[["我叫小安", "你好"]], reply="小安")],
                       "replay": [{"dialogue_id": "r", "history": [], "prompt": "早安", "reply": "你好"}]}.items():
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    context_training.finetune_context(files["train"], files["val"], source, tmp_path / "out", epochs=1, batch_size=1,
                                      freeze_layers=0, response_budget=16, device="cpu",
                                      replay_path=files["replay"], replay_repeat=3)
    meta = json.loads((tmp_path / "out/config.json").read_text(encoding="utf-8"))["context_training"]
    assert meta["replay_repeat"] == 3 and meta["train_examples"] == 4
    with pytest.raises(ValueError, match="replay_repeat"):
        context_training.finetune_context(files["train"], files["val"], source, tmp_path / "out2",
                                          replay_path=files["replay"], replay_repeat=0)
