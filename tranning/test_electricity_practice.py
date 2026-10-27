"""先作答後讀鍵、唯一末行解析及 frozen heldout；模型替身不做 optimizer step。"""
import copy, json
from pathlib import Path
from types import SimpleNamespace

import pytest
import electricity_dataset as data
import electricity_practice as practice
from test_electricity_dataset import reviewed_sources


class FakeGPT:
    metadata = {"checkpoint": "fake-test-only", "files": {"model.pt": "f" * 64}}
    def __init__(self, completion="答案：A", status="ready"):
        self.completion, self.status, self.prompts = completion, status, []
    def inspect(self, prompt, max_new_tokens):
        return {"status": self.status, "block_size": 512, "prompt_tokens_with_sep": 120, "unknown_tokens": int(self.status == "not_attempted_encoding"), "unknown_chars": []}
    def generate(self, prompt, parameters, seed):
        self.prompts.append(prompt)
        return self.completion
    def verify_unchanged(self): pass


@pytest.fixture
def dataset(reviewed_sources):
    fixture = reviewed_sources
    path = fixture["root"] / "dataset"
    data.prepare_dataset(fixture["inventory_path"], fixture["bank_path"], path)
    return path


@pytest.mark.parametrize("text,answer", [("答案：A", "A"), ("計算完成。\n答案：(2)", "B"), ("Final answer: d.", "D"), ("答案：4", "D")])
def test_unique_final_choice(text, answer):
    assert practice.parse_mcq(text) == answer


@pytest.mark.parametrize("text", ["A", "The answer might be A or B", "答案：A 或 B", "答案：A\n答案：B", "答案：A\n答案：A", "A B C D", "A B C D\n答案：A", "答案：12", "答案：BA", "我選B\n答案：A", "B is correct\nAnswer: A", "選擇C\n答案：A", "答案：A\n補充說明", "", "答案：A、B、C、D"])
def test_ambiguity_enumeration_and_contradiction_never_rewarded(text):
    assert practice.parse_mcq(text) is None


def test_exact_choice_not_substring_and_numeric_units_are_manual():
    key = {"kind": "mcq", "correct_option": "B", "reviewed": True}
    assert practice.score_completion("答案：A", key)["verdict"] == "incorrect"
    assert practice.score_completion("答案：2", key)["verdict"] == "correct"
    assert practice.score_completion("答案：12", key)["verdict"] == "unparseable"
    for completion in ("答案：12 V", "答案：2 A", "答案：2e-3 V"):
        result = practice.score_completion(completion, {"kind": "numeric", "value": "2", "unit": "V", "reviewed": True})
        assert result["verdict"] == "needs_manual_review"


def test_question_text_preserves_minus_superscript_and_units():
    prompt = practice.format_question({"question": "電流為 –10⁻³ A，電阻 2 Ω。", "options": dict(zip("ABCD", ("2 μV", "3 μV", "4 μV", "5 μV")))})
    assert "–10⁻³ A" in prompt and "2 Ω" in prompt and "2 μV" in prompt
    assert "答案：A" not in prompt  # 固定格式不示範某個候選答案。


def test_real_adapter_preflight_counts_sep_and_full_response_budget():
    adapter = object.__new__(practice.GPTAdapter)
    adapter.block_size = 32
    adapter.tokenizer = SimpleNamespace(encode=lambda prompt: [5] * len(prompt), vocab={"x": 5})
    assert adapter.inspect("x" * 23, 8)["status"] == "ready"
    assert adapter.inspect("x" * 24, 8)["status"] == "not_attempted_context_limit"
    adapter.tokenizer = SimpleNamespace(encode=lambda prompt: [3], vocab={})
    assert adapter.inspect("Ω", 8)["status"] == "not_attempted_encoding"


def test_missing_checkpoint_is_error_not_untrained_placeholder(tmp_path):
    with pytest.raises(ValueError, match="不完整"):
        practice.GPTAdapter(tmp_path)


def test_generation_never_reads_gold_and_attempt_is_durable_before_grade(dataset, tmp_path, monkeypatch):
    output = tmp_path / "attempt.jsonl"
    generator = FakeGPT()
    real_hash = data.file_hash
    def key_guard(path):
        assert not str(path).endswith("evaluator/keys.jsonl"), "attempt must not read evaluator keys"
        return real_hash(path)
    monkeypatch.setattr(data, "file_hash", key_guard)
    rows = practice.attempt(dataset, "train", output, adapter=generator)
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert all("EVALUATOR_ONLY_SENTINEL" not in prompt and "correct_option" not in prompt for prompt in generator.prompts)
    monkeypatch.setattr(data, "file_hash", real_hash)
    original = practice._load_keys
    def load_after_attempt(path):
        saved = data.read_jsonl(output)
        assert saved == rows and saved[0]["raw_completion"] == "答案：A"
        return original(path)
    monkeypatch.setattr(practice, "_load_keys", load_after_attempt)
    report = practice.grade(dataset, output, tmp_path / "grade.json", tmp_path / "feedback.jsonl")
    assert report["summary"]["correct"] == 1 and report["summary"]["attempted"] == 1
    feedback = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8")
    assert "correct_option" not in feedback and "EVALUATOR_ONLY_SENTINEL" not in feedback and "答案：A" not in feedback


@pytest.mark.parametrize("status", ["not_attempted_context_limit", "not_attempted_encoding"])
def test_no_truncation_or_symbol_deletion_when_prompt_cannot_fit(dataset, tmp_path, status):
    generator = FakeGPT(status=status)
    output = tmp_path / "attempt.jsonl"
    rows = practice.attempt(dataset, "train", output, adapter=generator)
    assert not generator.prompts and rows[0]["status"] == status
    report = practice.grade(dataset, output, tmp_path / "grade.json")
    assert report["summary"][status] == 1 and report["summary"]["first_pass_accuracy"] is None


@pytest.mark.parametrize("split", ["dev", "test"])
def test_no_heldout_feedback_or_key_read_for_feedback_request(dataset, tmp_path, monkeypatch, split):
    output = tmp_path / "attempt.jsonl"
    practice.attempt(dataset, split, output, adapter=FakeGPT())
    monkeypatch.setattr(practice, "_load_keys", lambda path: pytest.fail("heldout feedback must be rejected before key read"))
    with pytest.raises(ValueError, match="只有 train"):
        practice.grade(dataset, output, tmp_path / "grade.json", tmp_path / "feedback.jsonl")
    assert not (tmp_path / "grade.json").exists() and not (tmp_path / "feedback.jsonl").exists()


def test_frozen_test_only_first_attempt(dataset, tmp_path):
    practice.attempt(dataset, "test", tmp_path / "first.jsonl", adapter=FakeGPT())
    with pytest.raises(ValueError, match="不可重試"):
        practice.attempt(dataset, "test", tmp_path / "second.jsonl", adapter=FakeGPT())
    assert not (tmp_path / "second.jsonl").exists()


@pytest.mark.parametrize("field,value", [("question_id", "unknown"), ("question_sha256", "bad"), ("source_sha256", "bad"), ("dataset_sha256", "bad"), ("split", "test"), ("prompt", "answer leak")])
def test_grade_rejects_tampered_attempt_identity_before_keys(dataset, tmp_path, monkeypatch, field, value):
    output = tmp_path / "attempt.jsonl"
    practice.attempt(dataset, "train", output, adapter=FakeGPT())
    rows = data.read_jsonl(output)
    rows[0][field] = value
    data.write_jsonl(output, rows)
    monkeypatch.setattr(practice, "_load_keys", lambda path: pytest.fail("identity check must run before gold read"))
    with pytest.raises(ValueError, match="不符"):
        practice.grade(dataset, output, tmp_path / "grade.json")


def test_key_tampering_and_empty_attempts_cannot_produce_fake_scores(dataset, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="為空"):
        practice.grade(dataset, empty, tmp_path / "empty_grade.json")
    output = tmp_path / "attempt.jsonl"
    practice.attempt(dataset, "train", output, adapter=FakeGPT())
    key = dataset / "evaluator/keys.jsonl"
    key.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        practice.grade(dataset, output, tmp_path / "grade.json")


def test_unparseable_is_in_accuracy_denominator_not_dropped(dataset, tmp_path):
    output = tmp_path / "attempt.jsonl"
    practice.attempt(dataset, "train", output, adapter=FakeGPT("我不知道"))
    summary = practice.grade(dataset, output, tmp_path / "grade.json")["summary"]
    assert summary["eligible"] == summary["attempted"] == summary["unparseable"] == 1
    assert summary["correct"] == 0 and summary["first_pass_accuracy"] == 0


def test_duplicate_attempt_rows_rejected(dataset, tmp_path):
    output = tmp_path / "attempt.jsonl"
    rows = practice.attempt(dataset, "train", output, adapter=FakeGPT())
    data.write_jsonl(output, rows + rows)
    with pytest.raises(ValueError, match="重複"):
        practice.grade(dataset, output, tmp_path / "grade.json")


def test_chat_retention_uses_raw_prompts_only(tmp_path):
    generator = FakeGPT("raw chat response")
    output = tmp_path / "chat.jsonl"
    practice.chat_baseline(output, adapter=generator)
    assert generator.prompts == ["你好", "你是誰？", "how are you?"]
    assert len(data.read_jsonl(output)) == 3
