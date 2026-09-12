"""Tests for schmitt_trigger_train.py and its data generator.

Same contract as test_transformer_chat.py: no claim about generation
*quality* (18 hand-designed pairs, see to_do_list.md #29) -- these prove (1)
every generated circuit is genuinely rule-check-clean, not just "the script
didn't crash", and (2) train()/design()/check_design() wire correctly into
transformer_chat.py's tested pretrain/finetune/reply, using a tiny synthetic
corpus+pairs set (not the real ~1700-token Schmitt corpus) so the pipeline
test itself stays fast.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import schmitt_trigger_train as stt
from circuit_rule_check import check_schematic
from schmitt_trigger_gen import build_corpus, validate_pairs, wrap_kicad

_TINY_KWARGS = dict(block_size=32, n_layer=2, n_embd=16, n_head=2, vocab_size=100)


def test_generated_corpus_has_no_rule_check_errors(tmp_path):
    pairs = build_corpus()
    assert len(pairs) > 0
    validate_pairs(pairs, tmp_path)  # raises AssertionError if any circuit is broken

    unique_replies = {p["reply"] for p in pairs}  # bodies only, see build_corpus() docstring
    for body in unique_replies:
        path = tmp_path / "recheck.kicad_sch"
        path.write_text(wrap_kicad(body), encoding="utf-8")
        errors = [f for f in check_schematic(path) if f["severity"] == "error"]
        assert errors == []


def test_generated_corpus_covers_every_topology():
    pairs = build_corpus()
    unique_replies = {p["reply"] for p in pairs}
    assert any("Sinco:OpAmp" in r for r in unique_replies)
    assert any("Sinco:SchmittInverter" in r for r in unique_replies)
    # 5 templates (schmitt-opamp, schmitt-inverter, inverting/non-inverting amp, comparator) --
    # R1/R2 are placeholder tokens shared across every value variant of a topology, see
    # build_corpus()'s docstring, so this is templates not unique (prompt, value) combos.
    # (voltage follower was tried as a 6th and reverted, see build_corpus()'s comment)
    assert len(unique_replies) == 5


def test_fill_values_substitutes_placeholders_from_prompt():
    from schmitt_trigger_gen import PLACEHOLDER_R1, PLACEHOLDER_R2

    text = f'(property "Value" "{PLACEHOLDER_R1}" ...)(property "Value" "{PLACEHOLDER_R2}" ...)'
    filled = stt.fill_values(text, "幫我設計一個史密特觸發電路，R1 用 10k，R2 用 100k")
    assert filled == '(property "Value" "10k" ...)(property "Value" "100k" ...)'


def test_fill_values_handles_decimal_and_no_space_prompts():
    from schmitt_trigger_gen import PLACEHOLDER_R1, PLACEHOLDER_R2

    text = f"{PLACEHOLDER_R1} {PLACEHOLDER_R2}"
    filled = stt.fill_values(text, "R1=4.7k R2=220k")
    assert filled == "4.7k 220k"


def test_check_design_reports_clean_for_valid_circuit(tmp_path):
    pairs = build_corpus()
    text = pairs[0]["reply"]
    findings = stt.check_design(text, tmp_path / "out.kicad_sch")
    assert findings == []


def test_check_design_reports_unparsable_instead_of_raising(tmp_path):
    findings = stt.check_design("not a valid kicad_sch (((", tmp_path / "broken.kicad_sch")
    assert len(findings) == 1
    assert findings[0]["rule"] == "unparsable"
    assert findings[0]["severity"] == "error"


def _tiny_synthetic_corpus() -> str:
    return '(kicad_sch (version 1) (paper "A4") (wire (pts (xy 0 0) (xy 1 1))))' * 10


def _tiny_synthetic_pairs() -> list[dict]:
    return [
        {"prompt": "設計電路 A", "reply": "(kicad_sch (paper A) (wire 1))"},
        {"prompt": "設計電路 B", "reply": "(kicad_sch (paper B) (wire 2))"},
    ] * 5


def test_train_and_design_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_tiny_synthetic_corpus(), encoding="utf-8")
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps(_tiny_synthetic_pairs()), encoding="utf-8")

    monkeypatch.setattr(stt, "regenerate_corpus", lambda: None)
    monkeypatch.setattr(stt, "CORPUS_PATH", corpus_path)
    monkeypatch.setattr(stt, "PAIRS_PATH", pairs_path)
    monkeypatch.setattr(stt, "PRETRAIN_DIR", tmp_path / "pretrain_runs")
    monkeypatch.setattr(stt, "FINETUNE_DIR", tmp_path / "chat_runs")
    for key, value in _TINY_KWARGS.items():
        monkeypatch.setattr(stt, key.upper() if key != "block_size" else "BLOCK_SIZE", value)
    monkeypatch.setattr(stt, "N_LAYER", _TINY_KWARGS["n_layer"])
    monkeypatch.setattr(stt, "N_EMBD", _TINY_KWARGS["n_embd"])
    monkeypatch.setattr(stt, "N_HEAD", _TINY_KWARGS["n_head"])
    monkeypatch.setattr(stt, "VOCAB_SIZE", _TINY_KWARGS["vocab_size"])

    stt.train(pretrain_epochs=1, finetune_epochs=1)

    assert (stt.PRETRAIN_DIR / "model.pt").exists()
    assert (stt.FINETUNE_DIR / "model.pt").exists()

    generated = stt.design("設計電路 A", max_new_tokens=10)
    assert isinstance(generated, str)
    findings = stt.check_design(generated, tmp_path / "design_out.kicad_sch")
    assert isinstance(findings, list)  # untrained-scale output likely won't parse; must not raise
