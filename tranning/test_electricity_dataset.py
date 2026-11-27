"""只測資料準備，不載入／更新模型，不執行附件程式或公式。"""
import copy, json, zipfile
from pathlib import Path

import pytest
import electricity_dataset as data


@pytest.fixture
def reviewed_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    inventory, extracts, indexes, bank = {"archive_sha256": "0" * 64, "files": []}, {}, [], []
    for index, topic in enumerate(("Voltage", "Current", "Power"), 101):
        question = f"{topic} test: choose the matching physical quantity."
        options = {"A": f"{topic} first", "B": f"{topic} second", "C": f"{topic} third", "D": f"{topic} fourth"}
        refs = []
        for role in ("question", "answer"):
            relative = f"source/{index}_{role}.pdf"
            path = tmp_path / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"raw immutable source {index} {role}".encode())
            digest = data.file_hash(path)
            entry = {"path": relative, "sha256": digest, "bytes": path.stat().st_size}
            inventory["files"].append(entry)
            header = f"{index}學年度 Basic electricity\n"
            text = ("統一入學測驗公告答案\n" + "\n".join(f"{number}\nA" for number in range(1, 51))) if role == "answer" else "1. " + question + "\n" + "\n".join(f"({k}) {v}" for k, v in options.items())
            extracts[relative] = {"parts": [{"page": 1, "text": header}, {"page": 2, "text": text}], "risks": []}
            indexes.append({"file": relative, "sha256": digest, "risks": [], "parts": 2})
            refs.append({"file": relative, "sha256": digest, "page": 2, "question_number": 1, "year": index, "subject": "Basic electricity"})
        refs[1]["method"] = "official_key"
        bank.append({"id": f"exam-{index}-q1", "question": question, "options": options, "correct_option": "A", "year": index,
                     "subject": "基本電學", "group_id": f"exam-{index}", "reviewed": True, "source": refs[0], "answer_source": refs[1], "review_note": "EVALUATOR_ONLY_SENTINEL"})
    inventory_path, bank_path = tmp_path / "inventory.json", tmp_path / "bank.json"
    data.write_json(inventory_path, inventory)
    data.write_json(bank_path, bank)
    monkeypatch.setattr(data, "index_sources", lambda source: (copy.deepcopy(indexes), copy.deepcopy(extracts), []))
    return {"root": tmp_path, "inventory": inventory, "extracts": extracts, "bank": bank, "inventory_path": inventory_path, "bank_path": bank_path}


def _validate(fixture, row):
    data.validate_reviewed(row, {entry["path"]: entry for entry in fixture["inventory"]["files"]}, fixture["extracts"])


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt", "C:/escape.txt", "..\\escape.txt"])
def test_zip_traversal_rejected_before_any_extract(tmp_path, monkeypatch, name):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe.txt", "safe")
        bundle.writestr(name, "bad")
    with pytest.raises(ValueError):
        data.safe_extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_zip_duplicate_and_size_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("same.txt", "1")
        bundle.writestr("SAME.txt", "2")
    with pytest.raises(ValueError, match="重名"):
        data.safe_extract_archive(archive, tmp_path / "out")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("big.txt", "123456")
    with pytest.raises(ValueError, match="大小"):
        data.safe_extract_archive(archive, tmp_path / "out", max_bytes=2)


def test_attached_python_is_data_not_executed(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    path = tmp_path / "attached.py"
    path.write_text("raise RuntimeError('must never run')", encoding="utf-8")
    assert "not_executed" in data.extract_source(path)["risks"][0]


def test_office_xml_keeps_math_and_formula_evidence_without_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    docx = tmp_path / "math.docx"
    with zipfile.ZipFile(docx, "w") as bundle:
        bundle.writestr("word/document.xml", f'<w:document xmlns:w="{data.NS["w"]}" xmlns:m="{data.NS["m"]}"><w:p><w:r><w:t>Formula </w:t></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p></w:document>')
    result = data.extract_source(docx)
    assert result["omml_count"] == 1 and "x" in result["parts"][0]["text"] and "omml_xml" in result
    xlsx = tmp_path / "formulas.xlsx"
    with zipfile.ZipFile(xlsx, "w") as bundle:
        bundle.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{data.NS["s"]}"><sheetData><row><c r="A1"><f>RAND()*100</f><v>17</v></c></row></sheetData></worksheet>')
    cell = data.extract_source(xlsx)["parts"][0]["cells"][0]
    assert cell == {"cell": "A1", "formula": "RAND()*100", "cached_value": "17"}


@pytest.mark.parametrize("value", ["", "AB", None, 1, [], "送分"])
def test_invalid_gold_is_never_substring_accepted(reviewed_sources, value):
    row = copy.deepcopy(reviewed_sources["bank"][0])
    row["correct_option"] = value
    with pytest.raises(ValueError):
        _validate(reviewed_sources, row)


def test_pairing_uses_year_subject_question_number_and_real_gold(reviewed_sources):
    source = reviewed_sources["bank"][0]
    _validate(reviewed_sources, source)
    for field, value in (("subject", "Wrong subject"), ("year", 999), ("question_number", 2), ("sha256", "1" * 64)):
        changed = copy.deepcopy(source)
        changed["answer_source"][field] = value
        with pytest.raises(ValueError):
            _validate(reviewed_sources, changed)
    changed = copy.deepcopy(source)
    changed["correct_option"] = "D"
    with pytest.raises(ValueError, match="原文"):
        _validate(reviewed_sources, changed)


def test_forged_identical_metadata_still_checks_source_header(reviewed_sources):
    row = copy.deepcopy(reviewed_sources["bank"][0])
    row["source"]["subject"] = row["answer_source"]["subject"] = "Forged subject"
    with pytest.raises(ValueError, match="首頁"):
        _validate(reviewed_sources, row)


def test_answer_leak_and_wrong_question_block_are_quarantined(reviewed_sources):
    row = copy.deepcopy(reviewed_sources["bank"][0])
    row["question"] += " 答案：A"
    with pytest.raises(ValueError, match="答案"):
        _validate(reviewed_sources, row)
    row = copy.deepcopy(reviewed_sources["bank"][0])
    row["source"]["question_number"] = row["answer_source"]["question_number"] = 2
    with pytest.raises(ValueError, match="題號"):
        _validate(reviewed_sources, row)


def test_decimal_option_is_not_a_new_question_boundary():
    block, answer = data.question_block("( A ) 1. 題幹\n(A) 1.9\n(B) 2.5\n( B ) 2. 下一題", 1)
    assert "2.5" in block and "下一題" not in block and answer == "A"


def test_variant_and_reordered_options_share_group_split():
    rows = [{"id": "a", "question": "A 2 V source drives a 3 Ω resistor.", "options": {"A": "1", "B": "2"}, "group_id": "year1"},
            {"id": "b", "question": "A 5 V source drives a 9 Ω resistor.", "options": {"A": "2", "B": "1"}, "group_id": "year2"},
            {"id": "c", "question": "Another independent topic", "options": {}, "group_id": "year3"}]
    assignment, groups = data.group_splits(rows)
    assert assignment["a"] == assignment["b"]
    assert len(groups) == 2 and all(split == "train" for _, split in assignment.values())


def test_prepare_is_sealed_question_only_and_group_disjoint(reviewed_sources):
    fixture = reviewed_sources
    output = fixture["root"] / "dataset"
    manifest = data.prepare_dataset(fixture["inventory_path"], fixture["bank_path"], output)
    assert manifest["eligible"] == 3 and manifest["heldout_groups_available"]
    sets = []
    for split in data.SPLITS:
        _, rows = data.load_questions(output, split)
        assert len(rows) == 1
        raw = (output / f"questions/{split}.jsonl").read_text(encoding="utf-8")
        assert "correct_option" not in raw and "SENTINEL" not in raw and "answer_source" not in raw
        sets.append({row["group_id"] for row in rows})
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    with pytest.raises(ValueError, match="覆寫"):
        data.prepare_dataset(fixture["inventory_path"], fixture["bank_path"], output)
    (output / "questions/train.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        data.load_questions(output, "train")


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_duplicate_keys_remove_every_copy(reviewed_sources, reverse):
    fixture = reviewed_sources
    duplicate = copy.deepcopy(fixture["bank"][0])
    duplicate["id"], duplicate["correct_option"] = "conflicting-copy", "B"
    bank = [duplicate, *fixture["bank"]] if reverse else [*fixture["bank"], duplicate]
    data.write_json(fixture["bank_path"], bank)
    output = fixture["root"] / "conflicts"
    manifest = data.prepare_dataset(fixture["inventory_path"], fixture["bank_path"], output)
    assert manifest["eligible"] == 2
    keys = data.read_jsonl(output / "evaluator/keys.jsonl")
    assert not {"conflicting-copy", fixture["bank"][0]["id"]} & {row["id"] for row in keys}


def test_current_corpus_contamination_is_detected(reviewed_sources):
    fixture = reviewed_sources
    path = fixture["root"] / "data/corpus_test.txt"
    path.parent.mkdir()
    path.write_text(fixture["bank"][0]["question"], encoding="utf-8")
    report = data.contamination_scan(fixture["bank"])
    assert report["matches"][0]["id"] == fixture["bank"][0]["id"]
    assert report["historical_training_exposure"].startswith("unknown")
