"""基本電學資料準備：只讀來源、人工核對種子、題目／評分鍵隔離；不訓練。"""
import argparse, hashlib, json, random, re, unicodedata, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/basic_electricity/v1"
SCHEMA_VERSION = 1
SPLITS = ("train", "dev", "test")
QUESTION_FIELDS = {"schema_version", "id", "kind", "question", "options", "diagram_description", "source", "source_group_id", "group_id", "split", "question_sha256"}
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def inside_project(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"輸出／專案來源不可離開專案：{path}")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def normalized(text):
    # 只用於比對，不回寫題目：原始 μ、負號與上／下標必須保留。
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def question_digest(question):
    return object_hash({key: value for key, value in question.items() if key != "question_sha256"})


def safe_extract_archive(archive, destination, max_files=100, max_bytes=200 * 1024 * 1024):
    """先驗證所有成員，再解壓；不覆寫、不執行附件。archive 可為使用者明確提供的位置。"""
    destination = inside_project(destination)
    with zipfile.ZipFile(archive) as bundle:
        files, seen, total = [], set(), 0
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if (PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or ".." in parts
                    or any(":" in part or part.rstrip(" .") != part for part in parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"不安全 ZIP 成員：{member.filename}")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or name.casefold() in seen or target.exists():
                raise ValueError(f"重名、覆寫或穿越 ZIP 成員：{name}")
            seen.add(name.casefold())
            if not member.is_dir():
                total += member.file_size
                files.append((member, target))
            if total > max_bytes or len(files) > max_files:
                raise ValueError("ZIP 超出數量／大小限制")
        for member, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                size = 0
                for chunk in iter(lambda: source.read(64 * 1024), b""):
                    size += len(chunk)
                    if size > member.file_size or size > max_bytes:
                        raise ValueError("ZIP 解壓大小不符")
                    output.write(chunk)
    return [target for _, target in files]


def extract_source(path):
    """保留頁／段落／公式結構證據。公式、巨集、附帶 Python 均不執行。"""
    path = inside_project(path)
    suffix, parts, risks = path.suffix.lower(), [], []
    details = {}
    if suffix == ".pdf":
        import fitz
        with fitz.open(path) as document:
            for number, page in enumerate(document, 1):
                text = page.get_text()
                parts.append({"page": number, "text": text, "images": len(page.get_images()), "drawings": len(page.get_drawings())})
        if sum(len(row["text"]) for row in parts) < 150 * len(parts):
            risks.append("scan_or_sparse_text_requires_visual_review")
        if any(row["images"] or row["drawings"] for row in parts):
            risks.append("diagrams_or_formula_layout_not_reconstructed")
    elif suffix == ".docx":
        with zipfile.ZipFile(path) as bundle:
            tree = ET.fromstring(bundle.read("word/document.xml"))
            for number, paragraph in enumerate(tree.findall(".//w:p", NS), 1):
                text = "".join(node.text or "" for node in paragraph.iter() if node.tag in (f"{{{NS['w']}}}t", f"{{{NS['m']}}}t"))
                parts.append({"paragraph": number, "text": text})
            math = tree.findall(".//m:oMath", NS)
            details = {"omml_count": len(math), "omml_xml": [ET.tostring(node, encoding="unicode") for node in math],
                       "embedded_media": [name for name in bundle.namelist() if name.startswith("word/media/")]}
            if math:
                risks.append("omml_structure_requires_review_not_plain_text")
            if details["embedded_media"]:
                risks.append("embedded_images_not_reconstructed")
    elif suffix == ".xlsx":
        with zipfile.ZipFile(path) as bundle:
            strings = []
            if "xl/sharedStrings.xml" in bundle.namelist():
                strings = ["".join(node.itertext()) for node in ET.fromstring(bundle.read("xl/sharedStrings.xml")).findall("s:si", NS)]
            for name in sorted(n for n in bundle.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)):
                cells = []
                for cell in ET.fromstring(bundle.read(name)).findall(".//s:c", NS):
                    value = cell.findtext("s:v", "", NS)
                    if cell.get("t") == "s" and value:
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", NS).itertext())
                    cells.append({"cell": cell.get("r"), "cached_value": value, "formula": cell.findtext("s:f", "", NS)})
                parts.append({"sheet_xml": name, "cells": cells, "text": ""})
        risks.append("spreadsheet_formulas_not_executed_not_verified_answer_keys")
    else:
        risks.append("attached_program_or_unsupported_format_not_executed")
    for row in parts:
        row["text_sha256"] = object_hash(normalized(row["text"]))
    return {"parts": parts, "risks": risks, **details}


def index_sources(inventory):
    indexes, extracts, pages = [], {}, defaultdict(list)
    for entry in inventory["files"]:
        path = inside_project(ROOT / entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"來源 SHA 不符：{entry['path']}")
        extracted = extract_source(path)
        extracts[entry["path"]] = extracted
        indexes.append({"file": entry["path"], "sha256": entry["sha256"], "bytes": path.stat().st_size,
                        "format": path.suffix.lower(), "parts": len(extracted["parts"]),
                        "text_chars": sum(len(row["text"]) for row in extracted["parts"]),
                        "risks": extracted["risks"], "omml_count": extracted.get("omml_count", 0)})
        for row in extracted["parts"]:
            if "page" in row and len(normalized(row["text"])) >= 200:
                pages[row["text_sha256"]].append({"file": entry["path"], "page": row["page"]})
    duplicates = [refs for refs in pages.values() if len(refs) > 1]
    return indexes, extracts, duplicates


def _verify_reference(reference, inventory, extracts):
    required = {"file", "sha256", "page", "question_number", "year", "subject"}
    if not required.issubset(reference) or reference["file"] not in inventory:
        raise ValueError("來源定位欄位不完整或來源未盤點")
    if reference["sha256"] != inventory[reference["file"]]["sha256"]:
        raise ValueError("來源定位 SHA 不符")
    if type(reference["page"]) is not int or type(reference["question_number"]) is not int or reference["question_number"] < 1:
        raise ValueError("頁碼／題號無效")
    parts = extracts[reference["file"]]["parts"]
    if not 1 <= reference["page"] <= len(parts) or not reference["subject"] or "page" not in parts[reference["page"] - 1]:
        raise ValueError("來源頁碼／科目無效")
    header = normalized(parts[0]["text"])
    if any(normalized(token) not in header for token in re.split(r"[\s、，,]+", reference["subject"]) if token):
        raise ValueError("宣告科目與來源首頁不符")
    year = reference["year"]
    if year is not None and (type(year) is not int or f"{year}學年度" not in header):
        raise ValueError("宣告年份與來源首頁不符")
    return parts[reference["page"] - 1]["text"]


def question_block(text, number):
    starts = list(re.finditer(r"(?m)^\s*(?:\(\s*([A-D])\s*\)\s*)?(\d{1,3})\s*[.．]\s*(?=[A-Za-z\u3400-\u9fff])", text))
    for index, match in enumerate(starts):
        if int(match[2]) == number:
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            return text[match.end():end], match[1]
    raise ValueError("原頁找不到明確題號區塊")


def official_answers(text):
    answers = {}
    for match in re.finditer(r"(?m)^\s*(\d{1,2})[ \t]*\n\s*([A-D]|送\s*分)[ \t]*(?:\n|$)", text):
        number, answer = int(match[1]), normalized(match[2])
        if 1 <= number <= 50:
            if number in answers:
                raise ValueError("公告答案題號重複")
            answers[number] = answer
    if set(answers) != set(range(1, 51)):
        raise ValueError("公告答案表無法完整可靠解析 1–50 題")
    return answers


def validate_reviewed(record, inventory, extracts):
    if record.get("reviewed") is not True or not re.fullmatch(r"[A-Za-z0-9_-]+", str(record.get("id", ""))):
        raise ValueError("缺少人工核對或合法 id")
    question, options = record.get("question"), record.get("options")
    if not isinstance(question, str) or not question.strip() or not isinstance(options, dict) or set(options) != set("ABCD"):
        raise ValueError("必須有完整題幹與 A-D 四選項")
    if (any(not isinstance(value, str) or not value.strip() for value in options.values())
            or not isinstance(record.get("correct_option"), str) or record["correct_option"] not in set("ABCD")):
        raise ValueError("選項／評分鍵無效")
    if re.search(r"(?:答案|正解|解析|解答)\s*[:：]|correct\s+answer|solution\s*:", question + "\n" + "\n".join(options.values()), re.I):
        raise ValueError("題目含答案／解析標記")
    if re.search(r"如圖|附圖|圖\s*\d|承上|續上|上題|送分", question) or record.get("diagram_description"):
        raise ValueError("圖形／承上／送分題未完整審核")
    source, answer = record["source"], record["answer_source"]
    text = _verify_reference(source, inventory, extracts)
    answer_text = _verify_reference(answer, inventory, extracts)
    if any(source[field] != answer[field] for field in ("year", "subject", "question_number")):
        raise ValueError("題目與答案的年份／科目／題號不一致")
    if source["year"] != record.get("year"):
        raise ValueError("題目年份與來源識別不一致")
    if answer.get("method") not in ("official_key", "inline_marker"):
        raise ValueError("答案來源方法未審核")
    block, _ = question_block(text, source["question_number"])
    page_text = normalized(block)
    if any(normalized(value) not in page_text for value in [question, *options.values()]):
        raise ValueError("題幹／選項無法完整對應原頁，需人工檢查文字或公式")
    if answer["method"] == "official_key":
        if "公告答案" not in normalized(answer_text):
            raise ValueError("答案來源不是公告答案表")
        key = official_answers(answer_text)[answer["question_number"]]
    else:
        _, key = question_block(answer_text, answer["question_number"])
    if key != record["correct_option"]:
        raise ValueError("評分鍵與來源原文不符或為送分")
    if not record.get("group_id"):
        raise ValueError("缺少來源群組")


def group_splits(records, seed=43):
    """來源群組與完全相同／僅數字不同的題族聯集，防止跨集合洩漏。"""
    parents = {row["group_id"]: row["group_id"] for row in records}
    def find(group):
        while parents[group] != group:
            group = parents[group]
        return group
    signatures = {}
    for row in records:
        text = normalized(row["question"])
        skeleton = re.sub(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", "<n>", text)
        if skeleton in signatures:
            parents[find(row["group_id"])] = find(signatures[skeleton])
        signatures[skeleton] = row["group_id"]
    members = defaultdict(list)
    for group in parents:
        members[find(group)].append(group)
    groups = sorted(members)
    random.Random(seed).shuffle(groups)
    splits = {group: "train" for group in groups}
    if len(groups) >= 3:
        for group in groups[:max(1, round(len(groups) * .2))]:
            splits[group] = "test"
        for group in groups[max(1, round(len(groups) * .2)):max(1, round(len(groups) * .2)) * 2]:
            splits[group] = "dev"
    return {row["id"]: (find(row["group_id"]), splits[find(row["group_id"])]) for row in records}, dict(members)


def contamination_scan(records):
    matches, scanned = [], []
    paths = sorted({*ROOT.glob("data/pairs*.json"), *ROOT.glob("data/*_pairs.json"), *ROOT.glob("data/corpus*.txt")})
    for path in paths:
        content = normalized(path.read_text(encoding="utf-8", errors="replace"))
        scanned.append({"file": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)})
        for row in records:
            stem = normalized(row["question"])
            if len(stem) >= 16 and stem in content:
                matches.append({"id": row["id"], "file": path.relative_to(ROOT).as_posix()})
    return {"scanned": scanned, "matches": matches, "historical_training_exposure": "unknown; only current listed files checked"}


def prepare_dataset(inventory_path, bank_path, out_dir=DEFAULT_DATASET, seed=43):
    out_dir = inside_project(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("資料版本目錄非空；不可覆寫已凍結資料，請選新的版本目錄")
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    if not isinstance(bank, list) or not bank or not all(isinstance(row, dict) for row in bank):
        raise ValueError("人工核對題庫為空或格式錯誤")
    indexes, extracts, duplicate_pages = index_sources(inventory)
    lookup = {row["path"]: row for row in inventory["files"]}
    accepted, quarantine, ids, contents, conflicted, group_records = [], [], set(), {}, set(), []
    declared = defaultdict(set)
    for record in bank:
        signature = object_hash(normalized(str(record.get("question", "")) + json.dumps(record.get("options", {}), ensure_ascii=False, sort_keys=True)))
        declared[signature].add(str(record.get("correct_option")))
    for record in bank:
        try:
            # 衝突先依內容識別，再核對 key，避免只剔除後出現的錯誤答案。
            signature = object_hash(normalized(str(record.get("question", "")) + json.dumps(record.get("options", {}), ensure_ascii=False, sort_keys=True)))
            if len(declared[signature]) > 1:
                conflicted.add(record.get("id"))
                raise ValueError("重複題目的評分鍵衝突，全部隔離")
            validate_reviewed(record, lookup, extracts)
            if record["id"] in ids:
                raise ValueError("重複題目 id")
            group_records.append(record)
            if signature in contents:
                earlier = contents[signature]
                raise ValueError(f"完全重複題，保留 {earlier['id']}")
            ids.add(record["id"])
            contents[signature] = record
            accepted.append(record)
        except (KeyError, TypeError, ValueError) as error:
            quarantine.append({"scope": "question", "id": record.get("id"), "reason": str(error), "source": record.get("source")})
    accepted = [row for row in accepted if row["id"] not in conflicted]
    contamination = contamination_scan(accepted)
    contaminated = {row["id"] for row in contamination["matches"]}
    quarantine.extend({"scope": "question", "id": row["id"], "reason": "current_training_material_exact_stem_match"} for row in accepted if row["id"] in contaminated)
    accepted = [row for row in accepted if row["id"] not in contaminated]
    if not accepted:
        raise ValueError("沒有通過審核的題目；不產生空資料集或假評分")
    assignments, groups = group_splits([row for row in group_records if row["id"] not in conflicted | contaminated], seed)
    questions, keys = [], []
    for row in accepted:
        group, split = assignments[row["id"]]
        question = {"schema_version": SCHEMA_VERSION, "id": row["id"], "kind": "mcq", "question": row["question"],
                    "options": row["options"], "source": row["source"], "source_group_id": row["group_id"], "group_id": group, "split": split}
        if row.get("diagram_description"):
            question["diagram_description"] = row["diagram_description"]
        question["question_sha256"] = question_digest(question)
        questions.append(question)
        keys.append({"schema_version": SCHEMA_VERSION, "id": row["id"], "kind": "mcq", "correct_option": row["correct_option"],
                     "question_sha256": question["question_sha256"], "split": split, "source": row["answer_source"],
                     "reviewed": True, "review_note": row.get("review_note", "")})
    used = {row["source"]["file"] for row in accepted} | {row["answer_source"]["file"] for row in accepted}
    for row in indexes:
        row["status"] = "reviewed_subset_only" if row["file"] in used else "quarantined_not_in_reviewed_seed"
        quarantine.append({"scope": "source_remainder", "file": row["file"], "sha256": row["sha256"],
                           "reason": row["risks"] or ["not_manually_paired_and_reviewed"], "reviewed_subset_exists": row["file"] in used})
    write_json(out_dir / "source_inventory.json", inventory)
    write_json(out_dir / "source_index.json", indexes)
    write_json(out_dir / "reports/duplicate_pages.json", duplicate_pages)
    write_json(out_dir / "reports/contamination.json", contamination)
    write_json(out_dir / "evaluator/reviewed_bank.json", bank)
    write_jsonl(out_dir / "evaluator/keys.jsonl", keys)
    write_jsonl(out_dir / "quarantine.jsonl", quarantine)
    for split in SPLITS:
        write_jsonl(out_dir / f"questions/{split}.jsonl", [row for row in questions if row["split"] == split])
    for index, (source, extracted) in enumerate(extracts.items()):
        write_json(out_dir / f"evaluator/extracted/{index:02d}.json", {"source": source, **extracted})
    files = {path.relative_to(out_dir).as_posix(): file_hash(path) for path in sorted(out_dir.rglob("*")) if path.is_file()}
    manifest = {"schema_version": SCHEMA_VERSION, "dataset_version": out_dir.name, "split_seed": seed,
                "archive_sha256": inventory["archive_sha256"], "reviewed_bank_sha256": file_hash(bank_path), "files": files,
                "counts": dict(Counter(row["split"] for row in questions)), "eligible": len(questions), "source_files": len(indexes),
                "group_members": groups, "heldout_groups_available": len(groups) >= 3, "test_frozen": True,
                "normalization": "NFKC/whitespace used only for matching; prompt text preserved",
                "isolation": "software data-flow separation; not encryption or user access control"}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def verify_dataset(dataset, include_keys=False):
    dataset = inside_project(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("test_frozen") is not True:
        raise ValueError("不支援的資料 schema 或未凍結 test")
    files = [f"questions/{split}.jsonl" for split in SPLITS] + ["source_inventory.json", "source_index.json"]
    if include_keys:
        files.append("evaluator/keys.jsonl")
    for relative in files:
        if relative not in manifest["files"] or file_hash(dataset / relative) != manifest["files"][relative]:
            raise ValueError(f"資料 hash 不符：{relative}")
    return manifest


def load_questions(dataset, split):
    if split not in SPLITS:
        raise ValueError("split 必須是 train/dev/test")
    manifest = verify_dataset(dataset, include_keys=False)
    rows = read_jsonl(Path(dataset) / f"questions/{split}.jsonl")
    for row in rows:
        if (set(row) - QUESTION_FIELDS or row.get("schema_version") != SCHEMA_VERSION or row.get("split") != split
                or row.get("kind") != "mcq" or not isinstance(row.get("question"), str)
                or not isinstance(row.get("options"), dict) or set(row["options"]) != set("ABCD")
                or any(not isinstance(value, str) for value in row["options"].values())
                or set(row.get("source", {})) - {"file", "sha256", "page", "question_number", "year", "subject"}):
            raise ValueError("題目欄位洩漏／schema／split 不符")
        if row.get("question_sha256") != question_digest(row):
            raise ValueError("題目內容／來源 hash 不符")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("題目 id 重複")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.inventory, args.bank, args.out, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
