"""Download verified sources and build isolated Sinco emotion training sets.

No model APIs or training are used. Chat pairs are English; CPED (converted
with OpenCC) and GoEmotions are kept as separate classification label spaces.
The existing data/pairs.json is read, never overwritten. See the data README.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "emotion_datasets"
STRATEGIES = {"Question", "Affirmation and Reassurance", "Reflection of feelings",
              "Restatement or Paraphrasing"}
SPLITS = ("test", "val", "train")  # held-out texts take precedence over training
SPLIT_NAMES = {"train": "train", "valid": "val", "validation": "val", "test": "test"}


def clean(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def key(text):
    return clean(text).casefold()


def hash_split(group, seed=42):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "val" if bucket < 20 else "train"


def download_sources(manifest_path, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        filename = record["filename"]
        if Path(filename).name != filename:
            raise ValueError("Source filenames must not contain directories")
        path = raw_dir / filename
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(record["url"], timeout=90) as response, temporary.open("wb") as output:
                digest, size = hashlib.sha256(), 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > record["bytes"]:
                        raise ValueError(f"Unexpected source size: {filename}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Source checksum mismatch: {filename}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {filename}", flush=True)
    return manifest


def pair(prompt, reply, source, conversation_id, split, **metadata):
    return dict(prompt=clean(prompt), reply=clean(reply), source=source,
                conversation_id=f"{source}:{conversation_id}", split=split, **metadata)


def esconv_pairs(conversations, seed=42):
    output = []
    for conversation in conversations:
        turns = conversation["dialog"]
        identity = [(t["speaker"], clean(t["content"])) for t in turns]
        group = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        split = hash_split(group, seed)
        # Merge consecutive turns by the same role, including seeker follow-ups.
        blocks = []
        for turn in turns:
            role = turn["speaker"]
            if not blocks or blocks[-1]["role"] != role:
                blocks.append({"role": role, "texts": [], "strategies": []})
            blocks[-1]["texts"].append(turn["content"])
            if role == "supporter":
                blocks[-1]["strategies"].append(turn.get("annotation", {}).get("strategy", ""))
        for index, (question, response) in enumerate(zip(blocks, blocks[1:])):
            if question["role"] != "seeker" or response["role"] != "supporter":
                continue
            strategies = response["strategies"]
            if not strategies or any(s not in STRATEGIES for s in strategies):
                continue
            prompt, reply = " ".join(question["texts"]), " ".join(response["texts"])
            if len(clean(prompt)) < 20:  # greeting/short feedback is not an emotional disclosure
                continue
            output.append(pair(prompt, reply, "esconv", group, split, language="en",
                               turn=index, emotion=conversation.get("emotion_type"), strategies=strategies))
    return output


def empathetic_pairs(rows, split):
    groups = defaultdict(list)
    for row in rows:
        groups[row["conv_id"]].append(row)
    output = []
    for group, turns in groups.items():
        turns.sort(key=lambda row: int(row["utterance_idx"]))
        if len(turns) < 2:
            continue
        # Only the initiating disclosure and first listener response. Later
        # replies often require history which the current Sinco path ignores.
        first, second = turns[:2]
        if first["speaker_idx"] == second["speaker_idx"]:
            continue
        output.append(pair(first["utterance"].replace("_comma_", ","),
                           second["utterance"].replace("_comma_", ","),
                           "empathetic", group, split, language="en", emotion=first["context"]))
    return output


def dailydialog_pairs(dialogues):
    output = []
    for dialogue in dialogues:
        split = SPLIT_NAMES[dialogue["data_split"]]
        for first, second in zip(dialogue["turns"], dialogue["turns"][1:]):
            if first["speaker"] != "user" or second["speaker"] != "system":
                continue
            if first["emotion"] == "no emotion" or second["emotion"] in {"anger", "disgust"}:
                continue
            output.append(pair(first["utterance"], second["utterance"], "dailydialog",
                               dialogue["dialogue_id"], split, language="en", emotion=first["emotion"],
                               turn=first["utt_idx"]))
    return output


def cped_records(rows, split, convert):
    return [dict(text=clean(convert(row["Utterance"])), labels=[row["Emotion"]],
                 sentiment=row["Sentiment"], dialogue_act=row["DA"], source="cped",
                 label_space="cped_emotion", language="zh-Hant", split=split,
                 conversation_id=f"cped:{row['TV_ID']}:{row['Dialogue_ID']}",
                 utterance_id=row["Utterance_ID"], speaker=row["Speaker"])
            for row in rows if clean(row["Utterance"])]


def goemotions_records(rows, labels, split):
    return [dict(text=clean(text), labels=[labels[int(i)] for i in ids.split(",")],
                 source="goemotions", label_space="goemotions", language="en", split=split,
                 conversation_id=f"goemotions:{identifier}")
            for text, ids, identifier in rows if clean(text)]


def isolate(records, text_field):
    """Drop cross-split text leakage without moving a source conversation.

    Repeated prompts within a split retain distinct replies. Duplicate full
    entries are collapsed. Entire conversation groups appearing in multiple
    splits are rejected rather than quietly splitting their turns.
    """
    by_split = {split: [] for split in SPLITS}
    groups, stats = {}, Counter()
    for record in records:
        split, group = record["split"], record["conversation_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Conversation crosses source splits: {group}")
        groups[group] = split
        by_split[split].append(record)
    previous_texts = set()
    for split in SPLITS:
        kept, texts, entries = [], set(), set()
        for record in by_split[split]:
            text = key(record[text_field])
            identity = (text, key(record["reply"])) if text_field == "prompt" else (text, tuple(record["labels"]))
            if text in previous_texts:
                stats["cross_split_texts_removed"] += 1
                continue
            if identity in entries:
                stats["duplicates_removed"] += 1
                continue
            kept.append(record)
            texts.add(text)
            entries.add(identity)
        by_split[split] = kept
        previous_texts.update(texts)
    return by_split, dict(stats)


def acceptable_pair(record, max_chars):
    prompt, reply = record["prompt"], record["reply"]
    # Do not silently train on truncated replies. Character count is a safe
    # upper bound for this repository's character-based BPE tokenization.
    return (len(prompt) >= 5 and len(reply) >= 5 and len(prompt) + len(reply) + 2 <= max_chars
            and key(prompt) != key(reply))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(raw_dir, output_dir, legacy_path, seed=42, max_chars=480):
    from opencc import OpenCC
    convert = OpenCC("s2t").convert
    # Immutable archive members are read directly: no extractall/path traversal.
    sources = esconv_pairs(json.loads((raw_dir / "esconv.json").read_text(encoding="utf-8")), seed)
    with tarfile.open(raw_dir / "empathetic.tar.gz") as archive:
        for name in ("train", "valid", "test"):
            member = archive.extractfile(f"empatheticdialogues/{name}.csv")
            if member is None:
                raise ValueError(f"Missing EmpatheticDialogues {name} split")
            rows = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"))
            sources.extend(empathetic_pairs(rows, SPLIT_NAMES[name]))
    with zipfile.ZipFile(raw_dir / "dailydialog.zip") as archive:
        sources.extend(dailydialog_pairs(json.loads(archive.read("data/dialogues.json"))))
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    for index, row in enumerate(legacy):
        group = hashlib.sha256(key(row["prompt"]).encode()).hexdigest()
        sources.append(pair(row["prompt"], row["reply"], "sinco_existing", group,
                            hash_split(group, seed), language="zh-Hant" if re.search(r"[\u4e00-\u9fff]", row["prompt"]) else "en",
                            original_index=index))
    filtered = [r for r in sources if acceptable_pair(r, max_chars)]
    pairs, dropped = isolate(filtered, "prompt")
    report = {"seed": seed, "max_chars": max_chars, "raw_candidate_pairs": len(sources),
              "length_or_content_filtered": len(sources) - len(filtered), "deduplication": dropped,
              "pairs": {}, "classification": {}, "legacy_source_sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
              "notes": ["External chat pairs are English; no English-to-Chinese translation was performed.",
                        "CPED and GoEmotions are separate classification label spaces, not chat replies.",
                        "ESConv and legacy use deterministic conversation/prompt hash splits; others keep published splits.",
                        "Pre-existing Sinco pairs may already have appeared in pretraining; new external test files must stay out of pretraining/rehearsal."]}
    for split, rows in pairs.items():
        write_json(output_dir / f"pairs_{split}.json", rows)
        external = [r for r in rows if r["source"] != "sinco_existing"]
        write_json(output_dir / f"external_pairs_{split}.json", external)
        report["pairs"][split] = {"count": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
                                  "by_language": dict(Counter(r["language"] for r in rows))}
    labels = (raw_dir / "goemotions_labels.txt").read_text().splitlines()
    classification = {"cped": [], "goemotions": []}
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        with (raw_dir / f"cped_{name}.csv").open(encoding="utf-8-sig", newline="") as source:
            classification["cped"].extend(cped_records(csv.DictReader(source), split, convert))
    for split, name in (("train", "train"), ("val", "dev"), ("test", "test")):
        with (raw_dir / f"goemotions_{name}.tsv").open(encoding="utf-8", newline="") as source:
            classification["goemotions"].extend(goemotions_records(csv.reader(source, delimiter="\t"), labels, split))
    for dataset, rows in classification.items():
        isolated, counts = isolate(rows, "text")
        report["classification"][dataset] = {"deduplication": counts, "splits": {s: len(r) for s, r in isolated.items()}}
        for split, entries in isolated.items():
            write_jsonl(output_dir / "classification" / f"{dataset}_{split}.jsonl", entries)
    license_dir = output_dir / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    for source in raw_dir.glob("*_LICENSE.txt"):
        (license_dir / source.name).write_bytes(source.read_bytes().rstrip(b"\r\n") + b"\n")
    write_json(output_dir / "import_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def validate_outputs(output_dir, tokenizer_dir=None, block_size=512):
    """Verify actual split isolation and optionally the existing BPE limits."""
    report = {"pairs": {}, "classification": {}, "cross_split_prompt_overlap": 0,
              "cross_split_conversation_overlap": 0}
    tokenizer = None
    token_counts, unknown_counts = Counter(), Counter()
    if tokenizer_dir is not None:
        from bpe_tokenizer import BPETokenizer, UNK
        tokenizer = BPETokenizer.load(tokenizer_dir)
    tasks = {"pairs": [output_dir / f"pairs_{s}.json" for s in ("train", "val", "test")]}
    for source in ("cped", "goemotions"):
        tasks[source] = [output_dir / "classification" / f"{source}_{s}.jsonl" for s in ("train", "val", "test")]
    for task, paths in tasks.items():
        texts, groups, counts = {}, {}, {}
        for split, path in zip(("train", "val", "test"), paths):
            rows = json.loads(path.read_text(encoding="utf-8")) if task == "pairs" else [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            field = "prompt" if task == "pairs" else "text"
            if any(not r[field] or r["split"] != split for r in rows):
                raise ValueError(f"Invalid {task}/{split} row")
            texts[split] = {key(r[field]) for r in rows}
            groups[split] = {r["conversation_id"] for r in rows}
            counts[split] = len(rows)
            for row in rows:
                if task != "pairs":
                    if not row["labels"]:
                        raise ValueError("Empty classification label")
                elif tokenizer is not None:
                    ids = tokenizer.encode(row["prompt"]) + tokenizer.encode(row["reply"])
                    if len(ids) + 2 > block_size:
                        raise ValueError("Pair exceeds model block size")
                    token_counts[row["source"]] += len(ids)
                    unknown_counts[row["source"]] += ids.count(UNK)
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if texts[first] & texts[second] or groups[first] & groups[second]:
                raise ValueError(f"Cross-split leakage in {task}")
        if task == "pairs":
            report["pairs"] = counts
        else:
            report["classification"][task] = counts
    if tokenizer is not None:
        report["tokenizer_check"] = {"tokenizer_dir": str(tokenizer_dir), "block_size": block_size,
                                     "overlength_pairs": 0, "by_source": {
            source: {"tokens": n, "unk_tokens": unknown_counts[source], "unk_rate": unknown_counts[source] / n}
            for source, n in token_counts.items()}}
    write_json(output_dir / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--legacy-data", type=Path, default=ROOT / "data" / "pairs.json")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="optional existing BPE checkpoint for length/UNK checks")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=480)
    args = parser.parse_args()
    manifest_path = args.source_dir / "source_downloads.json"
    download_sources(manifest_path, args.source_dir / "raw")
    output = args.output_dir or args.source_dir
    build(args.source_dir / "raw", output, args.legacy_data, args.seed, args.max_chars)
    validate_outputs(output, args.tokenizer_dir, args.block_size)


if __name__ == "__main__":
    main()
