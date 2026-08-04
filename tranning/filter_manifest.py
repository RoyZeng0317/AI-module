"""Filter manifest to only include clips with text labels.

Creates train_clean.json / val_clean.json with only labeled clips.
"""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

data_dir = Path(__file__).resolve().parent / "speech_data"

for split in ["train", "val"]:
    src = data_dir / f"{split}.json"
    dst = data_dir / f"{split}_clean.json"
    records = json.loads(src.read_text(encoding="utf-8"))
    labeled = [r for r in records if r.get("text") and r["text"].strip()]
    dst.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{split}.json: {len(records)} total -> {len(labeled)} labeled -> {dst.name}")
