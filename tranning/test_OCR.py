"""Pipeline smoke test for OCR.py.

Does NOT claim any recognition accuracy — there is no real OCR dataset yet.
Only proves the CRNN+CTC training loop (dataset loading, model, CTC loss,
early stopping, checkpoint + charset/config output) and the recognize()
inference path run end-to-end without crashing, using a handful of tiny
synthetic character-image pairs.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from OCR import recognize, train

TEXTS = ["12", "AB", "9X"]


def _make_manifest(images_dir: Path, prefix: str, repeats: int) -> list[dict]:
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for text in TEXTS:
        for i in range(repeats):
            img = Image.new("L", (100, 32), color=255)
            ImageDraw.Draw(img).text((5, 5), text, fill=0)
            name = f"{prefix}_{text}_{i}.png"
            img.save(images_dir / name)
            records.append({"image": name, "text": text})
    return records


def test_training_and_recognize_run_end_to_end(tmp_path):
    images_dir = tmp_path / "images"
    train_records = _make_manifest(images_dir, "train", repeats=6)
    val_records = _make_manifest(images_dir, "val", repeats=2)

    train_manifest = tmp_path / "train.json"
    val_manifest = tmp_path / "val.json"
    train_manifest.write_text(json.dumps(train_records), encoding="utf-8")
    val_manifest.write_text(json.dumps(val_records), encoding="utf-8")

    out_dir = tmp_path / "runs"
    model, charset, history = train(
        train_manifest=train_manifest, val_manifest=val_manifest, images_root=images_dir,
        out_dir=out_dir, epochs=2, batch_size=4, lr=1e-3, patience=5,
        img_height=32, img_width=100, hidden_size=32, dropout=0.1, weight_decay=1e-4,
    )

    assert set(charset) == set("".join(TEXTS))
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "charset.json").read_text(encoding="utf-8")) == charset
    assert all(h["val_cer"] >= 0.0 for h in history)

    sample_image = images_dir / f"train_{TEXTS[0]}_0.png"
    result = recognize(sample_image, out_dir=out_dir)
    assert isinstance(result, str)


def test_recognize_without_checkpoint_returns_placeholder(tmp_path):
    message = recognize(Path("does_not_matter.png"), out_dir=tmp_path / "no_checkpoint_here")
    assert "尚未訓練" in message
