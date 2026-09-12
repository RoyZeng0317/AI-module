"""Pipeline smoke test for img.py.

Does NOT claim any generation/editing quality — no real captioned-image or
before/after dataset exists yet. Only proves both training loops and both
inference functions run end-to-end without crashing, using synthetic solid-
color squares (same spirit as tranning/imagegen/test_draw.py).
"""

import json

from PIL import Image, ImageDraw

from img import deblur, edit, generate, train_deblur, train_editor, train_text_to_image

CAPTIONS = ["a red circle", "a blue circle", "a green circle"]


def _make_image(path, color):
    image = Image.new("RGB", (48, 48), color=color)
    ImageDraw.Draw(image).ellipse((10, 10, 38, 38), fill=(255, 255, 255))
    image.save(path)


def _make_t2i_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    entries = []
    colors = [(220, 60, 60), (60, 160, 220), (60, 200, 100)]
    for i, (caption, color) in enumerate(zip(CAPTIONS * 4, colors * 4)):
        name = f"{i:03d}.png"
        _make_image(image_dir / name, color)
        entries.append({"image": f"images/{name}", "caption": caption})
    manifest = tmp_path / "captions.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return manifest


def _make_edit_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    entries = []
    for i in range(12):
        before, after = f"before_{i:03d}.png", f"after_{i:03d}.png"
        _make_image(image_dir / before, (220, 60, 60))
        _make_image(image_dir / after, (60, 160, 220))
        entries.append({"source": f"images/{before}", "instruction": "make it blue", "target": f"images/{after}"})
    manifest = tmp_path / "edits.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return manifest


def test_train_text_to_image_runs_end_to_end(tmp_path):
    manifest = _make_t2i_manifest(tmp_path)
    out_dir = tmp_path / "t2i_runs"

    model, history = train_text_to_image(manifest, out_dir=out_dir, epochs=2, batch_size=4,
                                          latent_dim=16, image_size=32)

    assert len(history) == 2
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.json").exists()


def test_generate_without_checkpoint_returns_empty():
    assert generate(["a red circle"], out_dir="no_such_run_dir") == []


def test_generate_after_training_produces_images(tmp_path):
    manifest = _make_t2i_manifest(tmp_path)
    out_dir = tmp_path / "t2i_runs"
    train_text_to_image(manifest, out_dir=out_dir, epochs=1, batch_size=4, latent_dim=16, image_size=32)

    paths = generate(["a red circle", "a blue circle"], out_dir=out_dir, save_to=tmp_path / "samples")
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_train_editor_runs_end_to_end(tmp_path):
    manifest = _make_edit_manifest(tmp_path)
    out_dir = tmp_path / "edit_runs"

    model, history = train_editor(manifest, out_dir=out_dir, epochs=2, batch_size=4,
                                   latent_dim=16, image_size=32)

    assert len(history) == 2
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.json").exists()


def test_edit_without_checkpoint_returns_none():
    assert edit("no_such_image.png", "make it blue", out_dir="no_such_run_dir") is None


def test_edit_after_training_produces_image(tmp_path):
    manifest = _make_edit_manifest(tmp_path)
    out_dir = tmp_path / "edit_runs"
    train_editor(manifest, out_dir=out_dir, epochs=1, batch_size=4, latent_dim=16, image_size=32)

    source = tmp_path / "probe.png"
    _make_image(source, (220, 60, 60))
    result = edit(source, "make it blue", out_dir=out_dir, save_to=tmp_path / "edited.png")
    assert result is not None and result.exists()


def _make_sharp_photo_folder(tmp_path):
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    colors = [(220, 60, 60), (60, 160, 220), (60, 200, 100), (230, 200, 60)]
    for i, color in enumerate(colors * 3):  # 12 images, no manifest needed (blur is synthesized)
        _make_image(photo_dir / f"{i:03d}.png", color)
    return photo_dir


def test_train_deblur_runs_end_to_end_from_a_plain_photo_folder(tmp_path):
    photo_dir = _make_sharp_photo_folder(tmp_path)
    out_dir = tmp_path / "deblur_runs"

    model, history = train_deblur(photo_dir, out_dir=out_dir, epochs=2, batch_size=4, image_size=32)

    assert len(history) == 2
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.json").exists()


def test_deblur_without_checkpoint_returns_none():
    assert deblur("no_such_image.png", out_dir="no_such_run_dir") is None


def test_deblur_after_training_produces_image(tmp_path):
    photo_dir = _make_sharp_photo_folder(tmp_path)
    out_dir = tmp_path / "deblur_runs"
    train_deblur(photo_dir, out_dir=out_dir, epochs=1, batch_size=4, image_size=32)

    probe = tmp_path / "probe.png"
    _make_image(probe, (220, 60, 60))
    result = deblur(probe, out_dir=out_dir, save_to=tmp_path / "deblurred.png")
    assert result is not None and result.exists()
