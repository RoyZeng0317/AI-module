"""Pipeline smoke test for draw.py.

This does NOT claim any generation quality — there is no real image dataset
yet. It only proves the VAE training loop and sampling both run end-to-end
without crashing, using a handful of synthetic solid-color squares (drawn
with PIL, same spirit as OCR.py's synthetic "12"/"AB"/"9X" text images).
"""

from PIL import Image, ImageDraw

from draw import generate, train

COLORS = [(220, 60, 60), (60, 160, 220), (60, 200, 100), (230, 200, 60), (180, 90, 220), (240, 240, 240)]


def _make_synthetic_images(dir_path):
    for i, color in enumerate(COLORS * 3):  # 18 images total
        image = Image.new("RGB", (48, 48), color=color)
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 38, 38), fill=(255, 255, 255))
        image.save(dir_path / f"{i:03d}.png")


def test_training_loop_runs_end_to_end(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_synthetic_images(image_dir)
    out_dir = tmp_path / "runs"

    model, history = train(
        train_dir=image_dir, out_dir=out_dir, epochs=2, batch_size=4,
        latent_dim=16, image_size=32, beta=1.0, lr=1e-3, val_split=0.2, hflip=True,
    )

    assert len(history) == 2
    assert (out_dir / "vae.pt").exists()
    assert (out_dir / "config.json").exists()
    assert all(h["train_loss"] >= 0 for h in history)


def test_generate_without_checkpoint_returns_empty():
    paths = generate(3, out_dir="no_such_run_dir")
    assert paths == []


def test_generate_after_training_produces_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_synthetic_images(image_dir)
    out_dir = tmp_path / "runs"

    train(train_dir=image_dir, out_dir=out_dir, epochs=1, batch_size=4,
          latent_dim=16, image_size=32, beta=1.0, lr=1e-3, val_split=0.2, hflip=True)

    paths = generate(3, out_dir=out_dir, save_to=tmp_path / "samples")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)
