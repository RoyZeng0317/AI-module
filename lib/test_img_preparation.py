"""Preparation only: no optimizer, backward or training steps are executed."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest, torch
from PIL import Image
from lib.Image import img


@pytest.fixture
def random_checkpoint(tmp_path):
    """Random, untrained test fixture; never a learned capability artifact."""
    config = {"vocab": img.build_vocab(["紅色圓形"]), "latent_dim": 4, "image_size": 16, "cond_dim": 8, "test_only_untrained": True}
    with torch.random.fork_rng(devices=[]):
        model = img.TextToImageVAE(config["vocab"], 4, 16, 8)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "model.pt")
    return tmp_path


def test_seed_cold_warm_and_provenance(random_checkpoint):
    img._cache.clear()
    before = torch.random.get_rng_state().clone()
    first = img.generate(["紅色圓形"], random_checkpoint, seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    second = img.generate(["紅色圓形"], random_checkpoint, seed=7)
    assert first != second and first[0].read_bytes() == second[0].read_bytes()
    assert torch.equal(before, torch.random.get_rng_state())
    metadata = json.loads((first[0].parent / "samples.json").read_text(encoding="utf-8"))
    assert len(metadata["checkpoint_sha256"]) == 64
    assert metadata["samples"][0]["prompt"] == "紅色圓形"
    third = img.generate(["紅色圓形"], random_checkpoint, seed=8)
    assert first[0].read_bytes() != third[0].read_bytes()


def test_missing_model_no_draw(tmp_path):
    assert img.generate(["紅色圓形"], tmp_path) == []
    assert not (tmp_path / "samples").exists()


def test_stale_checkpoint_invalidates_cache(random_checkpoint):
    build = lambda c: img.TextToImageVAE(c["vocab"], 4, 16, 8)
    old, _ = img._load(random_checkpoint, "t2i", build)
    state = old.state_dict()
    state["text_encoder.embed.weight"] = state["text_encoder.embed.weight"] + 1
    torch.save(state, random_checkpoint / "model.pt")
    new, _ = img._load(random_checkpoint, "t2i", build)
    assert old is not new


def test_validation_config_without_training(tmp_path):
    for name, caption in [("train", "紅色圓形"), ("dev", "驗證")]:
        Image.new("RGB", (16, 16), "white").save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(json.dumps([{"image": f"{name}.png", "caption": caption}]), encoding="utf-8")
    with patch.object(img, "_split", side_effect=AssertionError("must not random split")), patch.object(img, "_train", return_value=[]) as run:
        model, _ = img.train_text_to_image(tmp_path / "train.json", tmp_path / "run", val_manifest=tmp_path / "dev.json", image_size=16, latent_dim=4, cond_dim=8, device="cpu")
    assert "驗" not in model.vocab
    assert len(run.call_args.args[1]) == len(run.call_args.args[2]) == 1
    with pytest.raises(FileExistsError):
        img.train_text_to_image(tmp_path / "train.json", tmp_path / "run")
    with pytest.raises(ValueError):
        img.train_text_to_image(tmp_path / "train.json", tmp_path / "invalid", epochs=0)

def test_budget_and_nan_stop_before_optimizer():
    model = torch.nn.Linear(1, 1)
    class ForbiddenOptimizer:
        def zero_grad(self):
            pass
        def step(self):
            raise AssertionError("no optimizer step authorized")
    loader = [(torch.zeros(1, 1), torch.zeros(1, 1))]
    with pytest.raises(InterruptedError):
        img._run_epoch(model, loader, ForbiddenOptimizer(), None, "cpu", True,
                       {"deadline": float("inf"), "max_steps": 0, "steps": 0})
    with pytest.raises(FloatingPointError):
        img._run_epoch(model, loader, ForbiddenOptimizer(), lambda *a: torch.tensor(float("nan")), "cpu", True)


def test_untrained_forward_shapes_only():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = img.TextToImageVAE(img.build_vocab(["紅色圓形"]), 4, 16, 8).eval()
        result, mu, logvar = model(torch.zeros(2, 3, 16, 16), ["紅色圓形"] * 2)
    assert result.shape == (2, 3, 16, 16) and mu.shape == logvar.shape == (2, 4)
    assert torch.isfinite(result).all()
