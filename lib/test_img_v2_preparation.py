"""Preparation only: no backward, optimizer or training execution is authorized."""
import hashlib, json
from pathlib import Path
from unittest.mock import patch
import pytest, torch
from PIL import Image
from lib import img

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def guard_updates():
    with patch.object(torch.Tensor, "backward", side_effect=AssertionError("no backward authorized")), patch.object(torch.optim.Adam, "step", side_effect=AssertionError("no step authorized")):
        yield

@pytest.mark.parametrize("kind", ["transpose", "resize_conv"])
def test_forward_raw_rgb_and_rng(kind):
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = img.TextToImageVAE(img.build_vocab(["紅色圓形"]), 32, 64, 64, kind).eval()
        before = torch.random.get_rng_state().clone()
        a = model.generate(["紅色圓形"], seed=42)
        b = model.generate(["紅色圓形"], seed=42)
        assert torch.equal(a, b) and torch.equal(before, torch.random.get_rng_state())
        assert a.shape == (1, 3, 64, 64) and 0 <= a.min() <= a.max() <= 1
        assert torch.unique(a).numel() > 100
        result, mu, logvar = model(torch.zeros(1, 3, 64, 64), ["紅色圓形"])
        assert result.shape == a.shape and mu.shape == logvar.shape == (1, 32)
        assert torch.isfinite(result).all()
        if kind == "resize_conv":
            assert not any(isinstance(layer, torch.nn.ConvTranspose2d) for layer in model.decoder.modules())
            assert sum(isinstance(layer, torch.nn.Upsample) for layer in model.decoder.modules()) == 4

def test_foreground_l1_exact_scale_target_only():
    # White pixel weight1, black pixel weight5; normalized weights1/3,5/3.
    target = torch.tensor([[[[1., 0.]], [[1., 0.]], [[1., 0.]]]], requires_grad=True)
    recon = torch.tensor([[[[0., 1.]], [[0., 1.]], [[0., 1.]]]])
    mu, logvar = torch.tensor([[2., 0.]]), torch.zeros(1, 2)
    loss, reconstruction, kl = img.foreground_l1_loss(recon, target, mu, logvar, beta=2)
    assert reconstruction.item() == pytest.approx(6) and kl.item() == pytest.approx(2)
    assert loss.item() == pytest.approx(10)
    duplicate = img.foreground_l1_loss(recon.repeat(2,1,1,1), target.repeat(2,1,1,1), mu.repeat(2,1), logvar.repeat(2,1), beta=2)
    assert duplicate[0].item() == pytest.approx(10)
    # No autograd path through the weight mask, while reconstruction still sees target.
    weight = 1 + 4 * (1 - target.detach()).amax(dim=1, keepdim=True)
    assert not weight.requires_grad

@pytest.mark.parametrize("kwargs", [{"decoder_kind":"bad"}, {"loss_kind":"bad"}, {"foreground_weight":float("nan")}, {"foreground_weight":float("inf")}, {"foreground_weight":-1}, {"architecture_version":3}, {"decoder_kind":"resize_conv", "architecture_version":1}])
def test_reject_contract(kwargs):
    with pytest.raises(ValueError):
        img._t2i_contract(**kwargs)

@pytest.fixture
def manifests(tmp_path):
    for split in ("train", "dev"):
        Image.new("RGB", (16,16), "white").save(tmp_path / f"{split}.png")
        (tmp_path / f"{split}.json").write_text(json.dumps([{"image":f"{split}.png", "caption":"紅色圓形"}]),encoding="utf-8")
    return tmp_path

def test_config_summary_mocked_no_training(manifests):
    out = manifests / "run"
    def fake_train(*args):
        args[-1]["steps"] = 9
        args[-1]["stop_reason"] = "mocked_stop"
        return []
    with patch.object(img, "_train", side_effect=fake_train):
        img.train_text_to_image(manifests / "train.json", out, val_manifest=manifests / "dev.json", image_size=16, latent_dim=4, cond_dim=8, device="cpu", decoder_kind="resize_conv", loss_kind="foreground_l1")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert (config["architecture_version"], config["decoder_kind"], config["loss_kind"], config["foreground_weight"]) == (2,"resize_conv","foreground_l1",4)
    assert config["train_manifest_sha256"] == hashlib.sha256((manifests / "train.json").read_bytes()).hexdigest()
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["steps"] == 9 and summary["stop_reason"] == "mocked_stop" and summary["status"] == "stopped"
    assert summary["device"] == "cpu" and summary["cuda_max_memory_allocated_bytes"] is None
    assert summary["run_elapsed_seconds"] >= summary["training_elapsed_seconds"] >= 0
    with pytest.raises(FileExistsError):
        img.train_text_to_image(manifests / "train.json", out)

@pytest.mark.parametrize("kwargs", [{"lr":float("nan")}, {"beta":float("inf")}, {"max_seconds":float("nan")}, {"epochs":float("inf")}, {"foreground_weight":-1}])
def test_invalid_before_training(manifests,kwargs):
    with patch.object(img, "_train", side_effect=AssertionError("no train")), pytest.raises(ValueError):
        img.train_text_to_image(manifests / "train.json", manifests / "bad",**kwargs)

@pytest.mark.parametrize("field,value", [("decoder_kind","unknown"),("loss_kind","unknown"),("architecture_version",99),("architecture_version",2)])
def test_invalid_config_load(tmp_path,field,value):
    config = {"vocab":{},"latent_dim":4,"image_size":16,"cond_dim":8,field:value}
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save({},tmp_path / "model.pt")
    with pytest.raises(ValueError):
        img.generate(["紅色圓形"],tmp_path)


def test_v2_fixture_cold_warm_api(manifests):
    config = {"vocab": img.build_vocab(["紅色圓形"]),"latent_dim":4,"image_size":16,"cond_dim":8,"decoder_kind":"resize_conv","loss_kind":"foreground_l1","architecture_version":2,"foreground_weight":4}
    with torch.random.fork_rng(devices=[]):
        model = img.TextToImageVAE(config["vocab"],4,16,8,"resize_conv")
    (manifests / "config.json").write_text(json.dumps(config))
    torch.save(model.state_dict(),manifests / "model.pt")
    img._cache.clear()
    before = torch.random.get_rng_state().clone()
    first = img.generate(["紅色圓形"],manifests,seed=42)
    second = img.generate(["紅色圓形"],manifests,seed=42)
    assert first[0].read_bytes() == second[0].read_bytes() and torch.equal(before,torch.random.get_rng_state())
    config["decoder_kind"],config["architecture_version"] = "transpose",1
    (manifests / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError):
        img.generate(["紅色圓形"],manifests,seed=42)


def test_real_v1_exact_png_regression(tmp_path):
    run = ROOT / "lib/img_runs/t2i_shapes/pilot_v1"
    before = hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()
    assert before == "7ed7696177a9e44dd3089b81f9208d05ef4d37eecea9efdf1c6060b6de7ec0ae"
    records = ROOT / "output/imagegen-shapes/pilot_v1/raw"
    reference = None
    for path in records.glob("*/samples.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["seed"] == 42 and record["samples"][0]["prompt"] == "紅色圓形":
            reference = path.parent / record["samples"][0]["image"]
            break
    assert reference is not None
    img._cache.clear()
    before_rng = torch.random.get_rng_state().clone()
    result = img.generate(["紅色圓形"],run,save_to=tmp_path,seed=42)
    assert result[0].read_bytes() == reference.read_bytes()
    assert torch.equal(before_rng,torch.random.get_rng_state())
    assert hashlib.sha256((run / "model.pt").read_bytes()).hexdigest() == before


def test_editor_deblur_fixture_forward_and_api(manifests):
    config = {"vocab": img.build_vocab(["make blue"]),"latent_dim":4,"image_size":16,"cond_dim":8}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        editor, deblur = img.ImageEditor(config["vocab"],4,16,8).eval(), img.DeblurNet().eval()
        x = torch.ones(1,3,16,16)
        assert editor.edit(x,["make blue"]).shape == deblur(x).shape == x.shape
        assert isinstance(editor.decoder,img.CondDecoder)
    for name,model in (("editor",editor),("deblur",deblur)):
        run = manifests / name
        run.mkdir()
        (run / "config.json").write_text(json.dumps(config))
        torch.save(model.state_dict(),run / "model.pt")
    assert img.edit(manifests / "train.png","make blue",manifests / "editor").exists()
    assert img.deblur(manifests / "train.png",manifests / "deblur").exists()
