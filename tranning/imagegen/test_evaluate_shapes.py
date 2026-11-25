"""Synthetic fixtures verify orchestration only; these are NOT quality evidence."""
import json, shutil, tempfile
from pathlib import Path
import pytest
from PIL import Image
from tranning.imagegen import evaluate_shapes as module

def test_stage_seeds_match_locked_proposal():
    assert module.SEEDS == {"dev": [42, 43, 44], "confirm": [101, 102, 103], "test": [42, 43, 44]}

@pytest.fixture
def fixture_area(monkeypatch):
    base = module.ROOT / "output/imagegen-shapes-v2-tests"
    base.mkdir(parents=True, exist_ok=True)
    area = Path(tempfile.mkdtemp(dir=base))
    dataset, model = area / "dataset", area / "run"
    dataset.mkdir(); model.mkdir()
    (model / "model.pt").write_bytes(b"FIXTURE ONLY no model")
    module.write(model / "config.json", {"fixture": True})
    for split, combos in (("train", module.TRAIN), ("dev", module.DEV), ("test", module.TEST)):
        rows = []
        for combo in sorted(combos):
            color, shape = combo.split("_")
            rows.append({"caption": combo, "color": color, "shape": shape, "combo": combo, "split": split})
        module.write(dataset / f"{split}.json", rows)
    hashes = {f"{split}.json": module.sha(dataset / f"{split}.json") for split in ("train", "dev")}
    module.write(model / "config.json", {f"{split}_manifest_sha256": hashes[f"{split}.json"] for split in ("train", "dev")})
    monkeypatch.setattr(module, "LOCKED_SOURCES", {**hashes, "scorer_sha256": module.sha(Path(module.__file__).with_name("shapes_score.py"))})
    yield area, dataset, model
    shutil.rmtree(area)

def fake_generate(prompts, out_dir, save_to, seed):
    # White fixture deliberately cannot pass the real frozen scorer.
    directory = save_to / f"{prompts[0]}_{seed}"
    directory.mkdir()
    path = directory / "raw.png"
    Image.new("RGB", (64, 64), "white").save(path)
    return [path]

def fixture_score(manifest, success=True):
    samples = module.read(manifest)["samples"]
    details, combos = [], {}
    for sample in samples:
        target = sample["target"]
        expected = {k: target[k] for k in ("color", "shape")}
        details.append({"image": sample["image"], "combo": target["combo"], "split": target["split"], "expected": expected,
                        "prediction": {**expected, "scorable": success}, "joint_correct": success})
        combos[target["combo"]] = {"count": 3, "joint_correct": 3 if success else 0, "joint_accuracy": 1. if success else 0.}
    splits = {}
    for split in {d["split"] for d in details}:
        count = sum(d["split"] == split for d in details)
        splits[split] = {"count": count, "joint_correct": count if success else 0, "joint_accuracy": 1. if success else 0.}
    return {"details": details, "by_combo": combos, "by_split": splits}

def execute(area, dataset, model, stage="dev", previous=None, score_fn=fixture_score):
    return module.run(model, area / stage, stage, previous, dataset, fake_generate, score_fn)

@pytest.mark.parametrize("mutation", ["train", "dev", "scorer", "config_train", "config_dev"])
def test_rejects_source_drift_before_first_dev(fixture_area, monkeypatch, mutation):
    area, dataset, model = fixture_area
    original_read, original_sha = module.read, module.sha
    def guarded_read(path):
        assert Path(path).name != "test.json", "Sealed test was read"
        return original_read(path)
    def no_generate(*args, **kwargs):
        pytest.fail("Source drift must be rejected before generation")
    if mutation in ("train", "dev"):
        with (dataset / f"{mutation}.json").open("a", encoding="utf-8") as file: file.write("\n")
    elif mutation == "scorer":
        monkeypatch.setattr(module, "sha", lambda path: "changed" if Path(path).name == "shapes_score.py" else original_sha(path))
    else:
        config = original_read(model / "config.json")
        config[f"{mutation.removeprefix('config_')}_manifest_sha256"] = "changed"
        module.write(model / "config.json", config)
    monkeypatch.setattr(module, "read", guarded_read)
    with pytest.raises(ValueError, match="source mismatch"):
        module.run(model, area / "dev", dataset=dataset, generate_fn=no_generate)
    assert not (area / "dev").exists()


def test_fixture_stages_and_raw_files(fixture_area):
    area, dataset, model = fixture_area
    dev = execute(*fixture_area)
    assert dev["gate_passed"] and len(dev["scores"]["details"]) == 30
    confirm = execute(*fixture_area, "confirm", area / "dev/summary.json")
    assert confirm["seeds"] == [101, 102, 103] and len(confirm["scores"]["details"]) == 6
    result = execute(*fixture_area, "test", area / "confirm/summary.json")
    assert result["seeds"] == [42, 43, 44]
    assert result["gate_passed"] and len(result["scores"]["details"]) == 6
    assert (area / "test/contact_sheet.png").is_file()
    manifest = module.read(area / "dev/samples.json")
    assert all(not Path(s["image"]).is_absolute() for s in manifest["samples"])

def test_fail_gate_does_not_read_test(fixture_area, monkeypatch):
    area, dataset, model = fixture_area
    execute(*fixture_area, score_fn=lambda m: fixture_score(m, False))
    original = module.read
    def guard(path):
        assert Path(path).name != "test.json", "Sealed test was read"
        return original(path)
    monkeypatch.setattr(module, "read", guard)
    with pytest.raises(ValueError, match="stage/checkpoint/source mismatch"):
        execute(*fixture_area, "test", area / "dev/summary.json")
    with pytest.raises(ValueError, match="gate failed"):
        execute(*fixture_area, "confirm", area / "dev/summary.json")
    assert not (area / "test").exists()

@pytest.mark.parametrize("mutation", ["nan", "seed", "count", "combo", "checkpoint", "raw"])
def test_rejects_tampered_proof(fixture_area, mutation):
    area, dataset, model = fixture_area
    execute(*fixture_area)
    path = area / "dev/summary.json"
    result = module.read(path)
    if mutation == "nan": result["scores"]["by_combo"]["red_circle"]["joint_accuracy"] = float("nan")
    elif mutation == "seed": result["seeds"] = [1, 2, 3]
    elif mutation == "count": result["scores"]["by_combo"]["red_circle"]["count"] = 2
    elif mutation == "combo": result["scores"]["by_combo"].pop("red_circle")
    elif mutation == "checkpoint": (model / "model.pt").write_bytes(b"changed")
    elif mutation == "raw":
        sample = module.read(area / "dev/samples.json")["samples"][0]
        (area / "dev" / sample["image"]).write_bytes(b"changed")
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        execute(*fixture_area, "confirm", path)

def test_unscorable_failure_and_overwrite(fixture_area):
    result = execute(*fixture_area, score_fn=module.evaluate)
    assert not result["gate_passed"]
    with pytest.raises(ValueError, match="nonempty"):
        execute(*fixture_area)

def test_missing_checkpoint_and_invalid_stage(fixture_area):
    area, dataset, model = fixture_area
    with pytest.raises(ValueError, match="Invalid stage"):
        execute(*fixture_area, "unknown")
    (model / "model.pt").unlink()
    with pytest.raises(FileNotFoundError): execute(*fixture_area)
