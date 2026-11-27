import json, shutil
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from tranning.imagegen.shapes_data import build, COLORS
from tranning.imagegen.shapes_score import infer, evaluate

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def folder():
    path = ROOT / "output/imagegen-shapes-tests"
    path.mkdir(parents=True, exist_ok=True)
    child = path / "unit"
    child.mkdir(exist_ok=False)
    yield child
    shutil.rmtree(child)

def test_dataset(folder):
    spec = build(folder / "data", variants=3)
    assert spec["counts"] == {"train": 24, "dev": 6, "test": 6}
    groups = spec["combinations"]
    assert not set(groups["train"]) & set(groups["dev"] + groups["test"])
    assert spec["unique_pixel_hashes"] == 36
    with pytest.raises(FileExistsError): build(folder / "data")

@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
@pytest.mark.parametrize("color", list(COLORS))
def test_independent_pixels(folder, shape, color):
    # Analytic masks, independent from PIL drawing used by the teaching renderer.
    y,x = np.mgrid[:64,:64]; dx,dy = x-33,y-31
    if shape == "circle": mask = dx*dx+dy*dy <= 16*16
    elif shape == "square": mask = (abs(dx)<=16)&(abs(dy)<=16)
    else: mask = (dy>=-16)&(dy<=16)&(abs(dx)<=(dy+16)/2)
    pixels = np.full((64,64,3),255,dtype=np.uint8); pixels[mask] = COLORS[color]
    path = folder / "sample.png"; Image.fromarray(pixels).save(path)
    result = infer(path)
    assert result["scorable"] and (result["color"],result["shape"]) == (color,shape)

def test_failures_and_wrong_targets(folder):
    image = Image.new("RGB",(64,64),"white")
    path = folder / "sample.png"; image.save(path)
    assert not infer(path)["scorable"]
    Image.new("RGB", (64,64), COLORS["red"]).save(path)
    assert not infer(path)["scorable"]
    rng = np.random.default_rng(17)
    Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(path)
    assert not infer(path)["scorable"]
    ImageDraw.Draw(image).rectangle((16,16,47,47),fill=COLORS["red"])
    image.filter(ImageFilter.GaussianBlur(2)).save(path)
    assert not infer(path)["scorable"]
    image.save(path)
    manifest = folder / "samples.json"
    manifest.write_text(json.dumps({"samples": [{"image":"sample.png","prompt":"藍色圓形"}]}),encoding="utf-8")
    result = evaluate(manifest)
    assert result["groups"]["seen"]["joint_correct"] == 0
    assert not result["details"][0]["color_correct"] and not result["details"][0]["shape_correct"]

def test_metadata_unscorable_stays_in_denominator(folder):
    path = folder / "blank.png"
    Image.new("RGB",(64,64),"white").save(path)
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps([{"image":"blank.png","metadata":{"color":"green","shape":"circle","split":"test"}}]),encoding="utf-8")
    counters = evaluate(manifest)["groups"]["unseen"]
    assert counters["count"] == counters["unscorable"] == 1
    assert counters["joint_accuracy"] == 0

def test_separate_dev_test_and_combo(folder):
    image = Image.new("RGB", (64,64), "white")
    ImageDraw.Draw(image).rectangle((17,17,46,46), fill=COLORS["blue"])
    image.save(folder / "square.png")
    manifest = folder / "samples.json"
    rows = [{"image":"square.png", "prompt":"藍色正方形"}, {"image":"square.png", "prompt":"綠色圓形"}]
    manifest.write_text(json.dumps({"samples": rows}), encoding="utf-8")
    result = evaluate(manifest)
    assert result["groups"]["unseen"]["joint_accuracy"] == .5
    assert result["by_split"]["dev"]["joint_accuracy"] == 1
    assert result["by_split"]["test"]["joint_accuracy"] == 0
    assert result["by_combo"]["blue_square"]["joint_correct"] == 1
    assert result["by_combo"]["green_circle"]["joint_correct"] == 0
    assert result["details"][1]["expected"] == {"color":"green", "shape":"circle"}
    assert result["details"][1]["split"] == "test" and result["details"][1]["combo"] == "green_circle"
    rows[1]["metadata"] = {"color":"green", "shape":"circle", "combo":"blue_square"}
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="combo disagrees"): evaluate(manifest)
    rows[1]["metadata"].pop("combo")
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    assert evaluate(manifest)["by_split"]["test"]["joint_accuracy"] == 0
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"): evaluate(manifest)
