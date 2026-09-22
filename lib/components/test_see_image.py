import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.components import see_image as si


def test_usage_and_missing_and_bad_ext(tmp_path):
    assert "用法" in si.see_image("  ")
    assert "找不到" in si.see_image(str(tmp_path / "x.png"))
    f = tmp_path / "a.txt"; f.write_text("x")
    assert "不是支援" in si.see_image(str(f))


def test_report_uses_detections(tmp_path, monkeypatch):
    import numpy as np, cv2, types
    img = tmp_path / "a.png"; cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))
    monkeypatch.setattr(si, "PCB_DIR", tmp_path / "none")
    monkeypatch.setitem(sys.modules, "detector", types.SimpleNamespace(MODEL_NAME="x", detect=lambda im, conf: [{"label": "cat", "confidence": 0.9}]))
    out = si.see_image(str(img))
    assert "cat" in out and "a.png" in out and "尚未訓練" in out


def test_classifier_used_when_pcb_checkpoint_exists(tmp_path, monkeypatch):
    import numpy as np, cv2, types
    img = tmp_path / "a.png"; cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))
    (tmp_path / "pcb").mkdir(); (tmp_path / "pcb" / "best_model.pt").write_bytes(b"")
    monkeypatch.setattr(si, "PCB_DIR", tmp_path / "pcb")
    monkeypatch.setitem(sys.modules, "detector", types.SimpleNamespace(MODEL_NAME="x", detect=lambda im, conf: []))
    monkeypatch.setitem(sys.modules, "image_classifier_bnn", types.SimpleNamespace(
        classify_image=lambda p, out_dir: ("short", 0.93, 0.1), DEFAULT_MC_SAMPLES=1, DEFAULT_OUT_DIR=tmp_path))
    assert "short" in si.see_image(str(img)) and "93%" in si.see_image(str(img))
