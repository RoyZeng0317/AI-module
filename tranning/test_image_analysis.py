"""Pipeline smoke test for image_analysis.py.

detect() is monkeypatched (no real network/YOLO weights download in tests,
same reasoning as test_tools.py's recognize_image tests) — this only proves
analyze_image()/format_report() wire detect() + classify_image() together
correctly, not real detection/classification accuracy.
"""

import numpy as np
from PIL import Image

import image_analysis
from image_classifier_bnn import classify_image  # noqa: F401  (imported for monkeypatch target clarity)


def _make_image(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.fromarray(np.full((64, 64, 3), (200, 30, 30), dtype=np.uint8)).save(path)
    return path


def test_analyze_image_combines_detections_and_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(image_analysis, "detect", lambda image, conf, weights: [
        {"label": "cat", "confidence": 0.9, "box": [1.0, 2.0, 3.0, 4.0]},
    ])
    monkeypatch.setattr(image_analysis, "classify_image", lambda image_path, out_dir, mc_samples: ("cat", 0.8, 0.1))

    result = image_analysis.analyze_image(_make_image(tmp_path), classifier_dir=tmp_path / "no_checkpoint")

    assert result["detections"] == [{"label": "cat", "confidence": 0.9, "box": [1.0, 2.0, 3.0, 4.0]}]
    assert result["classification"] == {"label": "cat", "confidence": 0.8, "entropy": 0.1}


def test_format_report_includes_detections_and_classification():
    report = image_analysis.format_report({
        "detections": [{"label": "cat", "confidence": 0.9, "box": [0, 0, 1, 1]}],
        "classification": {"label": "cat", "confidence": 0.8, "entropy": 0.1},
    })
    assert "偵測到 1 個物件" in report
    assert "cat" in report
    assert "自訂分類" in report


def test_format_report_with_no_detections_and_untrained_classifier():
    report = image_analysis.format_report({
        "detections": [],
        "classification": {"label": "模型尚未訓練，請先準備好...", "confidence": 0.0, "entropy": 0.0},
    })
    assert "沒有偵測到任何已知物件" in report
    assert "尚未訓練" in report


def test_analyze_image_raises_on_unreadable_file(tmp_path):
    bad_path = tmp_path / "not_an_image.jpg"
    bad_path.write_text("not an image")
    try:
        image_analysis.analyze_image(bad_path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "無法讀取圖片" in str(exc)
