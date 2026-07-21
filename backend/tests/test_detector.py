"""Smoke + correctness tests for the YOLO wrapper.

Uses the sample image bundled with the `ultralytics` package itself (a bus
with pedestrians) so the test needs no network access beyond what pip
install already required, and no webcam/display.
"""

from pathlib import Path

import cv2

from detector import detect


def _bundled_sample() -> Path:
    import ultralytics
    path = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
    assert path.exists(), f"expected sample image at {path}"
    return path


def test_detects_known_classes_in_sample_image():
    image = cv2.imread(str(_bundled_sample()))
    detections = detect(image, conf=0.35)

    labels = {d["label"] for d in detections}
    assert "bus" in labels
    assert "person" in labels
    assert all(0.0 <= d["confidence"] <= 1.0 for d in detections)
    assert all(len(d["box"]) == 4 for d in detections)


def test_blank_image_yields_no_detections():
    import numpy as np
    blank = np.zeros((480, 640, 3), dtype="uint8")
    assert detect(blank, conf=0.35) == []
