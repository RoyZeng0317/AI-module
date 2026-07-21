"""Classification tests against synthetic shapes drawn with cv2 primitives —
no webcam or display required, so this runs anywhere.
"""

import numpy as np
import cv2

from shape_vision.pipeline import DetectionParams, detect_shapes, summarize


def _canvas():
    return np.zeros((300, 300, 3), dtype=np.uint8)


def test_detects_triangle():
    img = _canvas()
    pts = np.array([[150, 40], [60, 220], [240, 220]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    shapes, _ = detect_shapes(img, DetectionParams())
    labels = summarize(shapes)
    assert labels.get("Triangle", 0) >= 1


def test_detects_square():
    img = _canvas()
    cv2.rectangle(img, (75, 75), (225, 225), (255, 255, 255), thickness=-1)

    shapes, _ = detect_shapes(img, DetectionParams())
    labels = summarize(shapes)
    assert labels.get("Square", 0) >= 1


def test_detects_rectangle():
    img = _canvas()
    cv2.rectangle(img, (40, 100), (260, 200), (255, 255, 255), thickness=-1)

    shapes, _ = detect_shapes(img, DetectionParams())
    labels = summarize(shapes)
    assert labels.get("Rectangle", 0) >= 1


def test_detects_circle():
    img = _canvas()
    cv2.circle(img, (150, 150), 90, (255, 255, 255), thickness=-1)

    shapes, _ = detect_shapes(img, DetectionParams())
    labels = summarize(shapes)
    assert labels.get("Circle", 0) >= 1


def test_area_filter_rejects_tiny_noise():
    img = _canvas()
    cv2.rectangle(img, (10, 10), (18, 18), (255, 255, 255), thickness=-1)  # 8x8 speck

    params = DetectionParams(min_area=500)
    shapes, _ = detect_shapes(img, params)
    assert shapes == []


def test_shape_toggle_hides_label():
    img = _canvas()
    cv2.rectangle(img, (75, 75), (225, 225), (255, 255, 255), thickness=-1)

    params = DetectionParams(show_square=False)
    shapes, _ = detect_shapes(img, params)
    labels = summarize(shapes)
    assert "Square" not in labels and "Rectangle" not in labels
