"""detect_and_read() 冒煙測試：YOLO 偵測部分用 monkeypatch 假框（不需要真的跑推論），
只驗證「偵測框 -> 裁圖 -> 既有古典 CV 判讀」這條接線本身正確，判讀邏輯本身已經由
test_resistor.py/test_capacitor.py 驗證過，這裡不重複驗證色碼/數字代碼演算法。"""
import cv2
import numpy as np

import detect_and_read as dar

# 一組畫在 160x60 畫布上的電阻色帶（brown/black/red/gold = 1000Ω ±5%），
# 顏色值跟 test_resistor.py 的 BGR 對照表一致。
_BAND_COLORS = [(19, 69, 139), (0, 0, 0), (0, 0, 255), (0, 165, 255)]


def _resistor_patch(band_w=40, h=60):
    img = np.full((h, band_w * (len(_BAND_COLORS) + 2), 3), (160, 190, 222), dtype=np.uint8)
    for i, color in enumerate(_BAND_COLORS):
        x0 = band_w * (i + 1)
        cv2.rectangle(img, (x0, 0), (x0 + band_w, h), color, thickness=-1)
    return img


def test_detect_and_read_matches_box_to_reading(monkeypatch):
    patch = _resistor_patch()
    ph, pw = patch.shape[:2]
    canvas = np.full((ph + 40, pw + 40, 3), 50, dtype=np.uint8)
    canvas[20:20 + ph, 20:20 + pw] = patch

    monkeypatch.setattr(dar, "detect", lambda image, conf, weights: [
        {"label": "resistor", "confidence": 0.9, "box": [20, 20, 20 + pw, 20 + ph]},
    ])

    results = dar.detect_and_read(canvas, weights="dummy.pt")
    assert len(results) == 1
    assert results[0]["label"] == "resistor"
    assert results[0]["reading"] is not None
    assert results[0]["reading"].ohms == 1000


def test_detect_and_read_keeps_none_when_unreadable(monkeypatch):
    canvas = np.full((100, 100, 3), 50, dtype=np.uint8)  # 純灰畫面，判讀不出任何讀數
    monkeypatch.setattr(dar, "detect", lambda image, conf, weights: [
        {"label": "capacitor", "confidence": 0.5, "box": [10, 10, 60, 60]},
    ])

    results = dar.detect_and_read(canvas, weights="dummy.pt")
    assert len(results) == 1
    assert results[0]["reading"] is None
