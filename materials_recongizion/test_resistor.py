"""用 cv2 畫合成色帶圖片驗證電阻色碼判讀管線本身能跑完、算出正確數值，
不代表真實鏡頭下的辨識準確度。"""
import numpy as np
import cv2

from resistor import segment_bands, orient_bands, read_resistance, recognize_resistor, locate_body

BGR = {
    "black": (0, 0, 0), "brown": (19, 69, 139), "red": (0, 0, 255),
    "orange": (0, 140, 255), "yellow": (0, 255, 255), "green": (0, 128, 0),
    "blue": (255, 0, 0), "violet": (211, 0, 148), "gray": (128, 128, 128),
    "white": (255, 255, 255), "gold": (0, 165, 255), "silver": (192, 192, 192),
}


def _draw_bands(names, band_w=40, h=60, body=(160, 190, 222)):
    img = np.full((h, band_w * (len(names) + 2), 3), body, dtype=np.uint8)
    for i, name in enumerate(names):
        x0 = band_w * (i + 1)
        cv2.rectangle(img, (x0, 0), (x0 + band_w, h), BGR[name], thickness=-1)
    return img


def test_segment_bands_reads_left_to_right():
    img = _draw_bands(["brown", "black", "red", "gold"])
    assert segment_bands(img) == ["brown", "black", "red", "gold"]


def test_orient_bands_flips_when_tolerance_band_is_first():
    assert orient_bands(["gold", "red", "black", "brown"]) == ["brown", "black", "red", "gold"]
    assert orient_bands(["brown", "black", "red", "gold"]) == ["brown", "black", "red", "gold"]


def test_read_resistance_four_band():
    reading = read_resistance(["brown", "black", "red", "gold"])
    assert reading.ohms == 1000
    assert reading.tolerance_pct == 5
    assert "1k" in reading.text


def test_read_resistance_five_band():
    reading = read_resistance(["red", "violet", "black", "brown", "brown"])
    assert reading.ohms == 2_700
    assert reading.tolerance_pct == 1


def test_read_resistance_rejects_wrong_band_count():
    assert read_resistance(["brown", "black", "red"]) is None


def test_recognize_resistor_end_to_end():
    img = _draw_bands(["brown", "black", "orange", "gold"])
    reading = recognize_resistor(img)
    assert reading is not None
    assert reading.ohms == 10_000


def test_recognize_resistor_survives_loose_roi_with_skin_background():
    """對應使用者截圖回報的情境：框選範圍含大量手掌背景、電阻本體只佔框上方一小段，
    垂直中段落在背景上——locate_body() 應該能先貼合電阻本體，掃描才不會失敗。"""
    skin = (150, 190, 235)
    body = _draw_bands(["brown", "black", "orange", "gold"])
    bh, bw = body.shape[:2]
    canvas = np.full((bh * 4, bw, 3), skin, dtype=np.uint8)
    canvas[10:10 + bh, 0:bw] = body  # 電阻本體只貼在框的最上方，下面全是背景
    reading = recognize_resistor(canvas)
    assert reading is not None
    assert reading.ohms == 10_000


def test_locate_body_crops_tighter_than_loose_roi():
    body = _draw_bands(["brown", "black", "red", "gold"])
    bh, bw = body.shape[:2]
    canvas = np.full((bh * 4, bw, 3), (150, 190, 235), dtype=np.uint8)
    canvas[10:10 + bh, 0:bw] = body
    cropped = locate_body(canvas)
    assert cropped.shape[0] < canvas.shape[0]
