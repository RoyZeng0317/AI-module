"""用合成圖（紅圓、綠方、藍三角）驗證 img_convert_gray.py 的灰階→特徵→上色→報告流程。"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import img_convert_gray as icg  # noqa: E402


def make_image():
    img = np.full((300, 300, 3), 40, np.uint8)
    cv2.circle(img, (70, 70), 40, (0, 0, 230), -1)                 # 紅圓（左上）
    cv2.rectangle(img, (190, 30), (270, 110), (0, 200, 0), -1)     # 綠方（右上）
    tri = np.array([[150, 180], [100, 270], [200, 270]], np.int32)
    cv2.fillPoly(img, [tri], (230, 0, 0))                         # 藍三角（下中）
    return img


def test_analyze_finds_shapes_and_colors():
    img = make_image()
    gray, edges, regions = icg.analyze(img)
    assert gray.ndim == 2 and edges.shape == gray.shape
    found = {(r["shape"], r["color"]) for r in regions}
    assert ("圓形", "紅") in found
    assert ("正方形", "綠") in found
    assert ("三角形", "藍") in found


def test_recolor_modes_and_report(tmp_path):
    img = make_image()
    path = tmp_path / "shapes.png"
    icg.write_image(path, img)
    for mode in ("original", "shape", "brightness"):
        report, colored = icg.run(path, mode, tmp_path / "out")
        assert colored.shape == img.shape
        assert (tmp_path / "out" / f"shapes_{mode}.png").exists()
    assert "圖片特徵分析報告" in report
    assert "規則幾何形狀" in report
    # original 模式：圓的中心要變回偏紅（灰階轉回彩色）
    _, colored = icg.run(path, "original", tmp_path / "out")
    b, g, r = colored[70, 70]
    assert r > g and r > b
