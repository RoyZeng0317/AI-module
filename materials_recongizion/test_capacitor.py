"""用同一套 cv2.putText 樣板重畫的合成數字圖片驗證電容代碼判讀管線本身能跑完、
算出正確數值，不代表真實印刷字體下的辨識準確度。"""
import numpy as np
import cv2

from capacitor import find_digit_glyphs, match_digit, code_to_pf, recognize_capacitor, locate_face


def _draw_code(code, w=100, h=40):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, code, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


def _draw_loose_roi(code, w=300, h=300):
    """模擬使用者實際框選：深色背景 + 一塊鮮豔(高飽和度)圓形本體，數字印在本體上。
    重現真實照片會失敗的情境——印刷數字其實是本體色塊「內部」的孔洞輪廓，框選範圍
    又比本體大很多。"""
    img = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)  # 深色背景(低飽和度)
    cv2.circle(img, (w // 2, h // 2), 80, (30, 140, 230), -1)  # 鮮豔橙色本體(高飽和度)
    cv2.putText(img, code, (w // 2 - 45, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 2)
    return img


def test_find_digit_glyphs_splits_three_digits():
    glyphs = find_digit_glyphs(_draw_code("104"))
    assert len(glyphs) == 3


def test_match_digit_identifies_each_digit():
    for d in range(10):
        glyphs = find_digit_glyphs(_draw_code(str(d) + "00"))
        assert match_digit(glyphs[0]) == d


def test_code_to_pf_standard_formula():
    assert code_to_pf("104") == 100_000
    assert code_to_pf("471") == 470
    assert code_to_pf("12") is None


def test_recognize_capacitor_end_to_end():
    reading = recognize_capacitor(_draw_code("104"))
    assert reading is not None
    assert reading.code == "104"
    assert reading.pf == 100_000


def test_locate_face_crops_out_the_background():
    face = locate_face(_draw_loose_roi("104"))
    assert face.shape[0] < 300 and face.shape[1] < 300


def test_recognize_capacitor_with_loose_roi_and_colored_body():
    """重現截圖裡的真實情境：使用者框選範圍比本體大很多，數字印在鮮豔本體上而不是
    直接貼在白底——原本會因為 RETR_EXTERNAL 只抓最外層輪廓(整個本體)而完全漏看數字。"""
    reading = recognize_capacitor(_draw_loose_roi("104"))
    assert reading is not None
    assert reading.code == "104"
