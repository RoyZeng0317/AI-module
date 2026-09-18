"""色碼電阻辨識：古典影像處理（HSV 色帶分段比對色碼表），不訓練模型、不需要資料集，
比照 shape-vision/shape_vision/pipeline.py 的純函式設計，方便用合成圖片單元測試。"""
import cv2
import numpy as np
from dataclasses import dataclass

# name: (數字, 倍率, 誤差%, HSV下界, HSV上界)
# 色相區間依相鄰色標準 hue 值的中點切分（黑→棕→橙→金→黃→綠→藍→紫），避免互相重疊；
# 無彩色(黑/灰/銀/白)靠飽和度低+亮度分段區分，這組門檻是近似值，實際鏡頭拍攝的色偏
# 仍可能需要依現場光線再微調。
BANDS = {
    "black":  (0, 1,           None, (0, 0, 0),     (180, 60, 50)),
    "brown":  (1, 10,          1,    (7, 100, 20),   (14, 255, 255)),
    "red":    (2, 100,         2,    (0, 100, 20),   (6, 255, 255)),
    "orange": (3, 1_000,       None, (15, 100, 20),  (17, 255, 255)),
    "yellow": (4, 10_000,      None, (25, 100, 20),  (45, 255, 255)),
    "green":  (5, 100_000,     0.5,  (46, 100, 20),  (90, 255, 255)),
    "blue":   (6, 1_000_000,   0.25, (91, 100, 20),  (130, 255, 255)),
    "violet": (7, 10_000_000,  0.1,  (131, 100, 20), (165, 255, 255)),
    "gray":   (8, 100_000_000, 0.05, (0, 0, 51),     (180, 60, 150)),
    "white":  (9, 1_000_000_000, None, (0, 0, 211),  (180, 60, 255)),
    "gold":   (None, 0.1,  5,  (18, 100, 20),  (24, 255, 255)),
    "silver": (None, 0.01, 10, (0, 0, 151),    (180, 60, 210)),
}
DIGIT_BANDS = {k: v for k, v in BANDS.items() if v[0] is not None}
TOL_ONLY = {"gold", "silver"}


def classify_hsv(mean_hsv):
    """回傳落在範圍內、色相最接近平均值的色帶名稱；沒有任何範圍命中則回傳 None。
    刻意在飽和度 60~100 之間留一段沒有色帶對應的空白（電阻本體常見的淺褐色背景飽和度
    多半落在這段），避免本體顏色被誤判成金/橙/棕色帶。"""
    h, s, v = mean_hsv
    best, best_dist = None, 1e9
    for name, (_, _, _, lo, hi) in BANDS.items():
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            dist = abs(h - (lo[0] + hi[0]) / 2)
            if dist < best_dist:
                best, best_dist = name, dist
    return best


def locate_body(roi_bgr, min_area=150):
    """在使用者框選的 ROI 內，用邊緣/輪廓再抓出電阻本體那條細長區域，避免框選範圍含
    太多導線/背景（例如手掌）時，色帶掃描被稀釋到背景上而完全掃不到任何色帶。抓不到
    明顯候選（例如背景太雜、電阻本體跟背景色太接近）就原樣回傳整個 ROI 當備援。"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 30, 100), np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w * h < min_area or w / h < 1.3:
            continue
        if w * h > best_area:
            best, best_area = (x, y, w, h), w * h
    if best is None:
        return roi_bgr
    x, y, w, h = best
    pad = max(2, int(h * 0.2))
    y0, y1 = max(0, y - pad), min(roi_bgr.shape[0], y + h + pad)
    return roi_bgr[y0:y1, x:x + w]


def segment_bands(roi_bgr, min_width=4):
    """沿 ROI 水平軸掃描，回傳由左到右排列的色帶名稱清單（過濾掉判讀不出的背景區段）。"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    col_mean = hsv[h // 4: h * 3 // 4].mean(axis=0)  # 只取中段，避開上下邊緣反光/陰影
    segments, cur_name, cur_start = [], None, 0
    for x in range(w):
        name = classify_hsv(col_mean[x])
        if name != cur_name:
            if cur_name is not None and x - cur_start >= min_width:
                segments.append(cur_name)
            cur_name, cur_start = name, x
    if cur_name is not None and w - cur_start >= min_width:
        segments.append(cur_name)
    return segments


def orient_bands(names):
    """誤差環(金/銀)通常落在其中一端；若落在左邊就整個反轉，確保順序= 高位數 -> 誤差環。"""
    if names and names[0] in TOL_ONLY and names[-1] not in TOL_ONLY:
        return names[::-1]
    return names


def format_ohms(ohms):
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if ohms >= div:
            return f"{ohms / div:g}{unit}Ω"
    return f"{ohms:g}Ω"


@dataclass
class ResistorReading:
    bands: list
    ohms: float
    tolerance_pct: float | None
    text: str


def read_resistance(band_names):
    """依已排好方向的色帶名稱算出電阻值；支援 4 環(2位數+倍率+誤差)與 5 環(3位數+倍率+誤差)。"""
    if len(band_names) not in (4, 5):
        return None
    *digit_names, mult_name, tol_name = band_names
    if any(n not in DIGIT_BANDS for n in digit_names) or mult_name not in BANDS:
        return None
    digits = "".join(str(BANDS[n][0]) for n in digit_names)
    ohms = int(digits) * BANDS[mult_name][1]
    tolerance = BANDS.get(tol_name, (None, None, None))[2]
    text = f"{format_ohms(ohms)} ±{tolerance}%" if tolerance is not None else format_ohms(ohms)
    return ResistorReading(band_names, ohms, tolerance, text)


def recognize_resistor(roi_bgr):
    """輸入使用者框選的 BGR ROI（不用框得很精準，內部會先自動貼合電阻本體），
    回傳 ResistorReading 或 None。"""
    names = orient_bands(segment_bands(locate_body(roi_bgr)))
    return read_resistance(names)
