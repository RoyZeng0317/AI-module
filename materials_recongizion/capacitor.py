"""陶瓷電容三位數字代碼辨識：自己用 cv2.putText 畫 0-9 樣板做樣板比對(template
matching)，不訓練模型、不需要資料集（tranning/OCR.py 尚無真實資料集可訓練數字辨識，
見 to_do_list.md #08）。同樣是純函式設計，方便用合成圖片單元測試。"""
import cv2
import numpy as np
from dataclasses import dataclass

TEMPLATE_SIZE = (24, 32)  # (w, h)


def _build_templates():
    """畫在較大的畫布上再依實際筆畫緊裁+縮放到 TEMPLATE_SIZE，跟 find_digit_glyphs()
    裁出來的候選數字用同一種「先緊裁再縮放」流程，兩邊比例才對得上（否則像「1」這種
    窄字元硬被拉伸塞滿整個樣板寬度，會跟模板對不起來）。"""
    templates = {}
    for d in range(10):
        canvas = np.zeros((60, 50), np.uint8)
        cv2.putText(canvas, str(d), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        x, y, w, h = cv2.boundingRect(canvas)
        glyph = cv2.resize(canvas[y:y + h, x:x + w], TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        templates[d] = glyph
    return templates


DIGIT_TEMPLATES = _build_templates()


def locate_face(roi_bgr, min_area=150):
    """在框選範圍內找出電容本體（飽和度最高的色塊：陶瓷電容多為橙/棕/藍等鮮豔顏色，
    比灰暗背景或反光導線更飽和），裁掉多餘的導線/背景，比照 resistor.locate_body() 的
    作法。本體通常是圓形/橢圓，裁出來的矩形四個角落一定還是背景色——這圈背景比本體暗，
    會在 find_digit_glyphs() 二值化時形成包住印刷數字的外層輪廓，讓 RETR_EXTERNAL 又
    看不到數字（跟不裁切時同一種巢狀輪廓問題，只是換了一層），所以額外把本體輪廓以外
    的角落塗成本體的平均色，消除這圈背景。抓不到明顯候選就原樣回傳整個 ROI 當備援，
    不會讓合成測試圖(全灰階、無飽和度)的行為改變。"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, binary = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area, best_contour = None, 0, None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or not (0.4 <= w / h <= 2.5):
            continue
        if area > best_area:
            best, best_area, best_contour = (x, y, w, h), area, c
    if best is None:
        return roi_bgr
    x, y, w, h = best
    pad = max(2, int(min(w, h) * 0.15))
    y0, y1 = max(0, y - pad), min(roi_bgr.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_bgr.shape[1], x + w + pad)
    face = roi_bgr[y0:y1, x0:x1].copy()

    mask = np.zeros(face.shape[:2], np.uint8)
    cv2.drawContours(mask, [best_contour - (x0, y0)], -1, 255, -1)
    mean_color = cv2.mean(face, mask=mask)[:3]
    face[mask == 0] = mean_color
    return face


def find_digit_glyphs(roi_bgr, min_area=20):
    """把 ROI 二值化，找出候選數字輪廓的裁圖，依 x 座標由左到右排序回傳。前提是 roi_bgr
    已經只含元件本體（呼叫端先用 locate_face() 裁過)：真實照片如果直接對整個使用者
    框選範圍(還混著導線/背景)做這一步，印刷數字會變成背景色塊「內部」的孔洞輪廓，
    RETR_EXTERNAL 只抓最外層輪廓會直接漏掉數字本身——這正是原本抓不到真實數字、但合成
    測試圖(數字直接貼在畫面邊緣的白底上)卻能通過單元測試的原因。"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted((cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area),
                    key=lambda b: b[0])
    return [binary[y:y + h, x:x + w] for x, y, w, h in boxes]


def match_digit(glyph):
    """單一數字裁圖跟 10 個樣板做 normalized cross-correlation，回傳最像的數字。"""
    resized = cv2.resize(glyph, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    scores = {d: cv2.matchTemplate(resized, tpl, cv2.TM_CCOEFF_NORMED)[0][0]
              for d, tpl in DIGIT_TEMPLATES.items()}
    return max(scores, key=scores.get)


def code_to_pf(code):
    """3 位數字代碼轉 pF：前兩碼為有效數字，第三碼為 10 的次方數（EIA 標準）。"""
    if len(code) != 3 or not code.isdigit():
        return None
    return int(code[:2]) * 10 ** int(code[2])


def format_pf(pf):
    if pf >= 1_000_000:
        return f"{pf / 1_000_000:g}µF"
    if pf >= 1_000:
        return f"{pf / 1_000:g}nF"
    return f"{pf:g}pF"


@dataclass
class CapacitorReading:
    code: str
    pf: float
    text: str


def recognize_capacitor(roi_bgr):
    """輸入使用者框選的 BGR ROI（不用框得很精準，內部會先自動貼合電容本體），
    回傳 CapacitorReading 或 None。"""
    glyphs = find_digit_glyphs(locate_face(roi_bgr))
    if len(glyphs) != 3:
        return None
    code = "".join(str(match_digit(g)) for g in glyphs)
    pf = code_to_pf(code)
    if pf is None:
        return None
    return CapacitorReading(code, pf, format_pf(pf))
