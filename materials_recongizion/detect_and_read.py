"""YOLO 定位 + 既有古典 CV 判讀（resistor.py/capacitor.py 原樣重用，不重寫）：取代
main.py 原本手動 cv2.selectROI 框選那一步，改用 tranning/materials_yolo_train.py
訓出來的偵測器自動找出畫面裡每個候選元件的位置，裁出候選框後分別丟給電阻色碼/
電容數字代碼判讀。目前沒有真實資料集訓出來的權重，weights 參數可以指到任何
YOLO 權重先跑通管線（例如 tranning/yolo_custom.py 的合成資料 best.pt），偵測
準確度不代表真實場景表現，判讀邏輯完全沒變、仍是 resistor.py/capacitor.py 的
純函式（不訓練模型）。

Usage:
    python detect_and_read.py --image photo.jpg --weights ../tranning/materials_yolo_runs/train/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "web" / "backend"), str(Path(__file__).resolve().parent)]

import cv2  # noqa: E402

from capacitor import recognize_capacitor  # noqa: E402
from detector import detect  # noqa: E402
from resistor import recognize_resistor  # noqa: E402

DEFAULT_WEIGHTS = _ROOT / "tranning" / "materials_yolo_runs" / "train" / "weights" / "best.pt"


def detect_and_read(image, weights=None, conf: float = 0.35) -> list:
    """回傳每個偵測框的 {label, confidence, box, reading}；reading 是
    ResistorReading/CapacitorReading/None（古典判讀抓不到讀數時維持 None，不編造）。"""
    results = []
    for det in detect(image, conf=conf, weights=str(weights or DEFAULT_WEIGHTS)):
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        roi = image[y1:y2, x1:x2]
        reading = recognize_resistor(roi) or recognize_capacitor(roi)
        results.append({**det, "reading": reading})
    return results


def main():
    ap = argparse.ArgumentParser(description="YOLO 定位元件 + 既有古典 CV 判讀電阻/電容")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    image = cv2.imread(str(a.image))
    if image is None:
        raise ValueError(f"無法讀取圖片：{a.image}")
    for det in detect_and_read(image, a.weights, a.conf):
        reading = f"（{det['reading'].text}）" if det["reading"] else "（判讀不出讀數）"
        print(f"{det['label']} {det['confidence']:.0%} {det['box']} {reading}")


if __name__ == "__main__":
    main()
