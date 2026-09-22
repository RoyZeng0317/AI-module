"""即時攝影機電阻色碼／陶瓷電容數字代碼辨識：按 C 凍結畫面、框選元件、疊字顯示判讀
結果，再按任意鍵恢復串流；按 Q 離開。判讀邏輯見 resistor.py / capacitor.py（純函式、
古典影像處理，不訓練模型，比照 shape-vision 子專案的作法）。"""
import cv2
from resistor import recognize_resistor
from capacitor import recognize_capacitor

WINDOW = "Frame"


def analyze_frame(frame):
    """讓使用者框選元件 ROI，同時嘗試電阻色碼與電容數字代碼判讀，回傳疊字後的畫面。"""
    x, y, w, h = cv2.selectROI(WINDOW, frame, showCrosshair=True)
    result = frame.copy()
    if w == 0 or h == 0:
        return result
    roi = frame[y:y + h, x:x + w]
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    lines = []
    resistor = recognize_resistor(roi)
    if resistor:
        lines.append(f"電阻: {resistor.text}")
    capacitor = recognize_capacitor(roi)
    if capacitor:
        lines.append(f"電容: {capacitor.text} (代碼 {capacitor.code})")
    if not lines:
        lines.append("無法判讀，請框選更精準的元件範圍")

    for i, line in enumerate(lines):
        cv2.putText(result, line, (x, max(20, y - 10 - 22 * i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv2.imshow(WINDOW, analyze_frame(frame))
            cv2.waitKey(0)  # 凍結顯示判讀結果，按任意鍵恢復即時串流

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
