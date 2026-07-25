# 使用視覺技術進行繪製
# 全程不得截圖查看或是錄影
# 而是直接調用大腦開始進行
#
# 即時攝影機畫面 -> 直接呼叫本機 YOLO 模型 (backend/detector.detect，同一個
# ultralytics 本機權重、非雲端 API) -> 把偵測結果畫回畫面 -> tkinter 即時顯示。
# 不落地成截圖檔或錄影檔，也不提供存檔按鈕；每一幀處理完就丟棄。
#
# 尚未提供電路元件標註資料集，所以預設仍載入 backend/yolov8n.pt (COCO 類別，
# 只認得 person/car 這類物件，不認得電阻/電容)。等
# frontend/src/components/circuit_diagram_train.py 訓練出電路元件權重後，
# 用 --weights 指到 best.pt 就能認電路元件。

import argparse
import time
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from web.backend.detector import MODEL_NAME, detect  # noqa: E402

BOX_COLOR = (60, 220, 60)


def open_source(camera, video):
    source = video if video is not None else camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟畫面來源: {source!r}")
    return cap


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f'{d["label"]} {d["confidence"]:.2f}', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    return frame


class CircuitVisionApp:
    """即時視窗:左邊畫面、下方狀態列。沒有存檔/錄影功能——每一幀畫完即丟棄，
    只留在記憶體裡短暫顯示,符合「不得截圖查看或是錄影」的限制。"""

    def __init__(self, cap, weights: str, conf: float):
        self.cap = cap
        self.weights = weights
        self.conf = conf

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("畫面來源沒有輸出任何影像")
        self.h, self.w = frame.shape[:2]

        self.root = tk.Tk()
        self.root.title("視覺電路圖 — 本機 AI 模型即時偵測")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg="black")
        self.canvas.pack()
        self.canvas_img_id = None
        self.photo = None

        self.status = tk.Label(self.root, text="", anchor="w", fg="#333")
        self.status.pack(fill="x")

        self._frame_times = []
        self._update()

    def _update(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            detections = detect(frame, conf=self.conf, weights=self.weights)
            drawn = draw_detections(frame, detections)

            self._frame_times.append(time.time())
            self._frame_times = self._frame_times[-30:]
            fps = 0.0
            if len(self._frame_times) > 1:
                fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
            labels = ", ".join(f'{d["label"]}({d["confidence"]:.2f})' for d in detections) or "無偵測"
            self.status.config(text=f"{fps:.0f} fps | {labels}")

            rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self.canvas_img_id is None:
                self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            else:
                self.canvas.itemconfig(self.canvas_img_id, image=self.photo)
        self.root.after(33, self._update)

    def close(self):
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def resolve_weights(weights_arg: str) -> str:
    if weights_arg:
        return weights_arg
    trained = Path(__file__).resolve().parent / "src" / "components" / "circuit_diagram_runs" / "train" / "weights" / "best.pt"
    if trained.exists():
        return str(trained)
    print(f"[視覺電路圖] 尚未訓練電路元件權重,退回通用 COCO 模型 ({MODEL_NAME})——"
          f"只能認出人/車等泛用物件,認不出電阻/電容。")
    return MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="視覺電路圖 — 即時攝影機 + 本機 YOLO 模型偵測")
    parser.add_argument("--camera", type=int, default=0, help="攝影機編號 (預設 0)")
    parser.add_argument("--video", default=None, help="改用影片檔而非攝影機")
    parser.add_argument("--weights", default=None,
                        help="本機權重路徑;預設先找已訓練好的電路元件權重,找不到才退回 yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="信心度門檻")
    args = parser.parse_args()

    cap = open_source(args.camera, args.video)
    weights = resolve_weights(args.weights)
    app = CircuitVisionApp(cap, weights, args.conf)
    app.run()


if __name__ == "__main__":
    main()
