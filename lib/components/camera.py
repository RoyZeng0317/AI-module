"""Real-time camera detection panel.

Opens its own window, streams the webcam through the exact same YOLO11n
detector web/backend/detector.py already uses (the local model that /api/detect
calls — no separate model, no cloud API, matches Rule 06), and draws the
boxes live. Same "no screenshots, no recording" spirit as 視覺電路圖.py: every
frame is discarded right after it's drawn, nothing is written to disk.

home_screen.py must add web/backend/ to sys.path before importing this
module (that's where detector.py lives).
"""

import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from detector import detect


class CameraPanel:
    def __init__(self, windows: tk.Tk, camera_index: int = 0, conf: float = 0.35):
        self.windows = windows
        self.camera_index = camera_index
        self.conf = conf
        self.cap = None
        self.win = None
        self.label = None
        self.running = False
        self._photo = None  # 保留最後一張畫面的參照，不然會被 GC 回收、畫面變空白

    def open(self):
        if self.win is not None:
            self.win.lift()
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("鏡頭", "無法開啟攝影機，確認鏡頭沒有被其他程式占用。")
            return

        self.win = tk.Toplevel(self.windows)
        self.win.title("即時攝影機偵測 (YOLO11n)")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.label = tk.Label(self.win)
        self.label.pack()

        self.running = True
        self._loop()

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.win is not None:
            self.win.destroy()
            self.win = None
            self.label = None

    def _loop(self):
        label = self.label
        if not self.running or self.cap is None or label is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.windows.after(30, self._loop)
            return

        try:
            detections = detect(frame, conf=self.conf)
        except Exception as exc:  # 鏡頭畫面不該因為單一幀推論失敗就整個當掉
            print(f"[camera] 偵測失敗: {exc}")
            detections = []

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            caption = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.putText(frame, caption, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=self._photo)

        self.windows.after(30, self._loop)
