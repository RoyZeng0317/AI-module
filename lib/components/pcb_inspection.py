"""PCB inspection panel — component detection + defect detection.

Opens its own window (same pattern as camera.py), lets the user pick a PCB
photo, and runs it through the same local YOLO detector web/backend/detector.py
already uses (no separate model, no cloud API, matches Rule 06).

No PCB-specific dataset exists yet (component classes like resistor/IC, or
defect classes like missing_hole/short/open_circuit/mouse_bite/spur), so this
only runs the COCO-pretrained yolo26n.pt for now — it will not recognise PCB
components or defects, this just validates the load-image -> detect -> draw
pipeline before a real PCB dataset + fine-tuned weights exist (same "training
scaffold first, dataset later" situation as tranning/circuit_diagram_train.py).
Once fine-tuned PCB weights exist, point `weights=` at that checkpoint — the
underlying detect() already supports swapping weights, no code change needed
here.

home_screen.py/lib.main must add web/backend/ to sys.path before importing
this module (that's where detector.py lives) — lib/main.py already does this.
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from detector import detect

IMAGE_FILETYPES = [("PCB 影像", "*.jpg *.jpeg *.png *.bmp"), ("全部檔案", "*.*")]


class PCBInspectionPanel:
    def __init__(self, windows: tk.Tk, conf: float = 0.35, weights: str = "yolo26n.pt"):
        self.windows = windows
        self.conf = conf
        self.weights = weights
        self.win = None
        self.label = None
        self._photo = None  # 保留參照，不然畫面會被 GC 回收變空白（同 camera.py）

    def open(self):
        if self.win is not None:
            self.win.lift()
            return

        self.win = tk.Toplevel(self.windows)
        self.win.title("PCB 檢查 (YOLO26n)")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        tk.Button(self.win, text="開啟 PCB 影像", command=self._pick_image).pack(pady=6)
        self.label = tk.Label(self.win)
        self.label.pack()

    def close(self):
        if self.win is not None:
            self.win.destroy()
            self.win = None
            self.label = None

    def _pick_image(self):
        path = filedialog.askopenfilename(title="選擇 PCB 影像", filetypes=IMAGE_FILETYPES)
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("PCB 檢查", f"無法讀取影像：{path}")
            return

        try:
            detections = detect(frame, conf=self.conf, weights=self.weights)
        except Exception as exc:
            messagebox.showerror("PCB 檢查", f"偵測失敗：{exc}")
            return

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            caption = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.putText(frame, caption, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self._photo)
