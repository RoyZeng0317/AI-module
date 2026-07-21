"""Tkinter live viewer: canvas preview on the left, tunable control panel on
the right. Wraps shape_vision.pipeline + shape_vision.camera — no detection
logic lives in this file.
"""

import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from .pipeline import DetectionParams, detect_shapes, draw_shapes, summarize

VIEWS = ["Raw (BGR)", "Gray", "Blur", "Canny Edges", "Shape Detection"]


class ShapeVisionApp:
    def __init__(self, source):
        self.source = source

        frame = source.read()
        if frame is None:
            raise RuntimeError("Frame source produced no frames")
        self.h, self.w = frame.shape[:2]
        self.last_frame = frame

        self.root = tk.Tk()
        self.root.title("Shape Vision — live OpenCV shape/circle detector")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(main)
        left.pack(side="left", padx=(0, 8))
        self.canvas = tk.Canvas(left, width=self.w, height=self.h, bg="black")
        self.canvas.pack()
        self.canvas_img_id = None
        self.photo = None
        self.status = tk.Label(left, text="", anchor="w", fg="#333")
        self.status.pack(fill="x", pady=(4, 0))

        right = tk.Frame(main)
        right.pack(side="left", fill="y")
        self.params = DetectionParams()
        self.view = tk.IntVar(value=4)
        self._build_controls(right)

        self._frame_times = []
        self._update()

    # -- UI construction -------------------------------------------------
    def _slider(self, parent, label, var, mn, mx, step):
        row = tk.Frame(parent)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, width=16, anchor="w").pack(side="left")
        tk.Scale(row, variable=var, from_=mn, to=mx, resolution=step,
                 orient="horizontal", length=170, showvalue=True).pack(side="left")

    def _build_controls(self, parent):
        p = self.params

        tk.Label(parent, text="View", font=(None, 11, "bold")).pack(anchor="w")
        for i, name in enumerate(VIEWS):
            tk.Radiobutton(parent, text=name, variable=self.view, value=i).pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=6)

        tk.Label(parent, text="Gaussian Blur", font=(None, 11, "bold")).pack(anchor="w")
        self._var_ksize = tk.IntVar(value=p.blur_ksize)
        self._var_sigma = tk.DoubleVar(value=p.blur_sigma)
        self._slider(parent, "ksize (odd)", self._var_ksize, 1, 31, 2)
        self._slider(parent, "sigma", self._var_sigma, 0.0, 10.0, 0.1)
        ttk.Separator(parent).pack(fill="x", pady=6)

        tk.Label(parent, text="Canny Edges", font=(None, 11, "bold")).pack(anchor="w")
        self._var_t1 = tk.IntVar(value=p.canny_t1)
        self._var_t2 = tk.IntVar(value=p.canny_t2)
        self._slider(parent, "T1 (low)", self._var_t1, 0, 500, 1)
        self._slider(parent, "T2 (high)", self._var_t2, 0, 500, 1)
        ttk.Separator(parent).pack(fill="x", pady=6)

        tk.Label(parent, text="Shape filter", font=(None, 11, "bold")).pack(anchor="w")
        self._var_min_area = tk.IntVar(value=p.min_area)
        self._var_max_area = tk.IntVar(value=p.max_area)
        self._var_eps = tk.IntVar(value=p.approx_eps_pct)
        self._var_circ = tk.IntVar(value=p.circularity_thr)
        self._slider(parent, "Min area", self._var_min_area, 0, 30000, 100)
        self._slider(parent, "Max area", self._var_max_area, 100, 200000, 1000)
        self._slider(parent, "Epsilon %", self._var_eps, 1, 15, 1)
        self._slider(parent, "Circularity %", self._var_circ, 0, 100, 1)
        ttk.Separator(parent).pack(fill="x", pady=6)

        tk.Label(parent, text="Shapes to show", font=(None, 11, "bold")).pack(anchor="w")
        self._var_tri = tk.BooleanVar(value=p.show_triangle)
        self._var_sq = tk.BooleanVar(value=p.show_square)
        self._var_ci = tk.BooleanVar(value=p.show_circle)
        self._var_pg = tk.BooleanVar(value=p.show_polygon)
        self._var_hough = tk.BooleanVar(value=p.use_hough)
        for text, var in [("Triangle", self._var_tri), ("Square / Rectangle", self._var_sq),
                          ("Circle", self._var_ci), ("Polygon 5+", self._var_pg)]:
            tk.Checkbutton(parent, text=text, variable=var).pack(anchor="w")
        tk.Checkbutton(parent, text="Cross-check circles with Hough transform",
                       variable=self._var_hough).pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=6)

        tk.Button(parent, text="Save snapshot", command=self.save_snapshot).pack(fill="x", pady=2)
        tk.Button(parent, text="Quit", command=self.close).pack(fill="x", pady=2)

    def _sync_params(self):
        p = self.params
        p.blur_ksize = self._var_ksize.get()
        p.blur_sigma = self._var_sigma.get()
        p.canny_t1 = self._var_t1.get()
        p.canny_t2 = self._var_t2.get()
        p.min_area = self._var_min_area.get()
        p.max_area = self._var_max_area.get()
        p.approx_eps_pct = self._var_eps.get()
        p.circularity_thr = self._var_circ.get()
        p.show_triangle = self._var_tri.get()
        p.show_square = self._var_sq.get()
        p.show_circle = self._var_ci.get()
        p.show_polygon = self._var_pg.get()
        p.use_hough = self._var_hough.get()

    # -- main loop ---------------------------------------------------------
    def _render(self, frame):
        self._sync_params()
        from .pipeline import to_gray_blur, to_edges  # local import keeps top-level API small

        view = self.view.get()
        if view == 0:
            return frame, "Raw BGR"
        gray, blur = to_gray_blur(frame, self.params)
        if view == 1:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Gray"
        if view == 2:
            return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), f"Blur ksize={self.params.blur_ksize} sigma={self.params.blur_sigma:.1f}"
        if view == 3:
            edges = to_edges(blur, self.params)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), f"Canny T1={self.params.canny_t1} T2={self.params.canny_t2}"

        shapes, _ = detect_shapes(frame, self.params)
        out = draw_shapes(frame, shapes)
        counts = summarize(shapes)
        text = "Shape Detection | " + ("  ".join(f"{k}:{v}" for k, v in counts.items()) if counts else "no shapes")
        return out, text

    def _update(self):
        frame = self.source.read()
        if frame is not None:
            self.last_frame = frame
            display, status_text = self._render(frame)
            self._frame_times.append(time.time())
            self._frame_times = self._frame_times[-30:]
            if len(self._frame_times) > 1:
                fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
                status_text += f"  |  {fps:.0f} fps"
            self.status.config(text=status_text)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self.canvas_img_id is None:
                self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            else:
                self.canvas.itemconfig(self.canvas_img_id, image=self.photo)
            self.last_display = display
        self.root.after(33, self._update)

    def save_snapshot(self):
        if hasattr(self, "last_display"):
            name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, self.last_display)
            self.status.config(text=f"Saved {name}")

    def close(self):
        try:
            self.source.release()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
