"""本地物品標註、YOLO 微調與即時偵測：python -m lib.yolo.object_labeler。"""
import argparse, copy, queue, subprocess, sys, threading, uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tranning"), str(ROOT / "web" / "backend")]
from object_labeling import DEFAULT_DATASET, LabelDataset, fit_geometry, project_path, rectangle


class CameraWorker:
    """相機與推論在背景執行；只保留最新幀，不累積影像。"""
    def __init__(self, index=0, weights=None, classes=None, capture=None, detector=None, loader=None, conf=0.35):
        self.index, self.weights, self.classes = index, weights, classes
        self.capture, self.detector, self.loader = capture, detector, loader
        self.conf = conf
        self.messages, self.stop_event = queue.Queue(maxsize=1), threading.Event()
        self.thread = None

    def publish(self, item):
        try:
            self.messages.get_nowait()
        except queue.Empty:
            pass
        self.messages.put_nowait(item)

    def run(self):
        cap = None
        try:
            if self.weights is not None:
                from detector import _model, detect
                weights = project_path(self.weights)
                if not weights.is_file():
                    raise ValueError("找不到自訂權重；請先標註與訓練，再選擇 best.pt。")
                model = (self.loader or _model)(str(weights))
                names = model.names
                names = list(names.values()) if isinstance(names, dict) else list(names)
                if names != list(self.classes):
                    raise ValueError("此權重類別與目前資料集不同，請選擇此資料集訓練的 best.pt。")
                self.detector = self.detector or detect
            cap = (self.capture or cv2.VideoCapture)(self.index)
            if not cap.isOpened():
                raise ValueError("無法開啟攝影機，請確認索引與其他程式占用狀態。")
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("無法讀取攝影機畫面。")
                detections = self.detector(frame, weights=str(weights), conf=self.conf) if self.weights is not None else []
                self.publish(("frame", Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), detections))
                self.stop_event.wait(0.02)
        except Exception as exc:
            self.publish(("error", str(exc)))
        finally:
            if cap is not None:
                cap.release()

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()


class Labeler:
    def __init__(self, root, folder=DEFAULT_DATASET):
        self.root, self.dataset = root, LabelDataset(folder)
        self.key, self.image, self.boxes = None, None, []
        self.dirty, self.worker, self.pending_camera = False, None, False
        self.frame, self.detections, self.drag = None, [], None
        self.process, self.log_handle, self.run_dir = None, None, None
        self.preparing, self.tasks = False, queue.Queue()
        self.closing = False
        self.photo, self.geometry = None, (1, 0, 0)
        self.status = tk.StringVar(value="匯入照片或開啟攝影機；滑鼠拖曳框選物品。")
        self.split, self.negative = tk.StringVar(value="train"), tk.BooleanVar(value=False)
        self.class_name = tk.StringVar(value=self.dataset.classes[0])
        self.weights = tk.StringVar(value=str(ROOT / "yolo11n.pt"))
        self.best = tk.StringVar()
        self.confidence = tk.StringVar(value="0.25")
        self.epochs, self.batch, self.camera_index = tk.StringVar(value="150"), tk.StringVar(value="4"), tk.StringVar(value="0")
        root.title("物品標註與自訂 YOLO 偵測")
        root.geometry("1120x820")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build()
        self.refresh_list()
        self.timer = root.after(50, self.poll)

    def button(self, parent, text, command):
        ttk.Button(parent, text=text, command=lambda: self.safe(command)).pack(side="left", padx=3, pady=3)

    def build(self):
        toolbar = ttk.Frame(self.root); toolbar.pack(fill="x")
        for title, action in (("匯入照片", self.import_files), ("攝影機預覽", self.preview), ("擷取這一幀", self.capture_frame), ("停止鏡頭", self.stop_camera)):
            self.button(toolbar, title, action)
        ttk.Label(toolbar, text="鏡頭索引").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.camera_index, width=3).pack(side="left")
        ttk.Label(toolbar, text="偵測門檻").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.confidence, width=5).pack(side="left")
        self.button(toolbar, "刪除未標註照片", self.delete_unmarked)
        edit = ttk.Frame(self.root); edit.pack(fill="x")
        self.class_box = ttk.Combobox(edit, textvariable=self.class_name, values=self.dataset.classes, state="readonly", width=16)
        self.class_box.pack(side="left", padx=5)
        for title, action in (("新增類別", self.add_class), ("更名類別", self.rename_class), ("刪除類別", self.delete_class)):
            self.button(edit, title, action)
        edit = ttk.Frame(self.root); edit.pack(fill="x")
        for title, action in (("刪除選取框", self.delete_box), ("儲存標註", self.save), ("重新載入", self.reload)):
            self.button(edit, title, action)
        ttk.Combobox(edit, textvariable=self.split, values=("train", "val"), state="readonly", width=7).pack(side="left")
        self.split.trace_add("write", lambda *_: self.changed())
        ttk.Checkbutton(edit, text="背景負例（沒有目標物品）", variable=self.negative, command=self.changed).pack(side="left")
        body = ttk.Frame(self.root); body.pack(fill="both", expand=True)
        side = ttk.Frame(body); side.pack(side="left", fill="y")
        ttk.Label(side, text="照片：○未確認 / ✓已儲存").pack()
        self.images = tk.Listbox(side, width=27, exportselection=False)
        self.images.pack(fill="both", expand=True)
        self.images.bind("<<ListboxSelect>>", self.select_image)
        self.box_list = tk.Listbox(side, height=7, exportselection=False)
        self.box_list.pack(fill="x")
        self.canvas = tk.Canvas(body, bg="#20242b", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.drag_start)
        self.canvas.bind("<B1-Motion>", self.drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.drag_end)
        training = ttk.LabelFrame(self.root, text="微調：只在按下開始後訓練，初始權重與輸出均在本專案")
        training.pack(fill="x", padx=5)
        ttk.Label(training, text="起始權重").grid(row=0, column=0)
        ttk.Entry(training, textvariable=self.weights).grid(row=0, column=1, sticky="ew")
        ttk.Button(training, text="選擇 .pt", command=lambda: self.safe(lambda: self.pick_weights(self.weights))).grid(row=0, column=2)
        options = ttk.Frame(training); options.grid(row=1, column=0, columnspan=3, sticky="ew")
        for title, variable in (("epochs", self.epochs), ("batch", self.batch)):
            ttk.Label(options, text=title).pack(side="left")
            ttk.Entry(options, textvariable=variable, width=5).pack(side="left")
        self.button(options, "匯出資料集", self.export)
        self.button(options, "開始微調", self.start_train)
        self.button(options, "停止訓練", self.stop_train)
        ttk.Label(training, text="偵測權重").grid(row=2, column=0)
        ttk.Entry(training, textvariable=self.best).grid(row=2, column=1, sticky="ew")
        ttk.Button(training, text="選擇 best.pt", command=lambda: self.safe(lambda: self.pick_weights(self.best))).grid(row=2, column=2)
        ttk.Button(training, text="使用自訂模型即時偵測", command=lambda: self.safe(self.detect)).grid(row=3, column=1, sticky="w")
        training.columnconfigure(1, weight=1)
        ttk.Label(self.root, textvariable=self.status, wraplength=1080).pack(fill="x", padx=5, pady=5)

    def safe(self, action):
        try:
            return action()
        except Exception as exc:
            messagebox.showerror("物品標註", str(exc), parent=self.root)

    def changed(self):
        if self.key is not None and self.worker is None:
            self.dirty = True

    def may_leave(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("尚未儲存", "是否先儲存目前標註？", parent=self.root)
        if answer is None:
            return False
        if answer:
            self.save()
        return True

    def refresh_list(self):
        self.keys = list(self.dataset.data["images"])
        self.images.delete(0, "end")
        for i, key in enumerate(self.keys):
            item = self.dataset.data["images"][key]
            self.images.insert("end", f"{'✓' if item['reviewed'] else '○'} {i+1} {item['split']} {key[:8]}")
        if self.key in self.keys:
            self.images.selection_set(self.keys.index(self.key))

    def load(self, key):
        self.key = key
        self.image, item = self.dataset.load(key)
        self.boxes = [list(b) for b in item["boxes"]]
        self.split.set(item["split"]); self.negative.set(item["negative"])
        self.dirty = False
        self.detections = []
        self.draw()

    def select_image(self, _):
        selected = self.images.curselection()
        if selected:
            self.safe(lambda: self.switch_image(self.keys[selected[0]]))

    def switch_image(self, key):
        if self.may_leave():
            self.stop_camera()
            self.load(key)

    def import_files(self):
        if not self.may_leave():
            return
        paths = filedialog.askopenfilenames(parent=self.root, filetypes=[("照片", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not paths:
            return
        self.stop_camera()
        for path in paths:
            key = self.dataset.import_image(path)
        self.load(key); self.refresh_list()

    def add_class(self):
        name = simpledialog.askstring("新增物品類別", "類別名稱，例如：鉛筆", parent=self.root)
        if name is not None:
            self.dataset.add_class(name)
            self.class_box.configure(values=self.dataset.classes)
            self.class_name.set(name.strip())

    def delete_box(self):
        selected = self.box_list.curselection()
        if selected and self.worker is None:
            self.boxes.pop(selected[0]); self.changed(); self.draw()

    def editing_ready(self):
        if self.worker is not None or self.pending_camera or self.process is not None or self.preparing:
            raise ValueError("請先停止鏡頭並等待訓練／匯出工作結束，再修改資料。")
        return self.may_leave()

    def refresh_dataset(self):
        if self.key in self.dataset.data["images"]:
            self.load(self.key)
        else:
            self.key, self.image, self.boxes = None, None, []
            self.dirty = False
            self.box_list.delete(0, "end")
            self.draw()
        self.class_box.configure(values=self.dataset.classes)
        if self.class_name.get() not in self.dataset.classes:
            self.class_name.set(self.dataset.classes[0])
        self.refresh_list()

    def delete_unmarked(self):
        if not self.editing_ready():
            return
        count = len(self.dataset.unmarked_keys())
        if not count:
            self.status.set("沒有可刪除的未標註照片；已儲存標註及背景負例會保留。")
            return
        if messagebox.askyesno("刪除未標註照片", f"刪除 {count} 張尚未確認且沒有標註框的照片？\n將保留於資料集 trash 備份。", parent=self.root):
            count = self.dataset.delete_unmarked()
            self.refresh_dataset()
            self.status.set(f"已刪除 {count} 張未標註照片，備份位於 {self.dataset.folder / 'trash'}。")

    def rename_class(self):
        if not self.editing_ready():
            return
        name = self.class_name.get()
        replacement = simpledialog.askstring("更名類別", "修正名稱（保留所有既有框）：", initialvalue=name, parent=self.root)
        if replacement is not None:
            self.dataset.rename_class(name, replacement)
            self.class_name.set(replacement.strip())
            self.refresh_dataset()
            self.status.set("類別已更名，標註框保留。舊模型名稱不會自動更新，需重新匯出與訓練。")

    def delete_class(self):
        if not self.editing_ready():
            return
        name = self.class_name.get()
        index = self.dataset.classes.index(name)
        count = sum(sum(box[0] == index for box in item["boxes"]) for item in self.dataset.data["images"].values())
        if messagebox.askyesno("刪除類別", f"刪除「{name}」及其 {count} 個標註框？\n受影響照片需重新確認；其他類別框會保留。\n若只修正錯字，請取消並使用「更名類別」。\n原標註會備份；既有模型不變，之後需重新訓練。", parent=self.root):
            affected = self.dataset.delete_class(name)
            self.refresh_dataset()
            self.status.set(f"已刪除「{name}」；{affected} 張照片需重新確認。類別編號已同步，請重新匯出與訓練。")

    def save(self):
        if self.key is None or self.worker is not None:
            raise ValueError("請先選擇照片，或擷取攝影機影像後再標註。")
        self.dataset.save(self.key, self.boxes, self.split.get(), self.negative.get())
        self.dirty = False
        self.refresh_list()
        self.status.set("標註已儲存。驗證 val 請使用不同拍攝場景，避免連續近似影格跨組。")

    def reload(self):
        if self.key is not None and self.may_leave():
            self.stop_camera(); self.load(self.key)

    def original_point(self, x, y):
        scale, ox, oy = self.geometry
        return (x-ox)/scale, (y-oy)/scale

    def drag_start(self, event):
        if self.image is not None and self.worker is None:
            self.drag = (event.x, event.y)

    def drag_move(self, event):
        if self.drag:
            self.canvas.delete("draft")
            self.canvas.create_rectangle(*self.drag, event.x, event.y, outline="#ffe16a", width=2, tags="draft")

    def drag_end(self, event):
        if self.drag is None:
            return
        start, self.drag = self.drag, None
        try:
            points = rectangle([*self.original_point(*start), *self.original_point(event.x, event.y)], self.image.width, self.image.height)
            self.boxes.append([self.dataset.classes.index(self.class_name.get()), *points])
            self.negative.set(False); self.changed()
        except ValueError:
            pass
        self.draw()

    def draw(self):
        image = self.frame if self.worker is not None else self.image
        self.canvas.delete("all")
        if image is None:
            return
        scale, ox, oy = self.geometry = fit_geometry(image.width, image.height, max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        self.photo = ImageTk.PhotoImage(image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale)))))
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        labels = [(self.dataset.classes[b[0]], b[1:]) for b in self.boxes] if self.worker is None else [(f"{d['label']} {d['confidence']:.0%}", d["box"]) for d in self.detections]
        for label, (x1, y1, x2, y2) in labels:
            self.canvas.create_rectangle(ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale, outline="#69ef9f", width=2)
            self.canvas.create_text(ox+x1*scale+3, max(oy, oy+y1*scale-20), text=label, fill="#69ef9f", anchor="nw", font=("Microsoft JhengHei", 13, "bold"))
        if self.worker is None:
            self.box_list.delete(0, "end")
            for i, box in enumerate(self.boxes):
                self.box_list.insert("end", f"{i+1}. {self.dataset.classes[box[0]]}")

    def start_camera(self, weights=None):
        if self.worker is not None or self.pending_camera:
            raise ValueError("請先停止目前鏡頭，等待釋放後再開啟。")
        if not self.may_leave():
            return
        index = int(self.camera_index.get())
        confidence = float(self.confidence.get())
        if not 0 < confidence < 1:
            raise ValueError("偵測門檻必須介於0與1之間。")
        if index < 0:
            raise ValueError("鏡頭索引不能小於零。")
        if weights is not None and not project_path(weights).is_file():
            raise ValueError("請先訓練並選擇本地 best.pt，尚無權重時不會使用通用模型替代。")
        self.frame, self.detections = None, []
        self.worker = CameraWorker(index, weights, self.dataset.classes.copy(), conf=confidence)
        self.worker.start()
        self.status.set("正在載入自訂模型與開啟鏡頭…" if weights else "攝影機預覽：按「擷取這一幀」後框選物品。")

    def preview(self):
        self.start_camera()

    def detect(self):
        if not self.best.get().strip():
            raise ValueError("尚未選擇自訂模型。請完成標註與訓練，再選擇 best.pt。")
        self.start_camera(self.best.get())

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.retired_worker = self.worker
            self.pending_camera = True
            self.worker = None
        self.frame, self.detections, self.drag = None, [], None
        self.draw()

    def capture_frame(self):
        if self.worker is None or self.frame is None:
            raise ValueError("請先開啟攝影機預覽並等待畫面。")
        image = self.frame.copy()
        self.stop_camera()
        self.load(self.dataset.add_image(image)); self.refresh_list()

    def pick_weights(self, variable):
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(ROOT), filetypes=[("YOLO 權重", "*.pt")])
        if path:
            variable.set(str(project_path(path)))

    def export(self):
        if not self.may_leave():
            return
        self.prepare_export()

    def prepare_export(self, training=None):
        if self.preparing:
            raise ValueError("資料匯出準備中，請等候完成。")
        snapshot = LabelDataset(self.dataset.folder)
        snapshot.data = copy.deepcopy(self.dataset.data)
        self.preparing = True
        self.status.set("正在背景匯出與檢查標註資料…")
        def work():
            try:
                from object_detection_train import validate_data
                path = snapshot.export()
                validate_data(path)
                self.tasks.put(("exported", path, training))
            except Exception as exc:
                self.tasks.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def start_train(self):
        if self.process is not None or self.preparing:
            raise ValueError("已有訓練執行中。")
        epochs, batch = int(self.epochs.get()), int(self.batch.get())
        if min(epochs, batch) <= 0:
            raise ValueError("epochs 與 batch 必須大於零。")
        weights = project_path(self.weights.get())
        if not weights.is_file():
            raise ValueError("起始權重不存在。")
        if not self.may_leave():
            return
        self.stop_camera()
        self.prepare_export((weights, epochs, batch))

    def launch_training(self, path, training):
        weights, epochs, batch = training
        run_id = uuid.uuid4().hex
        self.run_dir = getattr(self, "runs_root", ROOT / "tranning" / "object_detection_runs") / run_id
        logs = project_path(ROOT / "output" / "object_detection_logs")
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"{run_id}.log"
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        command = [sys.executable, "-u", str(ROOT / "tranning" / "object_detection_train.py"), "--data", str(path), "--weights", str(weights), "--out-dir", str(self.run_dir), "--epochs", str(epochs), "--batch", str(batch)]
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except Exception:
            self.log_handle.close(); self.log_handle = None
            raise
        self.status.set(f"微調中：資料 {path}；輸出 {self.run_dir}；紀錄 {self.log_path}")

    def stop_train(self):
        if self.process is not None:
            self.process.terminate()
            self.status.set("已要求停止訓練，等待程序結束。")

    def poll(self):
        try:
            task = self.tasks.get_nowait()
            self.preparing = False
            if task[0] == "error":
                messagebox.showerror("資料匯出", task[1], parent=self.root)
                self.status.set("資料尚未就緒，未開始訓練。")
            elif task[2] is None:
                self.status.set(f"資料集已匯出：{task[1]}")
            elif self.pending_camera and self.retired_worker.thread.is_alive():
                self.preparing = True
                self.tasks.put(task)
                self.status.set("資料已就緒，等待釋放鏡頭後開始訓練…")
            else:
                self.safe(lambda: self.launch_training(task[1], task[2]))
        except queue.Empty:
            pass
        if self.pending_camera and not self.retired_worker.thread.is_alive():
            self.pending_camera = False
        if self.worker is not None:
            try:
                message = self.worker.messages.get_nowait()
                if message[0] == "error":
                    self.stop_camera()
                    messagebox.showerror("攝影機", message[1], parent=self.root)
                else:
                    _, self.frame, self.detections = message
                    self.draw()
            except queue.Empty:
                pass
        if self.process is not None and self.process.poll() is not None:
            code, self.process = self.process.returncode, None
            self.log_handle.close(); self.log_handle = None
            best = self.run_dir / "weights" / "best.pt"
            if code == 0 and best.is_file():
                self.safe(lambda: self.training_finished(best))
            else:
                self.status.set(f"訓練未完成（結束碼 {code}），請查看 {self.log_path}")
        self.timer = self.root.after(50, self.poll)

    def training_finished(self, best):
        self.best.set(str(best))
        self.status.set(f"微調結束，已選擇 {best}。請用不同場景驗證辨識結果。")

    def close(self):
        if self.closing:
            return
        if not self.may_leave():
            return
        if self.process is not None or self.preparing:
            messagebox.showinfo("工作執行中", "請等待資料匯出完成；若正在訓練，先停止並等待程序結束後再關閉。", parent=self.root)
            return
        self.closing = True
        self.stop_camera()
        self.finish_close()

    def finish_close(self):
        if self.pending_camera and self.retired_worker.thread.is_alive():
            self.status.set("正在釋放攝影機，請稍候…")
            self.root.after(50, self.finish_close)
            return
        self.root.after_cancel(self.timer)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = tk.Tk()
    Labeler(root, args.dataset)
    root.mainloop()


if __name__ == "__main__":
    main()
