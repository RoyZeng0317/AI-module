"""獨立人物圖片識別：Tkinter預覽、characters.pt預測與繁體中文標記。"""
import argparse, contextlib, io, queue, sys, threading, numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# ------------------------------------------------------------------
# read the image and saved model
project_dir = Path(__file__).resolve().parents[1]
test_img_dir = None  # 可改成 project_dir / 'character_test/photo.jpg'；None則在視窗匯入圖片
model_path = project_dir / 'tranning/yolo_runs/custom/weights/characters.pt'
# Hyperparameter setting
confidence = 0.25
imgsz = 640


# ------------------------------------------------------------------
# prepare the test image
def prepare_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到圖片：{path}')
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGBA')
        background = Image.new('RGBA', image.size, 'white')
        image = Image.alpha_composite(background, image).convert('RGB')
        return np.asarray(image)[:, :, ::-1].copy()  # YOLO的NumPy輸入使用BGR，不套用MNIST灰階前處理


# ------------------------------------------------------------------
# load the saved model
def get_model():
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError('characters.pt不存在或是空檔，請先完成人物模型訓練。')
    from ultralytics import YOLO
    return YOLO(str(model_path))


# ------------------------------------------------------------------
# import a test image
def choose_image():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(title='選擇人物圖片', filetypes=[('圖片', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff')])
    finally:
        root.destroy()


# ------------------------------------------------------------------
# predict the test image
def detect_image(path, conf=confidence):
    if not 0 < conf < 1:
        raise ValueError('信心門檻必須介於0與1之間。')
    image = prepare_image(path)
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        result = get_model().predict(image, conf=conf, imgsz=imgsz,
                                     device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
    return image, result


def predict_image(path, conf=confidence):
    _, result = detect_image(path, conf)
    return [result.names[int(box.cls[0])] for box in result.boxes]


# ------------------------------------------------------------------
# show the image and prediction in a tkinter window
class CharacterImageWindow:
    def __init__(self, root, path=None, conf=confidence):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.path, self.preview = root, None, None
        self.busy, self.closing, self.worker = False, False, None
        self.messages, self.photo = queue.Queue(), None
        self.conf = tk.StringVar(value=str(conf))
        self.result = tk.StringVar(value='尚未辨識')
        self.status = tk.StringVar(value='請匯入人物圖片，使用characters.pt辨識。')
        root.title('人物圖片識別 — characters.pt')
        root.geometry('1000x760'); root.minsize(640, 480)
        root.protocol('WM_DELETE_WINDOW', self.close)
        toolbar = ttk.Frame(root); toolbar.pack(fill='x', padx=8, pady=8)
        self.import_button = ttk.Button(toolbar, text='匯入圖片', command=self.open_image)
        self.import_button.pack(side='left')
        self.retry_button = ttk.Button(toolbar, text='重新辨識', command=lambda: self.start(self.path) if self.path else None)
        self.retry_button.pack(side='left', padx=6)
        ttk.Label(toolbar, text='偵測門檻').pack(side='left')
        ttk.Entry(toolbar, textvariable=self.conf, width=6).pack(side='left', padx=6)
        ttk.Label(root, text=f'人物模型：{model_path}', wraplength=950).pack(fill='x', padx=8)
        self.canvas = tk.Canvas(root, bg='#20242b', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=8)
        self.canvas.bind('<Configure>', lambda _: self.draw())
        ttk.Label(root, textvariable=self.result, wraplength=950, font=('Microsoft JhengHei', 13)).pack(fill='x', padx=8)
        ttk.Label(root, textvariable=self.status, wraplength=950).pack(fill='x', padx=8, pady=8)
        self.timer = root.after(50, self.poll)
        if path:
            root.after_idle(lambda: self.start(path))

    def open_image(self):
        from tkinter import filedialog
        if not self.busy and not self.closing:
            path = filedialog.askopenfilename(parent=self.root, title='匯入人物圖片', filetypes=[('圖片', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff')])
            if path:
                self.start(path)

    def start(self, path):
        from tkinter import messagebox
        if self.busy or self.closing:
            return
        try:
            conf = float(self.conf.get())
            if not 0 < conf < 1:
                raise ValueError('信心門檻必須介於0與1之間。')
        except ValueError as exc:
            messagebox.showerror('人物圖片識別', str(exc), parent=self.root)
            return
        self.path, self.busy, self.preview = Path(path), True, None
        self.canvas.delete('all'); self.result.set('辨識中…')
        self.status.set(f'正在讀取與辨識：{self.path}')
        self.import_button.configure(state='disabled'); self.retry_button.configure(state='disabled')
        def work():
            try:
                image, result = detect_image(path, conf)
                if str(project_dir) not in sys.path:
                    sys.path.insert(0, str(project_dir))
                from lib.yolo.detection_overlay import draw_detections
                marked = draw_detections(image, result, empty_label='未偵測到人物：可調整門檻後重新辨識')
                labels = [f'{result.names[int(box.cls[0])]}　{float(box.conf[0]):.1%}' for box in result.boxes]
                self.messages.put(('result', Image.fromarray(marked[:, :, ::-1]), labels))
            except Exception as exc:
                self.messages.put(('error', str(exc)))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def draw(self):
        from PIL import ImageTk
        self.canvas.delete('all')
        if self.preview is not None:
            image = self.preview.copy()
            width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(width//2, height//2, image=self.photo)

    def poll(self):
        from tkinter import messagebox
        if self.closing:
            if self.worker is not None and self.worker.is_alive():
                self.timer = self.root.after(50, self.poll)
            else:
                self.root.destroy()
            return
        try:
            message = self.messages.get_nowait()
            self.busy = False
            self.import_button.configure(state='normal'); self.retry_button.configure(state='normal')
            if message[0] == 'error':
                self.result.set('辨識失敗'); self.status.set(message[1])
                messagebox.showerror('人物圖片識別', message[1], parent=self.root)
            else:
                self.preview = message[1]; self.draw()
                self.result.set('模型預測：' + '、'.join(message[2]) if message[2] else '未偵測到人物')
                self.status.set(f'{self.path}｜模型預測需人工確認，分數不是身份準確率。')
        except queue.Empty:
            pass
        self.timer = self.root.after(50, self.poll)

    def close(self):
        self.closing = True
        self.import_button.configure(state='disabled'); self.retry_button.configure(state='disabled')
        self.status.set('正在結束辨識，請稍候…')
        self.root.after_cancel(self.timer)
        self.poll()  # 以after等待背景完成，避免銷毀視窗後仍呼叫Tk


def main(path=None, conf=confidence, gui=True):
    if not 0 < conf < 1:
        raise ValueError('信心門檻必須介於0與1之間。')
    if gui:
        import tkinter as tk
        root = tk.Tk()
        CharacterImageWindow(root, path or test_img_dir, conf)
        root.mainloop()
        return
    path = path or test_img_dir or choose_image()
    if not path:  # 取消選圖時不載入模型
        return
    labels = predict_image(path, conf)
    print('\n'.join(labels) if labels else '未偵測到人物')


# ------------------------------------------------------------------
# command line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tkinter人物圖片預覽與characters.pt辨識結果。')
    parser.add_argument('image', nargs='?', type=Path, help='圖片路徑；未指定時在視窗按「匯入圖片」')
    parser.add_argument('--conf', type=float, default=confidence, help='信心門檻，預設0.25')
    parser.add_argument('--no-gui', action='store_true', help='僅輸出文字，不開啟結果視窗')
    args = parser.parse_args()
    try:
        main(args.image, args.conf, gui=not args.no_gui)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'人物圖片識別失敗：{exc}\n')
