"""Training GUI — pick a task, point at your data, tune hyperparameters, and
launch/monitor training without memorizing each script's CLI flags.

Wraps road_sign_train.py, OCR.py, circuit_diagram_train.py, speech_to_text.py,
chats.py, character_model.py, voice_clone.py, kicad_dataset_convert.py,
image_classifier_bnn.py and data_split.py. Each "開始訓練"/"匯入" button runs the *exact* command
line that script's own module docstring documents under "Usage" — as a real
subprocess (`python -u <script>.py --flag value ...`), not an in-process
function call. That keeps this one GUI process light (it never imports
torch/torchvision/ultralytics itself) and means Stop can just kill the
subprocess outright, no cooperative-cancellation plumbing needed inside the
training scripts themselves.

Only one job runs at a time — a single RTX 4060 8GB can't usefully train two
models at once anyway (Rule 06), so the whole app shares one subprocess slot,
one log console, and one Stop button across every tab.

Caveat surfaced in the UI itself: killing a running job does NOT save a
checkpoint. best_model.pt / encoder.pt+decoder.pt are only written once the
training loop finishes on its own (full epoch count or early stopping) —
see train_utils.save_checkpoint(). Stop is for abandoning a run, not pausing
one you want to keep.

Run: python train_gui.py
"""

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

TRANNING_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

BG = "#191a1b"
PANEL_BG = "#232425"
FIELD_BG = "#2d2f31"
FG = "white"
MUTED_FG = "#9aa0a6"
ACCENT = "#3b82f6"
STOP_RED = "#dc2626"


@dataclass
class Field:
    """One form row: a label, an input widget, and (if `flag` is set) how it
    maps onto a CLI argument when the job is launched.
    """

    key: str
    label: str
    kind: str  # "dir" | "file" | "save" | "str" | "int" | "float" | "bool"
    flag: str | None
    default: object = ""
    required: bool = False
    filetypes: tuple = ()
    hint: str = ""
    var: tk.Variable | None = None  # attached once the widget is built


def _browse_dir(var: tk.StringVar):
    path = filedialog.askdirectory()
    if path:
        var.set(path)


def _browse_file(var: tk.StringVar, filetypes):
    path = filedialog.askopenfilename(filetypes=filetypes or (("All files", "*.*"),))
    if path:
        var.set(path)


def _browse_save(var: tk.StringVar, filetypes):
    """跟 _browse_file 不同：這是給「還不存在、將要被寫出來」的檔案用的
    （dataset_import.py 的 --out），用 asksaveasfilename 而不是
    askopenfilename，不然使用者選不到一個還沒被建立的檔名。
    """
    path = filedialog.asksaveasfilename(filetypes=filetypes or (("All files", "*.*"),),
                                          defaultextension=".json")
    if path:
        var.set(path)


def build_form(parent: tk.Widget, fields: list[Field], start_row: int = 0) -> int:
    """Lays out `fields` as label/entry(/browse) rows starting at `start_row`,
    attaching a tk variable to each field. Returns the next free row.
    """
    parent.columnconfigure(1, weight=1)
    row = start_row
    for f in fields:
        label_text = f.label + ("" if not f.required else " *")
        tk.Label(parent, text=label_text, bg=PANEL_BG, fg=FG, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(10, 6), pady=4
        )

        if f.kind == "bool":
            f.var = tk.BooleanVar(value=bool(f.default))
            ttk.Checkbutton(parent, variable=f.var).grid(row=row, column=1, sticky="w", pady=4)
        else:
            f.var = tk.StringVar(value=str(f.default))
            entry = tk.Entry(parent, textvariable=f.var, bg=FIELD_BG, fg=FG,
                              insertbackground=FG, relief="flat")
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=4)

            if f.kind == "dir":
                ttk.Button(parent, text="瀏覽…", command=lambda v=f.var: _browse_dir(v)).grid(
                    row=row, column=2, pady=4
                )
            elif f.kind == "file":
                ttk.Button(parent, text="瀏覽…",
                           command=lambda v=f.var, ft=f.filetypes: _browse_file(v, ft)).grid(
                    row=row, column=2, pady=4
                )
            elif f.kind == "save":
                ttk.Button(parent, text="另存為…",
                           command=lambda v=f.var, ft=f.filetypes: _browse_save(v, ft)).grid(
                    row=row, column=2, pady=4
                )

        if f.hint:
            row += 1
            tk.Label(parent, text=f.hint, bg=PANEL_BG, fg=MUTED_FG, anchor="w",
                      wraplength=520, justify="left").grid(
                row=row, column=0, columnspan=3, sticky="w", padx=(10, 6), pady=(0, 6)
            )
        row += 1
    return row


def collect_values(fields: list[Field]) -> dict | None:
    """Reads each field's tk variable, validates type/required, and shows a
    messagebox + returns None on the first problem found.
    """
    values = {}
    for f in fields:
        assert f.var is not None, f"{f.key}: build_form() must run before collect_values()"

        if f.kind == "bool":
            values[f.key] = bool(f.var.get())
            continue

        raw = str(f.var.get()).strip()
        if f.required and not raw:
            messagebox.showerror("缺少欄位", f"請填寫「{f.label}」")
            return None
        if not raw:
            values[f.key] = ""
            continue

        if f.kind == "int":
            try:
                values[f.key] = int(raw)
            except ValueError:
                messagebox.showerror("輸入錯誤", f"「{f.label}」必須是整數")
                return None
        elif f.kind == "float":
            try:
                values[f.key] = float(raw)
            except ValueError:
                messagebox.showerror("輸入錯誤", f"「{f.label}」必須是數字")
                return None
        else:
            values[f.key] = raw
    return values


def build_argv(fields: list[Field], values: dict) -> list[str]:
    argv = []
    for f in fields:
        if f.flag is None:
            continue
        v = values[f.key]
        if f.kind == "bool":
            if v:
                argv.append(f.flag)
        elif v != "":
            argv += [f.flag, str(v)]
    return argv


# ---------------------------------------------------------------------------
# Per-task field definitions — mirror each script's argparse defaults exactly
# (see that script's `main()`), so an unmodified field reproduces its CLI
# default behaviour.
# ---------------------------------------------------------------------------

JSON_FT = (("JSON files", "*.json"), ("All files", "*.*"))
YAML_FT = (("YAML files", "*.yaml *.yml"), ("All files", "*.*"))


# ---------------------------------------------------------------------------
# dataset_import.py 的「步驟零」欄位——手上沒有現成 JSON manifest，只有一般
# 檔案（.txt / 圖片+.txt / 音檔+.txt / 影片）時，先在這裡自動轉換出 manifest
# 或圖片資料夾，再貼到下面既有的訓練欄位裡（CLAUDE.md 需求 #04）。用工廠
# function 而非共用同一份 list——同一批 Field 物件被兩個分頁共用會互相
# 偷走 tk 變數，這是 CLAUDE.md to-do #16 已經踩過、寫進去的坑。
# ---------------------------------------------------------------------------

def _pairs_import_fields(out_default: str) -> list[Field]:
    return [
        Field("source", "原始 .txt 資料夾（每個檔案一組 prompt/reply）", "dir", "--source", required=True,
              hint="每個 .txt 檔案格式二選一：(a) 第一行是 prompt，空一行後面是 reply；"
                   "(b) 只有兩行，第一行 prompt、第二行 reply。檔名本身不重要。"),
        Field("out", "輸出 manifest 路徑", "save", "--out", required=True, default=out_default,
              filetypes=JSON_FT, hint="完成後把這個路徑（或下面切出來的 _train.json/_val.json）"
                                       "貼到下方的 manifest 欄位。"),
        Field("val_ratio", "驗證集比例（0 = 不切分，只寫一個檔案）", "float", "--val-ratio", default=0.0),
    ]


def _sidecar_import_fields(kind: str, out_default: str) -> list[Field]:
    noun = "圖片" if kind == "image" else "音檔（16-bit PCM WAV）"
    return [
        Field("source", f"原始{noun}資料夾（每個檔案配一個同檔名 .txt 文字稿）", "dir", "--source",
              required=True, hint="例如 sign01.png + sign01.txt，或 line01.wav + line01.txt。"),
        Field("out", "輸出 manifest 路徑", "save", "--out", required=True, default=out_default,
              filetypes=JSON_FT, hint="完成後把這個路徑（或下面切出來的 _train.json/_val.json）"
                                       "貼到下方的 manifest 欄位。"),
        Field("val_ratio", "驗證集比例（0 = 不切分，只寫一個檔案）", "float", "--val-ratio", default=0.0),
    ]


def _video_import_fields() -> list[Field]:
    return [
        Field("source", "原始影片資料夾（每個子資料夾 = 一個類別，內含影片檔）", "dir", "--source",
              required=True, hint="例如 videos/停止/xxx.mp4、videos/讓路/yyy.mp4 ……"),
        Field("out", "輸出圖片資料夾", "dir", "--out", required=True,
              hint="完成後把這個路徑貼到下面「步驟一」的來源資料夾繼續分割。"),
        Field("interval", "擷取間隔（秒）", "float", "--interval", default=1.0),
    ]

SPLIT_FIELDS = [
    Field("source", "來源資料夾（每個子資料夾一個類別）", "dir", "--source", required=True),
    Field("output", "輸出資料夾", "dir", "--output", required=True,
          hint="會在裡面產生 train/ val/ test/ 三個子資料夾，交給下方「訓練模型」的「資料夾」使用。"),
    Field("train", "訓練集比例", "float", "--train", default=0.7),
    Field("val", "驗證集比例", "float", "--val", default=0.15),
    Field("test", "測試集比例", "float", "--test", default=0.15),
    Field("seed", "隨機種子", "int", "--seed", default=42),
    Field("move", "搬移檔案（取消勾選=複製）", "bool", "--move", default=False),
]

ROAD_SIGN_FIELDS = [
    Field("data_dir", "資料夾（--source/--output 分割後的輸出）", "dir", "--data-dir", required=True),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="road_sign_runs"),
    Field("epochs", "Epochs", "int", "--epochs", default=30),
    Field("batch_size", "Batch size", "int", "--batch-size", default=32),
    Field("lr", "學習率 (lr)", "float", "--lr", default=3e-4),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=5),
    Field("image_size", "圖片邊長 (image size)", "int", "--image-size", default=224),
    Field("dropout", "Dropout", "float", "--dropout", default=0.3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
    Field("unfreeze_backbone", "解凍 backbone（train_acc 一直偏低才勾）", "bool",
          "--unfreeze-backbone", default=False),
]

OCR_FIELDS = [
    Field("train_manifest", "訓練 manifest (JSON)", "file", "--train-manifest",
          required=True, filetypes=JSON_FT),
    Field("val_manifest", "驗證 manifest (JSON)", "file", "--val-manifest",
          required=True, filetypes=JSON_FT),
    Field("images_root", "圖片根目錄（留空 = manifest 所在資料夾）", "dir", "--images-root"),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="ocr_runs"),
    Field("epochs", "Epochs", "int", "--epochs", default=50),
    Field("batch_size", "Batch size", "int", "--batch-size", default=16),
    Field("lr", "學習率 (lr)", "float", "--lr", default=1e-3),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=6),
    Field("img_height", "圖片高度", "int", "--img-height", default=32),
    Field("img_width", "圖片寬度", "int", "--img-width", default=128),
    Field("hidden_size", "LSTM hidden size", "int", "--hidden-size", default=256),
    Field("dropout", "Dropout", "float", "--dropout", default=0.2),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
]

SPEECH_FIELDS = [
    Field("train_manifest", "訓練 manifest (JSON)", "file", "--train-manifest",
          required=True, filetypes=JSON_FT),
    Field("val_manifest", "驗證 manifest (JSON)", "file", "--val-manifest",
          required=True, filetypes=JSON_FT),
    Field("audio_root", "音檔根目錄（留空 = manifest 所在資料夾）", "dir", "--audio-root"),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="speech_runs"),
    Field("epochs", "Epochs", "int", "--epochs", default=50),
    Field("batch_size", "Batch size", "int", "--batch-size", default=16),
    Field("lr", "學習率 (lr)", "float", "--lr", default=1e-3),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=6),
    Field("max_frames", "最大時間幀數 (max frames)", "int", "--max-frames", default=200),
    Field("hidden_size", "LSTM hidden size", "int", "--hidden-size", default=256),
    Field("dropout", "Dropout", "float", "--dropout", default=0.2),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
]

CHAT_FIELDS = [
    Field("data", "對話資料 (JSON, [{prompt, reply}, ...])", "file", "--data",
          required=True, filetypes=JSON_FT),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="chat_runs"),
    Field("epochs", "Epochs", "int", "--epochs", default=50,
          hint="資料筆數變多時通常要拉高很多（例如 28 筆資料約需 1500 epochs 才會收斂，見 CLAUDE.md 修正日誌）。"),
    Field("batch_size", "Batch size", "int", "--batch-size", default=8),
    Field("embed_size", "Embedding size", "int", "--embed-size", default=64),
    Field("hidden_size", "GRU hidden size", "int", "--hidden-size", default=128),
    Field("lr", "學習率 (lr)", "float", "--lr", default=1e-3),
    Field("max_len", "最大字元長度 (max len)", "int", "--max-len", default=40),
    Field("teacher_forcing_ratio", "Teacher forcing 比例", "float",
          "--teacher-forcing-ratio", default=0.5),
    Field("dropout", "Dropout（貝氏 MC Dropout：聊天時保持開啟、重複取樣估計信心度）", "float",
          "--dropout", default=0.3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
]

CHARACTER_FIELDS = [
    Field("train_manifest", "訓練 manifest (JSON)", "file", "--train-manifest",
          required=True, filetypes=JSON_FT),
    Field("val_manifest", "驗證 manifest (JSON)", "file", "--val-manifest",
          required=True, filetypes=JSON_FT),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="characters/character_runs"),
    Field("characters_dir", "character card 輸出資料夾", "str", "--characters-dir", default="characters"),
    Field("epochs", "Epochs", "int", "--epochs", default=50),
    Field("batch_size", "Batch size", "int", "--batch-size", default=8),
    Field("embed_size", "Embedding size", "int", "--embed-size", default=64),
    Field("hidden_size", "GRU hidden size", "int", "--hidden-size", default=128),
    Field("lr", "學習率 (lr)", "float", "--lr", default=1e-3),
    Field("max_len", "最大字元長度 (max len)", "int", "--max-len", default=80),
    Field("dropout", "Dropout", "float", "--dropout", default=0.3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=6),
]

VOICE_CLONE_FIELDS = [
    Field("train_manifest", "訓練 manifest (JSON)", "file", "--train-manifest",
          required=True, filetypes=JSON_FT),
    Field("val_manifest", "驗證 manifest (JSON)", "file", "--val-manifest",
          required=True, filetypes=JSON_FT),
    Field("audio_root", "音檔根目錄（留空 = manifest 所在資料夾）", "dir", "--audio-root"),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="characters/character_voice_runs/角色名稱",
          hint="每個角色的聲音各自獨立訓練（single-speaker），請把「角色名稱」換成實際角色，"
               "例如 characters/character_voice_runs/周柯宇。"),
    Field("epochs", "Epochs", "int", "--epochs", default=300),
    Field("batch_size", "Batch size", "int", "--batch-size", default=4),
    Field("lr", "學習率 (lr)", "float", "--lr", default=1e-3),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=10),
    Field("max_len", "最大文字長度 (max len)", "int", "--max-len", default=60),
    Field("max_frames", "最大頻譜時間幀數 (max frames)", "int", "--max-frames", default=200),
    Field("embed_size", "文字 Embedding size", "int", "--embed-size", default=64),
    Field("hidden_size", "GRU hidden size", "int", "--hidden-size", default=256),
    Field("prenet_size", "Prenet size", "int", "--prenet-size", default=128),
    Field("dropout", "Dropout", "float", "--dropout", default=0.3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
    Field("teacher_forcing_ratio", "Teacher forcing 比例", "float",
          "--teacher-forcing-ratio", default=1.0),
    Field("stop_loss_weight", "Stop token loss 權重", "float", "--stop-loss-weight", default=1.0),
]

KICAD_FT = (("KiCad schematic", "*.kicad_sch"), ("All files", "*.*"))

KICAD_COMMON_FIELDS = [
    Field("out_dir", "輸出資料夾（YOLO 資料集）", "dir", "--out-dir", required=True,
          hint="轉換完成後，把這個路徑複製貼到下方「步驟二」的「資料夾 + 類別列表」模式的資料夾欄位，"
               "類別名稱也要貼成一樣的清單，兩邊必須對齊（label id 由順序決定）。"),
    Field("classes", "類別名稱（用空格或逗號分隔，順序 = label id）", "str", None,
          default="resistor capacitor inductor diode ic transistor battery switch led wire_junction"),
    Field("class_map", "class-map JSON（可留空，用內建的 DEFAULT_CLASS_MAP）", "file", "--class-map", filetypes=JSON_FT,
          hint="你的 .kicad_sch 如果是用專案自帶符號庫（例如 interf_u:R 而不是 Device:R），"
               "轉換完成後主控台會列出沒對到的 lib_id，需要用這個 JSON 檔補上對應關係。"),
    Field("resolution", "匯出解析度 (DPI)", "int", "--resolution", default=300),
    Field("kicad_cli", "kicad-cli 執行檔路徑（留空 = 從 PATH 找 kicad-cli）", "file", "--kicad-cli",
          filetypes=(("kicad-cli.exe", "kicad-cli.exe"), ("All files", "*.*"))),
    Field("exclude_drawing_sheet", "排除圖框／標題欄", "bool", "--exclude-drawing-sheet", default=False),
    Field("val_ratio", "驗證集比例 (val ratio)", "float", "--val-ratio", default=0.2),
    Field("seed", "隨機種子", "int", "--seed", default=42),
]

CIRCUIT_COMMON_FIELDS = [
    Field("weights", "起始權重（留空 = COCO 版 yolo11n.pt）", "file", "--weights"),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="circuit_diagram_runs/train"),
    Field("epochs", "Epochs", "int", "--epochs", default=50),
    Field("batch", "Batch size", "int", "--batch", default=16),
    Field("imgsz", "圖片邊長 (imgsz)", "int", "--imgsz", default=640),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=15),
    Field("freeze", "凍結前 N 層 backbone（0 = 全部解凍）", "int", "--freeze", default=10),
    Field("lr0", "初始學習率 (lr0)", "float", "--lr0", default=1e-3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=5e-4),
    Field("device", "裝置（留空 = 自動挑選，'0'=第一張GPU，'cpu'=強制CPU）", "str", "--device"),
]


IMAGE_SPLIT_FIELDS = [
    Field("source", "來源資料夾（每個子資料夾一個類別）", "dir", "--source", required=True),
    Field("output", "輸出資料夾", "dir", "--output", required=True,
          hint="會在裡面產生 train/ val/ test/ 三個子資料夾，交給下方「訓練模型」的「資料夾」使用。"),
    Field("train", "訓練集比例", "float", "--train", default=0.7),
    Field("val", "驗證集比例", "float", "--val", default=0.15),
    Field("test", "測試集比例", "float", "--test", default=0.15),
    Field("seed", "隨機種子", "int", "--seed", default=42),
    Field("move", "搬移檔案（取消勾選=複製）", "bool", "--move", default=False),
]

IMAGE_BNN_FIELDS = [
    Field("data_dir", "資料夾（--source/--output 分割後的輸出）", "dir", "--data-dir", required=True),
    Field("out_dir", "輸出資料夾", "str", "--out-dir", default="image_classifier_runs"),
    Field("epochs", "Epochs", "int", "--epochs", default=30),
    Field("batch_size", "Batch size", "int", "--batch-size", default=32),
    Field("lr", "學習率 (lr)", "float", "--lr", default=3e-4),
    Field("patience", "Early stopping 耐心值", "int", "--patience", default=5),
    Field("image_size", "圖片邊長 (image size)", "int", "--image-size", default=224),
    Field("dropout", "Dropout（貝氏 MC Dropout：推論時同一層繼續發揮作用）", "float",
          "--dropout", default=0.3),
    Field("weight_decay", "Weight decay", "float", "--weight-decay", default=1e-4),
    Field("unfreeze_backbone", "解凍 backbone（train_acc 一直偏低才勾）", "bool",
          "--unfreeze-backbone", default=False),
    Field("no_hflip", "停用水平翻轉增強（類別有方向性時勾選，例如箭頭/文字）", "bool",
          "--no-hflip", default=False),
]


class TrainGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.stopping = False
        self.log_queue: queue.Queue = queue.Queue()
        self.start_buttons: list[ttk.Button] = []

        root.title("sinco 訓練工具")
        root.geometry("880x720")
        root.configure(bg=BG)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_BG, foreground=FG, padding=(14, 6))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)])
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TButton", padding=6, background=FIELD_BG, foreground=FG)
        style.map("TButton", background=[("active", ACCENT)])
        style.configure("TCheckbutton", background=PANEL_BG, foreground=FG)
        style.configure("TRadiobutton", background=PANEL_BG, foreground=FG)

        notebook = ttk.Notebook(root)
        notebook.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 6))

        self._build_road_sign_tab(notebook)
        self._build_image_bnn_tab(notebook)
        self._build_ocr_tab(notebook)
        self._build_circuit_tab(notebook)
        self._build_speech_tab(notebook)
        self._build_chat_tab(notebook)
        self._build_character_tab(notebook)
        self._build_voice_clone_tab(notebook)

        self._build_status_bar()
        self._build_log_console()

        self.root.after(80, self._poll_log)

    # -- tabs ----------------------------------------------------------

    def _new_tab(self, notebook: ttk.Notebook, title: str, subtitle: str) -> tk.Frame:
        outer = tk.Frame(notebook, bg=PANEL_BG)
        notebook.add(outer, text=title)
        tk.Label(outer, text=subtitle, bg=PANEL_BG, fg=MUTED_FG, anchor="w",
                  wraplength=820, justify="left").pack(fill="x", padx=10, pady=(10, 4))
        return outer

    def _build_road_sign_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "路牌辨識",
            "MobileNetV2 遷移學習分類器 (road_sign_train.py)。原始圖片先用「步驟一」依類別子資料夾"
            "分割成 train/val/test，再用「步驟二」指到分割後的輸出資料夾進行訓練。手上只有影片、"
            "還沒有整理成一個子資料夾一個類別的圖片時，先用「步驟零」從影片擷取影格。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從影片擷取影格 (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        road_sign_video_fields = _video_import_fields()
        build_form(step0, road_sign_video_fields)
        video_btn = ttk.Button(
            step0, text="擷取影格",
            command=lambda: self._start_import("路牌影格擷取", "video-frames", road_sign_video_fields),
        )
        video_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(video_btn)

        step1 = tk.LabelFrame(outer, text=" 步驟一：分割資料集 (data_split.py) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step1.pack(fill="x", padx=10, pady=(0, 10))
        build_form(step1, SPLIT_FIELDS)
        split_btn = ttk.Button(step1, text="分割資料集",
                                command=lambda: self._start("資料集分割", "data_split.py", SPLIT_FIELDS))
        split_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(split_btn)

        step2 = tk.LabelFrame(outer, text=" 步驟二：訓練模型 (road_sign_train.py) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step2.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        build_form(step2, ROAD_SIGN_FIELDS)
        btn = ttk.Button(step2, text="開始訓練",
                          command=lambda: self._start("路牌辨識訓練", "road_sign_train.py", ROAD_SIGN_FIELDS))
        btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(btn)

    def _build_image_bnn_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "圖像分類（貝氏）",
            "貝氏（MC Dropout）MobileNetV2 遷移學習分類器 (image_classifier_bnn.py)，"
            "適用任何「一個子資料夾一個類別」的通用圖像分類任務（不限路牌，路牌請仍用"
            "「路牌辨識」分頁）。原始圖片先用「步驟一」依類別子資料夾分割成 train/val/test，"
            "再用「步驟二」指到分割後的輸出資料夾進行訓練。訓練用的 Dropout 層在推論時會"
            "刻意保持開啟、重複取樣多次（MC Dropout），藉此估計「模型有多不確定」而不是"
            "只給一個看似武斷的答案。訓練完成後，若要對單一圖片做這種貝氏推論"
            "（含信心度/熵），請用終端機另外執行 "
            "`python image_classifier_bnn.py --classify <image> --out-dir ... --mc-samples 20`"
            "（此 GUI 只負責訓練）。手上只有影片、還沒有整理成一個子資料夾一個類別的圖片時，"
            "先用「步驟零」從影片擷取影格。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從影片擷取影格 (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        image_bnn_video_fields = _video_import_fields()
        build_form(step0, image_bnn_video_fields)
        video_btn = ttk.Button(
            step0, text="擷取影格",
            command=lambda: self._start_import("圖像分類影格擷取", "video-frames", image_bnn_video_fields),
        )
        video_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(video_btn)

        step1 = tk.LabelFrame(outer, text=" 步驟一：分割資料集 (data_split.py) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step1.pack(fill="x", padx=10, pady=(0, 10))
        build_form(step1, IMAGE_SPLIT_FIELDS)
        split_btn = ttk.Button(step1, text="分割資料集",
                                command=lambda: self._start("資料集分割", "data_split.py", IMAGE_SPLIT_FIELDS))
        split_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(split_btn)

        step2 = tk.LabelFrame(outer, text=" 步驟二：訓練模型 (image_classifier_bnn.py) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step2.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        build_form(step2, IMAGE_BNN_FIELDS)
        btn = ttk.Button(step2, text="開始訓練",
                          command=lambda: self._start("圖像分類訓練", "image_classifier_bnn.py", IMAGE_BNN_FIELDS))
        btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(btn)

    def _build_ocr_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "OCR 文字辨識",
            "從零打造的 CRNN + CTC 文字辨識模型 (OCR.py)。manifest 是一個 JSON 清單，"
            '每筆為 {"image": 檔名, "text": 文字內容}。手上沒有現成 JSON 的話，'
            "先用「步驟零」從圖片 + 同檔名 .txt 文字稿自動產生 manifest。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從圖片＋文字稿匯入 manifest (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        ocr_import_fields = _sidecar_import_fields("image", "ocr_manifest.json")
        build_form(step0, ocr_import_fields)
        import_btn = ttk.Button(
            step0, text="產生 manifest",
            command=lambda: self._start_import("OCR manifest 匯入", "sidecar-image", ocr_import_fields),
        )
        import_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(import_btn)

        form = tk.Frame(outer, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=10)
        build_form(form, OCR_FIELDS)
        btn = ttk.Button(outer, text="開始訓練",
                          command=lambda: self._start("OCR 訓練", "OCR.py", OCR_FIELDS))
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _build_speech_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "語音辨識",
            "重用 OCR.py 的 CRNN + CTC 架構，輸入換成語音頻譜圖 (speech_to_text.py)。"
            'manifest 每筆為 {"audio": 檔名(16-bit PCM WAV), "text": 文字內容}。'
            "手上沒有現成 JSON 的話，先用「步驟零」從錄音 + 同檔名 .txt 逐字稿自動產生 manifest。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從錄音＋逐字稿匯入 manifest (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        speech_import_fields = _sidecar_import_fields("audio", "speech_manifest.json")
        build_form(step0, speech_import_fields)
        import_btn = ttk.Button(
            step0, text="產生 manifest",
            command=lambda: self._start_import("語音 manifest 匯入", "sidecar-audio", speech_import_fields),
        )
        import_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(import_btn)

        form = tk.Frame(outer, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=10)
        build_form(form, SPEECH_FIELDS)
        btn = ttk.Button(outer, text="開始訓練",
                          command=lambda: self._start("語音辨識訓練", "speech_to_text.py", SPEECH_FIELDS))
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _build_chat_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "聊天模型",
            "從零打造的 GRU 注意力 seq2seq 聊天模型 (chats.py)。"
            '資料為 JSON 清單，每筆為 {"prompt": 問句, "reply": 回覆}。'
            "手上沒有現成 JSON 的話，先用「步驟零」從一堆 .txt 對話檔自動產生 manifest。"
            "2026-07-28 貝氏化：Dropout 在聊天推論時會刻意保持開啟、重複解碼 20 次多數決"
            "（MC Dropout），回覆旁會附上信心度百分比——這不代表資料量需求變少，"
            "只是讓你看得出模型是「真的確定」還是「猜出好幾種不同答案」。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從文字檔匯入對話 manifest (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        chat_import_fields = _pairs_import_fields("imported_pairs.json")
        build_form(step0, chat_import_fields)
        import_btn = ttk.Button(
            step0, text="產生 manifest",
            command=lambda: self._start_import("聊天 manifest 匯入", "pairs", chat_import_fields),
        )
        import_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(import_btn)

        form = tk.Frame(outer, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=10)
        build_form(form, CHAT_FIELDS)
        btn = ttk.Button(outer, text="開始訓練",
                          command=lambda: self._start("聊天模型訓練", "chats.py", CHAT_FIELDS))
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _build_character_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "人物氣質模型",
            "從零打造的雙向 GRU 文字編碼器 (character_model.py)：把人物的文字描述轉成氣質特質評分 + "
            "persona embedding。manifest 是一個 JSON 清單，每筆為 "
            '{"name": 人物名稱, "description": 文字描述, "traits": {標籤: 0~1 強度, ...}}，'
            "標籤（tag）不是寫死的，是從 --train-manifest 資料裡自動蒐集出來的。"
            "訓練完的 checkpoint 要建立 character card，請用終端機另外執行 "
            "`python character_model.py --build --name ... --description ...`（此 GUI 只負責訓練）。"
            "注意：這裡只做文字→氣質，不含 voice clone——語音克隆需要真人參考音檔，是完全獨立、尚未開發的下一階段。",
        )
        form = tk.Frame(outer, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=10)
        build_form(form, CHARACTER_FIELDS)
        btn = ttk.Button(outer, text="開始訓練",
                          command=lambda: self._start("人物氣質模型訓練", "character_model.py", CHARACTER_FIELDS))
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _build_voice_clone_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "角色聲音克隆",
            "從零打造的單一講者 Tacotron 風格 TTS 模型 (voice_clone.py)：重用 chats.py 的 GRU 編碼器"
            "＋注意力機制，解碼器逐幀生成語音頻譜圖，再用古典的 Griffin-Lim 演算法還原成聲音"
            "（不需要額外的神經網路聲碼器）。manifest 是一個 JSON 清單，每筆為 "
            '{"audio": 該角色錄的 16-bit PCM WAV 檔名, "text": 這段錄音講的文字內容}——'
            "每個角色的聲音各自訓練一個 checkpoint，不是多人共用一個模型。"
            "訓練完成後，若要讓這個聲音接上既有的人物氣質 character card（voice_ref 欄位），"
            "或要試聽克隆結果，請用終端機另外執行 "
            "`python voice_clone.py --clone --text ... --out-wav ...` 或 "
            "`python voice_clone.py --attach-to-character <名稱> --out-dir ...`（此 GUI 只負責訓練）。"
            "注意：單一角色能錄到的語句通常很少，過擬合風險比其他模型更高，"
            "在真的錄到夠多語句之前不用太相信 val_loss 的絕對數字。手上已經有一批"
            "錄音檔＋逐字稿（不是用 prompt_voice.py 一筆一筆輸入）的話，"
            "先用「步驟零」批次匯入。",
        )

        step0 = tk.LabelFrame(outer, text=" 步驟零：從錄音＋逐字稿匯入 manifest (dataset_import.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step0.pack(fill="x", padx=10, pady=(0, 10))
        voice_import_fields = _sidecar_import_fields("audio", "voice_manifest.json")
        build_form(step0, voice_import_fields)
        import_btn = ttk.Button(
            step0, text="產生 manifest",
            command=lambda: self._start_import("聲音克隆 manifest 匯入", "sidecar-audio", voice_import_fields),
        )
        import_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(import_btn)

        form = tk.Frame(outer, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=10)
        build_form(form, VOICE_CLONE_FIELDS)
        btn = ttk.Button(outer, text="開始訓練",
                          command=lambda: self._start("角色聲音克隆訓練", "voice_clone.py", VOICE_CLONE_FIELDS))
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _build_circuit_tab(self, notebook: ttk.Notebook):
        outer = self._new_tab(
            notebook, "電路圖偵測",
            "本機 YOLO 遷移學習偵測器 (circuit_diagram_train.py)，供 視覺電路圖.py 即時推論用。"
            "手上如果只有 KiCad 的 .kicad_sch 電路圖檔案（還沒有標好框的 YOLO 資料集），"
            "先用「步驟一」自動轉換；已經有現成資料集的話可以跳過步驟一，直接用步驟二訓練。",
        )

        step1 = tk.LabelFrame(outer, text=" 步驟一：匯入 KiCad 電路圖 (kicad_dataset_convert.py，選用) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step1.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(step1, text="把 .kicad_sch 的 symbol 位置自動轉成 YOLO 標註框（不用手動框圖），"
                              "需要先安裝 KiCad（取得 kicad-cli）與 `pip install pymupdf`。",
                  bg=PANEL_BG, fg=MUTED_FG, anchor="w", wraplength=800, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=(10, 6), pady=(6, 4)
        )

        kicad_mode_frame = tk.Frame(step1, bg=PANEL_BG)
        kicad_mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=(10, 6), pady=(0, 6))
        self.kicad_mode = tk.StringVar(value="dir")
        ttk.Radiobutton(kicad_mode_frame, text="整個資料夾（批次轉換）", variable=self.kicad_mode,
                         value="dir", command=self._toggle_kicad_mode).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(kicad_mode_frame, text="單一檔案（先測試框有沒有對齊）", variable=self.kicad_mode,
                         value="file", command=self._toggle_kicad_mode).pack(side="left")

        self.kicad_dir_frame = tk.Frame(step1, bg=PANEL_BG)
        self.kicad_dir_fields = [
            Field("sch_dir", ".kicad_sch 資料夾", "dir", None, required=True),
        ]
        build_form(self.kicad_dir_frame, self.kicad_dir_fields)

        self.kicad_file_frame = tk.Frame(step1, bg=PANEL_BG)
        self.kicad_file_fields = [
            Field("sch_file", "單一 .kicad_sch 檔案（無 val split）", "file", None,
                  required=True, filetypes=KICAD_FT),
        ]
        build_form(self.kicad_file_frame, self.kicad_file_fields)

        self.kicad_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=(0, 0))
        self._toggle_kicad_mode()

        build_form(step1, KICAD_COMMON_FIELDS, start_row=3)
        kicad_btn = ttk.Button(step1, text="匯入並轉換", command=self._start_kicad_import)
        kicad_btn.grid(row=99, column=0, columnspan=3, pady=(4, 10))
        self.start_buttons.append(kicad_btn)

        step2 = tk.LabelFrame(outer, text=" 步驟二：訓練模型 (circuit_diagram_train.py) ",
                               bg=PANEL_BG, fg=FG, bd=1)
        step2.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        mode_frame = tk.Frame(step2, bg=PANEL_BG)
        mode_frame.pack(fill="x", padx=10, pady=(6, 6))
        self.circuit_mode = tk.StringVar(value="yaml")
        ttk.Radiobutton(mode_frame, text="使用現有 data.yaml", variable=self.circuit_mode,
                         value="yaml", command=self._toggle_circuit_mode).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(mode_frame, text="資料夾 + 類別列表", variable=self.circuit_mode,
                         value="dir", command=self._toggle_circuit_mode).pack(side="left")

        self.circuit_yaml_frame = tk.LabelFrame(step2, text=" data.yaml ", bg=PANEL_BG, fg=FG, bd=1)
        self.circuit_yaml_fields = [
            Field("data", "data.yaml 路徑", "file", None, required=True, filetypes=YAML_FT),
        ]
        build_form(self.circuit_yaml_frame, self.circuit_yaml_fields)

        self.circuit_dir_frame = tk.LabelFrame(step2, text=" 資料夾 + 類別（可直接接續上面步驟一的輸出資料夾） ",
                                                 bg=PANEL_BG, fg=FG, bd=1)
        self.circuit_dir_fields = [
            Field("dataset_dir", "資料夾（images/{train,val}, labels/{train,val}）", "dir", None,
                  required=True),
            Field("classes", "類別名稱（用空格或逗號分隔，順序 = label id）", "str", None,
                  required=True, hint="例：resistor capacitor inductor diode ic transistor battery"),
        ]
        build_form(self.circuit_dir_frame, self.circuit_dir_fields)

        self.circuit_yaml_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._toggle_circuit_mode()

        common_frame = tk.LabelFrame(step2, text=" 共用參數 ", bg=PANEL_BG, fg=FG, bd=1)
        common_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        build_form(common_frame, CIRCUIT_COMMON_FIELDS)

        btn = ttk.Button(step2, text="開始訓練", command=self._start_circuit)
        btn.pack(pady=10)
        self.start_buttons.append(btn)

    def _toggle_kicad_mode(self):
        if self.kicad_mode.get() == "dir":
            self.kicad_file_frame.grid_forget()
            self.kicad_dir_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=(0, 0))
        else:
            self.kicad_dir_frame.grid_forget()
            self.kicad_file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=(0, 0))

    def _start_kicad_import(self):
        common_values = collect_values(KICAD_COMMON_FIELDS)
        if common_values is None:
            return
        argv = build_argv(KICAD_COMMON_FIELDS, common_values)

        classes = [c for c in re.split(r"[,\s]+", common_values["classes"].strip()) if c]
        if not classes:
            messagebox.showerror("缺少欄位", "請填寫至少一個類別名稱")
            return
        argv += ["--classes", *classes]

        if self.kicad_mode.get() == "dir":
            values = collect_values(self.kicad_dir_fields)
            if values is None:
                return
            argv += ["--sch-dir", values["sch_dir"]]
        else:
            values = collect_values(self.kicad_file_fields)
            if values is None:
                return
            argv += ["--sch", values["sch_file"]]

        self._launch("KiCad 電路圖匯入", ["kicad_dataset_convert.py"] + argv)

    def _toggle_circuit_mode(self):
        if self.circuit_mode.get() == "yaml":
            self.circuit_dir_frame.pack_forget()
            self.circuit_yaml_frame.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.circuit_yaml_frame.pack_forget()
            self.circuit_dir_frame.pack(fill="x", padx=10, pady=(0, 10))

    def _start_circuit(self):
        common_values = collect_values(CIRCUIT_COMMON_FIELDS)
        if common_values is None:
            return
        argv = build_argv(CIRCUIT_COMMON_FIELDS, common_values)

        if self.circuit_mode.get() == "yaml":
            values = collect_values(self.circuit_yaml_fields)
            if values is None:
                return
            argv += ["--data", values["data"]]
        else:
            values = collect_values(self.circuit_dir_fields)
            if values is None:
                return
            classes = [c for c in re.split(r"[,\s]+", values["classes"].strip()) if c]
            if not classes:
                messagebox.showerror("缺少欄位", "請填寫至少一個類別名稱")
                return
            argv += ["--dataset-dir", values["dataset_dir"], "--classes", *classes]

        self._launch("電路圖偵測訓練", ["circuit_diagram_train.py"] + argv)

    # -- shared start/stop/log -----------------------------------------

    def _start(self, task_label: str, script: str, fields: list[Field]):
        values = collect_values(fields)
        if values is None:
            return
        self._launch(task_label, [script] + build_argv(fields, values))

    def _start_import(self, task_label: str, mode: str, fields: list[Field]):
        values = collect_values(fields)
        if values is None:
            return
        argv = ["--mode", mode] + build_argv(fields, values)
        self._launch(task_label, ["dataset_import.py"] + argv)

    def _launch(self, task_label: str, argv: list[str]):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("訓練中", "已經有一個訓練工作在執行，請先停止再開始新的。")
            return

        command = [PYTHON, "-u", *argv]
        self._append_log(f"\n$ {' '.join(command)}\n")
        self.status_var.set(f"訓練中: {task_label}")
        self._set_running_ui(True)

        self.proc = subprocess.Popen(
            command, cwd=str(TRANNING_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen):
        assert proc.stdout is not None  # always created with stdout=PIPE above
        for line in proc.stdout:
            self.log_queue.put(line)
        returncode = proc.wait()
        self.log_queue.put(("__DONE__", returncode))

    def _poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple):
                    self._on_finished(item[1])
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    def _on_finished(self, returncode: int):
        self.proc = None
        if self.stopping:
            self.status_var.set("已停止")
            self._append_log("\n[已手動停止 — 未保存 checkpoint]\n")
        elif returncode == 0:
            self.status_var.set("完成")
            self._append_log("\n[訓練完成]\n")
        else:
            self.status_var.set(f"發生錯誤（exit code {returncode}）")
            self._append_log(f"\n[程序以錯誤代碼 {returncode} 結束，往上捲看詳細錯誤訊息]\n")
        self.stopping = False
        self._set_running_ui(False)

    def stop_training(self):
        if self.proc is not None and self.proc.poll() is None:
            self.stopping = True
            self.proc.terminate()

    def _set_running_ui(self, running: bool):
        state = "disabled" if running else "normal"
        for btn in self.start_buttons:
            btn.configure(state=state)
        self.stop_btn.configure(state=("normal" if running else "disabled"))

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(side="top", fill="x", padx=10)

        self.status_var = tk.StringVar(value="閒置")
        tk.Label(bar, textvariable=self.status_var, bg=BG, fg=FG, anchor="w").pack(side="left")

        self.stop_btn = tk.Button(bar, text="停止訓練", bg=STOP_RED, fg="white", relief="flat",
                                    state="disabled", command=self.stop_training)
        self.stop_btn.pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="清除紀錄", command=self._clear_log).pack(side="right")

    def _build_log_console(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=(4, 10))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(frame, height=12, state="disabled", wrap="word",
                                  bg="#111213", fg="#d1d5db", relief="flat", bd=0,
                                  yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_text.yview)

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


def main():
    root = tk.Tk()
    TrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
