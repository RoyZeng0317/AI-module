import filecmp
import json
import shutil
import tkinter as tk
import wave
import winsound
from pathlib import Path
from tkinter import filedialog

# tranning/voice_clone.py 的 manifest 需要 {"audio": 檔名, "text": ...}，"audio" 是相對於
# --audio-root（預設 = manifest 自己的資料夾）解析的。把預設的 train/val manifest 直接放在
# 音檔資料夾「裡面」，voice_clone.py 才不用額外加 --audio-root 就能跑。
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_AUDIO_ROOT = DATA_DIR / "voice_audio"
DEFAULT_TRAIN_PATH = DEFAULT_AUDIO_ROOT / "voice_train.json"
DEFAULT_VAL_PATH = DEFAULT_AUDIO_ROOT / "voice_val.json"

window = tk.Tk()
window.title("voice manifest edit tool")
window.geometry("560x480")

selected_source: Path | None = None


def is_16bit_pcm_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getsampwidth() == 2
    except (wave.Error, OSError):
        return False


def unique_target(audio_root: Path, source: Path) -> Path:
    """避免不同錄音撞名互相覆蓋——同名但內容不同就加數字後綴；
    內容完全相同（例如重複匯入同一個檔案）就直接沿用既有檔案，不重複複製。
    """
    target = audio_root / source.name
    if not target.exists():
        return target
    if filecmp.cmp(source, target, shallow=False):
        return target
    stem, suffix = source.stem, source.suffix
    i = 1
    while True:
        candidate = audio_root / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def browse_audio_root():
    chosen = filedialog.askdirectory(title="選擇音檔資料夾", initialdir=audio_root_var.get())
    if chosen:
        audio_root_var.set(chosen)


def browse_path(var: tk.StringVar):
    chosen = filedialog.asksaveasfilename(
        title="選擇 manifest 檔案",
        initialdir=Path(var.get()).parent,
        initialfile=Path(var.get()).name,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
    )
    if chosen:
        var.set(chosen)


def pick_audio():
    global selected_source
    chosen = filedialog.askopenfilename(
        title="選擇音檔（.wav）",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
    )
    if not chosen:
        return
    path = Path(chosen)
    if not is_16bit_pcm_wav(path):
        status.config(text=f"「{path.name}」不是 16-bit PCM WAV，voice_clone.py 只支援這個格式，請先轉檔再匯入。")
        return

    selected_source = path
    audio_label.config(text=f"已選擇：{path.name}")
    status.config(text="已選擇音檔，播放確認內容後輸入文字，再按「新增」")


def play_audio():
    if selected_source is None:
        status.config(text="請先選擇音檔")
        return
    winsound.PlaySound(str(selected_source), winsound.SND_FILENAME | winsound.SND_ASYNC)


def stop_audio():
    winsound.PlaySound(None, winsound.SND_PURGE)


def append_record(record: dict, path: Path) -> int:
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def add_entry(event=None):
    global selected_source
    text = text_entry.get().strip()
    if selected_source is None:
        status.config(text="請先選擇音檔")
        return
    if not text:
        status.config(text="請輸入這段錄音講的文字內容")
        return

    audio_root = Path(audio_root_var.get())
    audio_root.mkdir(parents=True, exist_ok=True)
    target = unique_target(audio_root, selected_source)
    if not target.exists():
        shutil.copy2(selected_source, target)

    target_manifest = Path(train_path_var.get() if split_var.get() == "train" else val_path_var.get())
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    count = append_record({"audio": target.name, "text": text}, target_manifest)

    text_entry.delete(0, tk.END)
    audio_label.config(text="尚未選擇音檔")
    selected_source = None
    status.config(text=f"已寫入 {target_manifest.name}（音檔：{target.name}），目前共 {count} 筆")
    text_entry.focus_set()


# --- 音檔資料夾 ---
root_frame = tk.LabelFrame(window, text="音檔資料夾（--audio-root）", padx=10, pady=8)
root_frame.pack(fill="x", padx=12, pady=(12, 6))

audio_root_var = tk.StringVar(value=str(DEFAULT_AUDIO_ROOT))
root_row = tk.Frame(root_frame)
root_row.pack(fill="x")
tk.Entry(root_row, textvariable=audio_root_var).pack(side="left", fill="x", expand=True)
tk.Button(root_row, text="瀏覽", command=browse_audio_root).pack(side="left", padx=(6, 0))
tk.Label(root_frame, text="匯入的音檔會被複製一份到這裡（不會動到你原本的檔案）。",
          anchor="w", justify="left", fg="#666").pack(fill="x", pady=(4, 0))

# --- 資料集檔案：train / val manifest 路徑 ---
file_frame = tk.LabelFrame(window, text="資料集檔案", padx=10, pady=8)
file_frame.pack(fill="x", padx=12, pady=6)

train_path_var = tk.StringVar(value=str(DEFAULT_TRAIN_PATH))
val_path_var = tk.StringVar(value=str(DEFAULT_VAL_PATH))

train_row = tk.Frame(file_frame)
train_row.pack(fill="x", pady=2)
tk.Label(train_row, text="Train", width=6, anchor="w").pack(side="left")
tk.Entry(train_row, textvariable=train_path_var).pack(side="left", fill="x", expand=True)
tk.Button(train_row, text="瀏覽", command=lambda: browse_path(train_path_var)).pack(side="left", padx=(6, 0))

val_row = tk.Frame(file_frame)
val_row.pack(fill="x", pady=2)
tk.Label(val_row, text="Val", width=6, anchor="w").pack(side="left")
tk.Entry(val_row, textvariable=val_path_var).pack(side="left", fill="x", expand=True)
tk.Button(val_row, text="瀏覽", command=lambda: browse_path(val_path_var)).pack(side="left", padx=(6, 0))

split_var = tk.StringVar(value="train")
split_row = tk.Frame(file_frame)
split_row.pack(fill="x", pady=(6, 0))
tk.Label(split_row, text="寫入：").pack(side="left")
tk.Radiobutton(split_row, text="Train", variable=split_var, value="train").pack(side="left")
tk.Radiobutton(split_row, text="Val", variable=split_var, value="val").pack(side="left")

# --- 音檔區：選擇 + 試聽 ---
audio_frame = tk.LabelFrame(window, text="這一筆錄音", padx=10, pady=8)
audio_frame.pack(fill="x", padx=12, pady=6)

audio_btn_row = tk.Frame(audio_frame)
audio_btn_row.pack(fill="x")
tk.Button(audio_btn_row, text="選擇音檔…", command=pick_audio).pack(side="left")
tk.Button(audio_btn_row, text="▶ 播放", command=play_audio).pack(side="left", padx=(6, 0))
tk.Button(audio_btn_row, text="■ 停止", command=stop_audio).pack(side="left", padx=(6, 0))

audio_label = tk.Label(audio_frame, text="尚未選擇音檔", anchor="w")
audio_label.pack(fill="x", pady=(6, 0))

tk.Label(audio_frame, text="文字內容（這段錄音講的話）").pack(anchor="w", pady=(8, 0))
text_entry = tk.Entry(audio_frame)
text_entry.pack(fill="x", pady=(0, 4))
text_entry.bind("<Return>", add_entry)

# --- 動作區 ---
action_frame = tk.Frame(window)
action_frame.pack(fill="x", padx=12, pady=(6, 12))

tk.Button(action_frame, text="新增", command=add_entry).pack()

status = tk.Label(action_frame, text="", anchor="center", wraplength=520, justify="left")
status.pack(fill="x", pady=(6, 0))

window.mainloop()
