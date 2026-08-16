import tkinter as tk
from tkinter import filedialog
import json
from pathlib import Path

# character_model.py 的 manifest 格式要求氣質分數落在 [0, 1]（見 tranning/character_model.py
# 的 docstring），但滑桿介面用 1-10 正整數比較直覺，所以存檔時統一換算成 分數/10。
MIN_SCORE, MAX_SCORE = 1, 10
DEFAULT_TAGS = ["溫柔", "體貼", "寵溺", "俏皮", "深情", "保護慾"]

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_TRAIN_PATH = DATA_DIR / "character_traits_train.json"
DEFAULT_VAL_PATH = DATA_DIR / "character_traits_val.json"

window = tk.Tk()
window.title("character trait edit tool")
window.geometry("560x680")

trait_rows: dict[str, tuple[tk.Frame, tk.Scale]] = {}


def append_record(record: dict, path: Path):
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def add_trait_row(tag: str, initial: int = 5):
    if not tag or tag in trait_rows:
        return
    row = tk.Frame(traits_inner)
    row.pack(fill="x", pady=2)

    tk.Label(row, text=tag, width=10, anchor="w").pack(side="left")
    scale = tk.Scale(row, from_=MIN_SCORE, to=MAX_SCORE, orient="horizontal", resolution=1)
    scale.set(initial)
    scale.pack(side="left", fill="x", expand=True)
    tk.Button(row, text="移除", command=lambda: remove_trait_row(tag)).pack(side="left", padx=(6, 0))

    trait_rows[tag] = (row, scale)


def remove_trait_row(tag: str):
    row, _ = trait_rows.pop(tag, (None, None))
    if row is not None:
        row.destroy()


def add_new_tag():
    tag = new_tag_entry.get().strip()
    if not tag:
        status.config(text="請輸入標籤名稱")
        return
    if tag in trait_rows:
        status.config(text=f"「{tag}」已經存在")
        return
    add_trait_row(tag)
    new_tag_entry.delete(0, tk.END)
    status.config(text=f"已新增標籤「{tag}」")


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


def save_record():
    name = name_entry.get().strip()
    description = desc_text.get("1.0", tk.END).strip()
    if not name or not description:
        status.config(text="請輸入姓名與簡介描述")
        return
    if not trait_rows:
        status.config(text="請至少新增一個氣質標籤")
        return

    traits = {tag: round(scale.get() / MAX_SCORE, 2) for tag, (_, scale) in trait_rows.items()}
    record = {"name": name, "description": description, "traits": traits}

    target = Path(train_path_var.get() if split_var.get() == "train" else val_path_var.get())
    target.parent.mkdir(parents=True, exist_ok=True)
    count = append_record(record, target)

    name_entry.delete(0, tk.END)
    desc_text.delete("1.0", tk.END)
    status.config(text=f"已寫入 {target.name}，目前共 {count} 筆")
    name_entry.focus_set()


# --- 檔案區：train / val manifest 路徑 ---
file_frame = tk.LabelFrame(window, text="資料集檔案", padx=10, pady=8)
file_frame.pack(fill="x", padx=12, pady=(12, 6))

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

# --- 人物區：姓名 + 簡介描述 ---
person_frame = tk.LabelFrame(window, text="人物", padx=10, pady=8)
person_frame.pack(fill="x", padx=12, pady=6)

tk.Label(person_frame, text="姓名").pack(anchor="w")
name_entry = tk.Entry(person_frame)
name_entry.pack(fill="x", pady=(0, 6))

tk.Label(person_frame, text="簡介描述").pack(anchor="w")
desc_text = tk.Text(person_frame, height=4, wrap="word")
desc_text.pack(fill="x")

# --- 氣質標籤區：可拖動滑桿（1-10）+ 新增標籤 ---
traits_frame = tk.LabelFrame(window, text="氣質標籤（拖動滑桿設定 1-10）", padx=10, pady=8)
traits_frame.pack(fill="both", expand=True, padx=12, pady=6)

traits_canvas = tk.Canvas(traits_frame, highlightthickness=0)
traits_scrollbar = tk.Scrollbar(traits_frame, orient="vertical", command=traits_canvas.yview)
traits_inner = tk.Frame(traits_canvas)

traits_inner.bind("<Configure>", lambda e: traits_canvas.configure(scrollregion=traits_canvas.bbox("all")))
traits_canvas.create_window((0, 0), window=traits_inner, anchor="nw")
traits_canvas.configure(yscrollcommand=traits_scrollbar.set)

traits_canvas.pack(side="left", fill="both", expand=True)
traits_scrollbar.pack(side="right", fill="y")

for tag in DEFAULT_TAGS:
    add_trait_row(tag)

new_tag_row = tk.Frame(window)
new_tag_row.pack(fill="x", padx=12, pady=(0, 6))
new_tag_entry = tk.Entry(new_tag_row)
new_tag_entry.pack(side="left", fill="x", expand=True)
new_tag_entry.bind("<Return>", lambda e: add_new_tag())
tk.Button(new_tag_row, text="新增標籤", command=add_new_tag).pack(side="left", padx=(6, 0))

# --- 動作區 ---
action_frame = tk.Frame(window)
action_frame.pack(fill="x", padx=12, pady=(6, 12))

tk.Button(action_frame, text="儲存人物", command=save_record).pack()

status = tk.Label(action_frame, text="", anchor="center")
status.pack(fill="x", pady=(6, 0))

window.mainloop()
