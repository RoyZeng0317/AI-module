import tkinter as tk
from tkinter import filedialog
import json
from pathlib import Path

# 匯出成為 json 檔案
DEFAULT_PAIRS_PATH = Path(__file__).resolve().parents[3] / "data" / "pairs.json"
current_path = DEFAULT_PAIRS_PATH

window = tk.Tk()
window.title("propmt edit tool")
window.geometry("560x440")


def load_json():
    global current_path
    chosen = filedialog.askopenfilename(
        title="載入 JSON 檔案",
        initialdir=current_path.parent,
        filetypes=[("JSON files", "*.json")],
    )
    if not chosen:
        return

    path = Path(chosen)
    try:
        pairs = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        status.config(text=f"載入失敗：{exc}")
        return

    current_path = path
    file_label.config(text=f"目前檔案：{current_path}")
    status.config(text=f"已載入，目前共 {len(pairs)} 筆")


def new_json():
    global current_path
    chosen = filedialog.asksaveasfilename(
        title="新建 JSON 檔案",
        initialdir=current_path.parent,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
    )
    if not chosen:
        return

    path = Path(chosen)
    path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")

    current_path = path
    file_label.config(text=f"目前檔案：{current_path}")
    status.config(text="已新建空白檔案，目前共 0 筆")


def add_pair(event=None):
    prompt = promptbox.get().strip()
    reply = replybox.get().strip()
    if not prompt or not reply:
        status.config(text="請輸入 prompt 與 reply 後再新增")
        return

    pairs = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else []
    pairs.append({"prompt": prompt, "reply": reply})
    current_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    promptbox.delete(0, tk.END)
    replybox.delete(0, tk.END)
    status.config(text=f"已新增，目前共 {len(pairs)} 筆")
    promptbox.focus_set()


SYMBOLS = ["，", "。", "！", "？", "、", "~", "…", "「", "」", "♥"]
active_entry = None


def set_active_entry(entry):
    global active_entry
    active_entry = entry


def insert_symbol(symbol):
    target = active_entry or promptbox
    target.insert(tk.INSERT, symbol)
    target.focus_set()


# --- 檔案區 ---
file_frame = tk.LabelFrame(window, text="檔案", padx=10, pady=8)
file_frame.pack(fill="x", padx=12, pady=(12, 6))

file_label = tk.Label(file_frame, text=f"目前檔案：{current_path}", anchor="w", justify="left", wraplength=500)
file_label.pack(fill="x")

filebar = tk.Frame(file_frame)
filebar.pack(fill="x", pady=(6, 0))

loadbtn = tk.Button(filebar, text="載入 JSON", command=load_json)
loadbtn.pack(side="left")

newbtn = tk.Button(filebar, text="新建 JSON 檔案", command=new_json)
newbtn.pack(side="left", padx=(8, 0))

# --- 輸入區 ---
# 修正為一般輸入框的格式
# 需要 label 標籤
form_frame = tk.LabelFrame(window, text="新增對話", padx=10, pady=8)
form_frame.pack(fill="x", padx=12, pady=6)
form_frame.columnconfigure(1, weight=1)

tk.Label(form_frame, text="Prompt").grid(row=0, column=0, sticky="w", pady=4)
promptbox = tk.Entry(form_frame)
promptbox.grid(row=0, column=1, sticky="ew", pady=4)
promptbox.bind("<FocusIn>", lambda e: set_active_entry(promptbox))
promptbox.bind("<Return>", add_pair)

tk.Label(form_frame, text="Reply").grid(row=1, column=0, sticky="w", pady=4)
replybox = tk.Entry(form_frame)
replybox.grid(row=1, column=1, sticky="ew", pady=4)
replybox.bind("<FocusIn>", lambda e: set_active_entry(replybox))
replybox.bind("<Return>", add_pair)

active_entry = promptbox

# --- 符號區 ---
symbol_frame = tk.LabelFrame(window, text="符號", padx=10, pady=8)
symbol_frame.pack(fill="x", padx=12, pady=6)

for symbol in SYMBOLS:
    tk.Button(symbol_frame, text=symbol, width=3,
              command=lambda s=symbol: insert_symbol(s)).pack(side="left", padx=2)

# --- 動作區 ---
action_frame = tk.Frame(window)
action_frame.pack(fill="x", padx=12, pady=(6, 12))

addbtn = tk.Button(action_frame, text="新增", command=add_pair)
addbtn.pack()

status = tk.Label(action_frame, text="", anchor="center")
status.pack(fill="x", pady=(6, 0))

window.mainloop()
