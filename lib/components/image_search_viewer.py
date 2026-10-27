"""本輪網路圖片附件的獨立預覽；不下載、不重跑模型。"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "output/image-search"


def _allowed_path(value):
    try:
        path = Path(value).resolve()
        return path if path.is_relative_to(CACHE_DIR.resolve()) and path.suffix == ".png" and path.is_file() else None
    except (OSError, ValueError):
        return None


def attachment_paths(reply: str) -> list[Path]:
    paths = []
    for value in re.findall(r"!\[(?:\\.|[^\]\\])*\]\(<([^<>\r\n]+)>\)", reply):
        path = _allowed_path(value)
        if path and path not in paths:
            paths.append(path)
    return paths[:3]


def open_previews(reply: str) -> int:
    paths = attachment_paths(reply)
    if not paths:
        return 0
    if getattr(sys, "frozen", False):
        raise OSError("目前封裝執行檔不支援獨立圖片預覽，請使用 Python 版 CLI 或圖片連結。")
    with (CACHE_DIR / "preview-errors.log").open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *map(str, paths)], cwd=ROOT,
                         stdout=log, stderr=log,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return len(paths)


def show_images(paths):
    import tkinter as tk, webbrowser
    from tkinter import ttk
    from PIL import Image, ImageTk
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tranning.url_display import readable_url
    valid = [path for value in paths if (path := _allowed_path(value))]
    if not valid:
        return
    root = tk.Tk()
    root.title("網路圖片搜尋 / Image search")
    root.geometry("1000x850")
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)
    photos = []
    for index, path in enumerate(valid[:3], 1):
        metadata_path = path.with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.stat().st_size < 64 * 1024 else {}
        except (OSError, ValueError):
            metadata = {}
        frame = ttk.Frame(tabs, padding=12)
        query = str(metadata.get("query", "圖片"))
        tabs.add(frame, text=f"{query} {index}")
        score = metadata.get("score")
        english = metadata.get("lang") == "en"
        status = (f"YOLO prediction: {query} {score:.1%}. Not proof of identity." if english else f"YOLO 預測：{query} {score:.1%}；不代表身份驗證。") if isinstance(score, (int, float)) else ("Search image: not checked by the person model." if english else "網路搜尋候選：未經人物模型確認。")
        ttk.Label(frame, text=status, wraplength=900).pack(pady=8)
        with Image.open(path) as image:
            image.thumbnail((900, 600))
            photos.append(ImageTk.PhotoImage(image))
        ttk.Label(frame, image=photos[-1]).pack(expand=True)
        source = str(metadata.get("source", ""))
        ttk.Label(frame, text=readable_url(source), wraplength=900).pack(pady=8)
        if source.startswith(("https://", "http://")):
            ttk.Button(frame, text="Open source" if english else "開啟來源頁面",
                       command=lambda url=source: webbrowser.open(url)).pack()
    root.mainloop()


if __name__ == "__main__":
    show_images(sys.argv[1:])
