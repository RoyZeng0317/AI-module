# 開發用小工具：一鍵執行型別檢查，把全部錯誤整理成文字，按鈕即可複製，不用去 VSCode 面板一條條複製。
#
# Pyodide/PyScript 的假警報（js/document/window unresolved import）已經靠專案根目錄的
# typings/js.pyi、typings/pyscript.pyi + pyrightconfig.json 在 pyright 端消掉了，
# 不是這支檔案的責任，見 Agent/ErrorFinished/ErrorLog.md 第 17 條。
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

FONT = "Microsoft JhengHei"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 顯示名稱 -> 相對於專案根目錄的檢查路徑；「全專案」用 "." 讓 pyrightconfig.json 的
# exclude 清單自己過濾掉 .venv/build/checkpoint 等不該掃的資料夾。
TARGET_DIRS = {
    "lib": "lib",
    "tranning": "tranning",
    "web/backend": "web/backend",
    "web/frontend": "web/frontend",
    "shape-vision": "shape-vision",
    "全專案": ".",
}


class DiagGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-module - 錯誤一鍵輸出")
        self.root.geometry("860x580")
        self._build()

    def _build(self):
        style = ttk.Style(self.root)
        style.configure("TButton", font=(FONT, 10))
        style.configure("TLabel", font=(FONT, 10))
        style.configure("TCombobox", font=(FONT, 10))

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")

        ttk.Label(top, text="範圍：").pack(side="left")
        self.target_var = tk.StringVar(value="lib")
        target_box = ttk.Combobox(
            top, textvariable=self.target_var, values=list(TARGET_DIRS.keys()),
            state="readonly", width=13, font=(FONT, 10),
        )
        target_box.pack(side="left", padx=(0, 10))

        self.run_btn = ttk.Button(top, text="▶ 一鍵檢查並輸出錯誤", command=self._run_check)
        self.run_btn.pack(side="left")
        self.copy_btn = ttk.Button(top, text="📋 複製全部", command=self._copy_all, state="disabled")
        self.copy_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(top, text="💾 另存為 .txt", command=self._save_txt, state="disabled")
        self.save_btn.pack(side="left")
        self.status_var = tk.StringVar(value="就緒")
        ttk.Label(top, textvariable=self.status_var, foreground="#666666").pack(side="right")

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="none", font=("Consolas", 10), state="disabled")
        sb_y = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        sb_x = ttk.Scrollbar(body, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

    def _set_text(self, content):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def _run_check(self):
        self.run_btn.configure(state="disabled")
        self.copy_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.status_var.set("檢查中…")
        self._set_text("")
        target = TARGET_DIRS[self.target_var.get()]
        threading.Thread(target=self._worker, args=(target,), daemon=True).start()

    def _worker(self, target):
        output, note = self._collect_diagnostics(target)
        self.root.after(0, self._on_done, output, note)

    def _pyright_available(self):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pyright", "--version"],
                capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _collect_diagnostics(self, target):
        if self._pyright_available():
            return self._run_pyright(target)
        return self._run_syntax_fallback(target)

    def _run_pyright(self, target):
        cmd = [sys.executable, "-m", "pyright", target]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, cwd=PROJECT_ROOT,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return out.strip(), "pyright"
        except subprocess.TimeoutExpired:
            return "檢查逾時（超過 180 秒）。", "pyright"

    def _run_syntax_fallback(self, target):
        # pyright 沒裝時的備用方案：純用內建 ast 模組逐檔案掃語法錯誤，
        # 兩份環境（.venv／cuda）都不用額外裝套件就能跑，抓不到型別錯誤，
        # 只能抓語法錯誤（例如漏冒號、縮排錯誤）。
        import ast

        base = PROJECT_ROOT / target
        files = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        problems = []
        checked = 0
        for path in files:
            if any(part in {".venv", "__pycache__", "chat_runs", "code_runs"} for part in path.parts):
                continue
            checked += 1
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                rel = path.relative_to(PROJECT_ROOT)
                problems.append(f"{rel}:{exc.lineno}:{exc.offset} - {exc.msg}")
            except UnicodeDecodeError as exc:
                rel = path.relative_to(PROJECT_ROOT)
                problems.append(f"{rel} - 無法以 UTF-8 讀取：{exc}")

        header = (
            "⚠ 找不到 pyright，改用內建 ast 模組做語法檢查（只能抓語法錯誤，抓不到型別錯誤）。\n"
            f"   要用完整型別檢查請安裝：{sys.executable} -m pip install pyright\n\n"
        )
        if not problems:
            body = f"掃過 {checked} 個檔案，沒有語法錯誤。"
        else:
            body = f"掃過 {checked} 個檔案，發現 {len(problems)} 個語法錯誤：\n\n" + "\n".join(problems)
        return header + body, "fallback"

    def _on_done(self, output, note):
        self.run_btn.configure(state="normal")
        self._set_text(output or "（沒有輸出）")
        self.copy_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        if note == "fallback":
            self.status_var.set("完成（語法檢查模式，未安裝 pyright）")
        else:
            self.status_var.set("完成，可按「複製全部」貼給我")

    def _copy_all(self):
        content = self.text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("已複製到剪貼簿")

    def _save_txt(self):
        content = self.text.get("1.0", "end-1c")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt")],
            initialfile="errors.txt",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status_var.set(f"已儲存：{path}")
        except OSError as exc:
            messagebox.showerror("儲存失敗", str(exc))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DiagGUI().run()
