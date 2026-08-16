# 2026-08-13 修正完畢：lib/components/cli.py 匯入錯誤

## 對應錯誤（見 Agent/ErrorLog/2026-08-13.md）
Pylance `reportMissingImports`，`lib/components/cli.py` 第 32-35 行：
- `from prompt_toolkit import PromptSession`
- `from prompt_toolkit.completion import Completer, Completion`
- `from prompt_toolkit.formatted_text import HTML`
- `from prompt_toolkit.history import FileHistory`

## 根本原因
專案 `.venv` 沒有安裝 `prompt_toolkit`。這是既有已知缺口，`ErrorLog.md` 第 8 點
2026-08-05 那次就記錄過（當時範圍只修 `beautifulsoup4`，`prompt_toolkit` 故意留到
「之後真的要跑 cli.py/test_cli.py 時再處理」），今天就是那個「之後」。

## 修正內容
```bash
python -m pip install prompt_toolkit   # -> prompt_toolkit 3.0.53 + wcwidth 0.8.2
```

驗證過程中發現 `rich`（cli.py 第 36-40 行用到 `rich.console`/`rich.markup`/
`rich.panel`/`rich.syntax`/`rich.text`）在同一個 `.venv` 裡也沒裝——今天的 Pylance
錯誤清單沒列出來，研判是 Pylance 目前指向的直譯器跟這個 `.venv` 不同步
（見 CLAUDE.md「兩份 Python 環境並存」那條）。因為不裝 `rich` 的話 `cli.py` 實際上
還是匯入不了、無法驗證修正是否真的完整，所以一併裝上：
```bash
python -m pip install rich   # -> rich 15.0.0 + markdown-it-py 4.2.0 + mdurl 0.1.2
```

## 驗證
1. `python -c "from prompt_toolkit import PromptSession; ..."` 全部匯入成功。
2. `pytest lib/components/test_cli.py -q` → 15 個測試，14 個通過，
   `reportMissingImports` 相關的 collection 階段錯誤已消失。

## 留下的獨立問題（跟今天的匯入錯誤無關，未修）
`test_run_memory_delete_removes_entry` 這個測試失敗：`IndexError: list index out of range`
（`app/components/test_cli.py:124`，路徑字串是舊的 assertion-rewrite 快取殘留，
實體檔案已經在 `lib/components/test_cli.py`）。這是既有邏輯 bug，不屬於今天
`reportMissingImports` 的範圍，先記錄在這裡，之後要修再另外開一筆 ErrorLog。
