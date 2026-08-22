"""Pyright 型別存根：`pyscript` 是 PyScript 執行期才提供的虛擬模組
（`document`/`window` 等實際上是 Pyodide 注入的瀏覽器全域物件），
本機/CI 裝不了這個套件。見 typings/js.pyi 說明、
Agent/ErrorFinished/ErrorLog.md 第 17 條。
"""
from typing import Any

def __getattr__(name: str) -> Any: ...
