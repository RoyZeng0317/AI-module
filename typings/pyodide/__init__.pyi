"""Pyright 型別存根：`pyodide` 是 Pyodide 執行期才提供的套件（`pyodide.ffi`
底下的 `create_once_callable()` 等橋接 JS callback 用的工具函式），本機/CI
沒有安裝也裝不了這個套件。見 typings/js.pyi 說明、
Agent/ErrorFinished/ErrorLog.md 第 17 條。
"""
from typing import Any

def __getattr__(name: str) -> Any: ...
