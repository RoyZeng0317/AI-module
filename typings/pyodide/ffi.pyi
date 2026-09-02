"""Pyright 型別存根：`pyodide.ffi`（`action.py` 用到的 `create_once_callable()`，
包住 `toBlob()` 這種非同步 JS callback 的 resolve，讓借用代理不會在同步呼叫
結束時就被銷毀）。見 typings/pyodide/__init__.pyi、typings/js.pyi 說明、
Agent/ErrorFinished/ErrorLog.md 第 17 條。
"""
from typing import Any

def __getattr__(name: str) -> Any: ...
