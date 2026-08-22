"""Pyright 型別存根：`js` 是 Pyodide 在瀏覽器執行期才注入的虛擬模組，
本機/CI 沒有安裝也裝不了這個套件，靜態分析永遠看不到，只能用存根檔
告訴 pyright「這些名字合法存在、型別當作 Any」，藉此消除假警報。
不影響瀏覽器實際執行（瀏覽器讀的是 Pyodide 真的注入的 js 物件，不是這支檔案）。
見 Agent/ErrorFinished/ErrorLog.md 第 17 條。
"""
from typing import Any

def __getattr__(name: str) -> Any: ...
