"""action.py 的 render_markdown()（PyScript/Pyodide 版本，瀏覽器聊天視窗用）
渲染測試——重點驗證 CLAUDE.md 需求「模型輸出要有 markdown preview，不能讓
'##'、表格管線符號這些原始語法原封不動露出來」，比照 lib/components/
test_markdown_view.py 對桌面 GUI 版的同一份測試。

action.py 本身是給 Pyodide(瀏覽器 WASM) 執行的，import 階段會直接呼叫
`pyscript.document`/`window` 跟 `asyncio.ensure_future(check_health())`
這些瀏覽器專屬 API，在一般 CPython/pytest 環境下沒有真正的事件迴圈、也沒有
DOM，所以這裡用最小可用的假物件（MagicMock 自動接住任意屬性存取/呼叫）
把 pyscript/js/pyodide 這三個模組 stub 掉，讓檔案能被正常 import 到，藉此
測試裡面純邏輯的 render_markdown()，不用真的起一個瀏覽器。
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ACTION_PATH = Path(__file__).parent / "action.py"


@pytest.fixture(scope="module")
def action():
    """把 action.py 在假的 pyscript/js/pyodide 環境下 import 一次，回傳
    module 物件供各測試呼叫 module.render_markdown()。"""
    saved_modules = {name: sys.modules.get(name) for name in
                      ("pyscript", "js", "pyodide", "pyodide.ffi")}

    pyscript_mod = types.ModuleType("pyscript")
    pyscript_mod.document = MagicMock()
    pyscript_mod.document.getElementById.return_value = MagicMock()
    pyscript_mod.window = MagicMock()
    pyscript_mod.window.location = types.SimpleNamespace(
        protocol="http:", host="localhost:8000", pathname="/",
    )
    sys.modules["pyscript"] = pyscript_mod

    js_mod = types.ModuleType("js")
    js_mod.console = MagicMock()
    js_mod.WebSocket = MagicMock()
    js_mod.URL = MagicMock()
    js_mod.Promise = MagicMock()
    sys.modules["js"] = js_mod

    pyodide_mod = types.ModuleType("pyodide")
    pyodide_ffi_mod = types.ModuleType("pyodide.ffi")
    pyodide_ffi_mod.create_once_callable = lambda fn: fn
    sys.modules["pyodide"] = pyodide_mod
    sys.modules["pyodide.ffi"] = pyodide_ffi_mod

    import asyncio
    orig_ensure_future = asyncio.ensure_future
    asyncio.ensure_future = lambda coro, **kw: coro  # 沒有事件迴圈可跑，只要不炸即可

    try:
        spec = importlib.util.spec_from_file_location("web_action", _ACTION_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        asyncio.ensure_future = orig_ensure_future
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_h1_to_h3_headers_stripped(action):
    html = action.render_markdown("# 一級\n## 二級\n### 三級")
    assert "#" not in html
    assert "<h3" in html and "<h4" in html and "<h5" in html


def test_h4_to_h6_headers_capped_at_h6(action):
    """既有規則只到 h3；data/pairs.json 從 md-chat 匯入的真實回覆有 "####"
    這種四級標題，沒修就會原封不動當成一般段落文字露出 "####" 字面符號。"""
    html = action.render_markdown("#### 四級\n##### 五級\n###### 六級")
    assert "#" not in html
    assert html.count("<h6") == 3
    assert "四級" in html and "五級" in html and "六級" in html


def test_table_rendered_as_real_html_table(action):
    """data/pairs.json 裡有 81 行表格列；沒有表格解析時 "| A | B |" 跟
    "| --- | --- |" 這種原始 GFM 表格語法會整行照原樣顯示成一般段落。"""
    table_md = "| Name | Score |\n| --- | --- |\n| Roy | 100 |"
    html = action.render_markdown(table_md)
    assert "|" not in html
    assert "---" not in html
    assert "<table" in html
    assert "<th>Name</th>" in html and "<th>Score</th>" in html
    assert "<td>Roy</td>" in html and "<td>100</td>" in html


def test_plain_paragraph_unaffected(action):
    html = action.render_markdown("普通句子，沒有語法。")
    assert html == '<p class="md-p">普通句子，沒有語法。</p>'
