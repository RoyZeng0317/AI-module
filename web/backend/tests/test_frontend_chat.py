"""Browser-independent regression checks for PyScript chat state handling."""
import ast, asyncio
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "admin/frontend/src/components/action.py"


class Node:
    def __init__(self):
        self.disabled = False
        self.textContent = ""
        self.value = ""
        self.attrs = {}

    def setAttribute(self, key, value):
        self.attrs[key] = value

    def focus(self):
        pass


def load_functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names], type_ignores=[])
    namespace = {}
    exec(compile(selected, str(SOURCE), "exec"), namespace)
    return namespace


def test_send_prevents_duplicate_requests_and_resets_after_initialization_failure():
    async def scenario():
        ns = load_functions("send_chat")
        started, release = asyncio.Event(), asyncio.Event()
        calls, errors = [], []

        async def fail(text):
            calls.append(text)
            started.set()
            await release.wait()
            raise RuntimeError("offline")

        dom = {key: Node() for key in ("chat_send", "input_status", "chat_log", "chat_input")}
        ns.update(_state={}, _dom=dom, _send_chat_once=fail, _append_chat=lambda *args: errors.append(args))
        task = asyncio.create_task(ns["send_chat"]("hello"))
        await started.wait()
        await ns["send_chat"]("duplicate")
        assert calls == ["hello"] and dom["chat_send"].disabled
        release.set()
        await task
        assert not ns["_state"]["chat_busy"] and not dom["chat_send"].disabled
        assert dom["chat_input"].value == "hello"
        assert dom["chat_log"].attrs["aria-busy"] == "false"
        assert "offline" in errors[0][1]

    asyncio.run(scenario())


def test_markdown_links_escape_html_and_exclude_script_urls():
    import re
    ns = load_functions("_escape", "_inline")
    ns.update(_re=re, _ESCAPE_MAP={"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})
    assert 'rel="noopener noreferrer"' in ns["_inline"]("[來源](https://example.com)")
    assert "<a " not in ns["_inline"]("[來源](javascript:alert(1))")
    assert "<script>" not in ns["_inline"]("<script>alert(1)</script>")


def test_search_source_bare_url_is_clickable_without_nested_anchors():
    import re
    ns = load_functions("_escape", "_inline")
    ns.update(_re=re, _ESCAPE_MAP={"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})
    tools_source = SOURCE.parents[5] / "tranning/tools.py"
    tree = ast.parse(tools_source.read_text(encoding="utf-8"))
    helper = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_with_source"], type_ignores=[])
    exec(compile(helper, str(tools_source), "exec"), ns)
    text = ns["_with_source"]("搜尋摘要", "https://example.com/search?q=RAG&lang=zh")
    rendered = "\n".join(ns["_inline"](line) for line in text.splitlines())
    assert 'href="https://example.com/search?q=RAG&amp;lang=zh"' in rendered
    assert rendered.count("<a ") == 1 and 'rel="noopener noreferrer"' in rendered
    assert ns["_inline"]("[來源](https://example.com)").count("<a ") == 1
    assert "<a " not in ns["_inline"]("javascript:https://example.com")
    hostile = ns["_inline"]('https://example.com/?q=" onclick="alert(1)')
    assert 'href="https://example.com/?q=&quot;"' in hostile
    assert ' onclick="' not in hostile


def test_hidden_command_menu_has_author_style_override():
    import re
    css = SOURCE.with_name("style.css").read_text(encoding="utf-8")
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important\s*;", css)
