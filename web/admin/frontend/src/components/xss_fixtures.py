"""Export actual Python renderer outputs and dynamic sinks for Chromium XSS tests.
No app startup, model loading, network, or real conversation storage is used.
"""
import ast, html, json, re, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[5]
SOURCES = [ROOT / p for p in (
    "web/admin/frontend/src/components/action.py", "web/frontend/src/components/python.py",
    "web/admin/frontend/src/components/python.py")]
PAYLOADS = [
    '<img src=x onerror="window.__xss=1">', '<svg onload="window.__xss=1"></svg>',
    '<script>window.__xss=1</script>', '&lt;img src=x onerror=window.__xss=1&gt;',
    '[run](javascript:window.__xss=1)', '[run](data:text/html,<svg onload=window.__xss=1>)',
    '[run](vbscript:msgbox(1))', '[run](java\tscript:window.__xss=1)',
    '[run](https://example.com/`onmouseover=window.__xss=1`)',
    'https://example.com/`autofocus/onfocus=window.__xss=1`',
    '[**安全**](https://example.com/a**b**c?q=&quot;onmouseover=1)',
    '`<svg onload="window.__xss=1">` **粗體** 中文 \u2028 \u0000',
    '| title |\n| --- |\n| <img src=x onerror=window.__xss=1> |',
    '```html\n<img src=x onerror=window.__xss=1>\n```',
    'https://example.com/a*b*c__d__?q=" onmouseover="window.__xss=1',
]

class Element:
    def __init__(self, tag):
        self.tag, self.children, self.attrs = tag, [], {}
        self.innerHTML, self.textContent = "", None
        self.classList = MagicMock()
        self.scrollHeight = 0
    def setAttribute(self, name, value): self.attrs[name] = value
    def appendChild(self, child): self.children.append(child)
    def addEventListener(self, *args): pass
    def serialize(self):
        attrs = dict(self.attrs)
        if getattr(self, "className", None): attrs["class"] = self.className
        attributes = "".join(f' {k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items())
        content = html.escape(str(self.textContent)) if self.textContent is not None else self.innerHTML
        return f'<{self.tag}{attributes}>{content}{"".join(c.serialize() for c in self.children)}</{self.tag}>'


def load_renderer(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"el", "icon_button", "_escape", "_inline", "_split_table_row", "_is_table_separator",
             "_render_table", "render_markdown", "_draw_detections", "_build_message_element",
             "_render_messages", "_render_command_menu", "_render_history_list", "svg_span"}
    constants = {"_ESCAPE_MAP", "_TABLE_ROW_RE", "_TABLE_SEP_CELL_RE", "ICONS"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    ns = {"_re": re, "document": SimpleNamespace(createElement=Element), "create_proxy": lambda f:f,
          "_state": {}, "_close_open_menu":lambda:None, "_make_command_pick_handler": lambda i:lambda e:None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def fixtures():
    cases = []
    for path in SOURCES:
        ns = load_renderer(path)
        prefix = str(path.relative_to(ROOT))
        for i, payload in enumerate(PAYLOADS):
            cases.append({"name": f"{prefix}:markdown:{i}", "html": ns["render_markdown"](payload)})
            cases.append({"name": f"{prefix}:icon:{i}", "html": ns["icon_button"]("camera", payload).serialize()})
            result = Element("div")
            ns["_draw_detections"]({"width":100,"height":100,"detections":[{"label":payload,"box":[0,0,1,1],"confidence":1}]},
                MagicMock(), SimpleNamespace(width=100,height=100), result)
            cases.append({"name":f"{prefix}:detector:{i}", "html":result.serialize()})
            if "_render_messages" in ns:
                ns["_dom"] = {"chat_log":Element("div"),"chat_panel":Element("div")}
                ns["_render_messages"]([{"role":role,"content":payload} for role in ("user","assistant")])
                cases.append({"name":f"{prefix}:stored-replay:{i}","html":ns["_dom"]["chat_log"].serialize()})
            if "_render_command_menu" in ns:
                for kind in ("name","arg"):
                    ns["_dom"] = {"cmd_menu":Element("div")}
                    ns["_render_command_menu"](kind,[payload])
                    cases.append({"name":f"{prefix}:command-{kind}:{i}","html":ns["_dom"]["cmd_menu"].serialize()})
        if "_render_history_list" in ns:
            ns["_dom"]={"history_list":Element("div")}
            ns["_render_history_list"]([{"id":payload,"title":payload} for payload in PAYLOADS])
            cases.append({"name":f"{prefix}:history-titles","html":ns["_dom"]["history_list"].serialize()})
        if path.name == "action.py":
            cases.append({"name":"valid-link","html":ns["render_markdown"]('[**來源**](https://example.com/a_b?q=1&x=2) `**literal**`')})
            if len(sys.argv)>1:
                data=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
                ns["_dom"]={"chat_log":Element("div"),"chat_panel":Element("div")}
                ns["_render_messages"](data["messages"])
                cases.append({"name":"persisted-api-replay","html":ns["_dom"]["chat_log"].serialize()})
                ns["_dom"]={"history_list":Element("div")}
                ns["_render_history_list"]([data])
                cases.append({"name":"persisted-api-history-title","html":ns["_dom"]["history_list"].serialize()})
    return {"cases":cases,"payloads":PAYLOADS}

if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))

