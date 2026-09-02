"""極簡 Markdown → Tk Text 渲染器。

不是完整的 CommonMark 實作，只涵蓋 sinco 回覆會用到的語法（標題、粗體、斜體、
行內程式碼、程式碼區塊、清單、引言、表格），目的跟 CLI 那邊用
rich.markdown.Markdown（見 lib/components/cli.py）一樣：讓聊天視窗顯示的是
排版後的「預覽」畫面，而不是原始的 "**粗體**"、"# 標題" 這些符號本身。
Tkinter 的 Text 元件本身沒有 markdown 渲染能力，所以這裡手動解析、逐段插入
不同 tag。標題支援到 h6（data/pairs.json 裡從 md-chat 匯入的真實回覆有
"####" 這種四級以上標題，只支援到 h3 會讓 "####" 原封不動露出來）；表格
沒有真正的儲存格排版能力，改成用等寬字型手動補齊欄寬對齊，跟 CLI 版
rich 表格用純文字框線畫表格是同一個折衷。

各 md_* tag 的字型/顏色由呼叫端（lib/main.py）用 chat_display.tag_configure()
設定，這裡只負責解析文字、決定該用哪個 tag 名稱——跟這個檔案裡其餘 tag
（ai_text、trace...）的設定方式一致，維持「main.py 管主題，其他模組管邏輯」
的分工。
"""

import re
import tkinter as tk
import unicodedata

_INLINE_RE = re.compile(
    r"(?P<bold>\*\*(?P<bold_a>.+?)\*\*|__(?P<bold_b>.+?)__)"
    r"|(?P<code>`(?P<code_txt>[^`]+)`)"
    r"|(?P<italic>(?<!\*)\*(?P<italic_a>[^*\n]+?)\*(?!\*)|(?<!_)_(?P<italic_b>[^_\n]+?)_(?!_))"
)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
_QUOTE_RE = re.compile(r"^>\s?(.*)")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)")
_NUMBERED_RE = re.compile(r"^\s*(\d+\.)\s+(.*)")
_FENCE_RE = re.compile(r"^\s*```")

_TABLE_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-+:?$")


def _split_table_row(line: str) -> list[str]:
    inner = _TABLE_ROW_RE.match(line).group(1)
    return [cell.strip() for cell in inner.split("|")]


def _is_table_separator(line: str) -> bool:
    if not _TABLE_ROW_RE.match(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_SEP_CELL_RE.match(c) for c in cells)


def _display_width(s: str) -> int:
    """粗略估算等寬字型下的視覺寬度：全形字元(中日韓)算 2，其餘算 1，讓中英文
    混排的表格欄位也能大致對齊——標準 len() 對中文字元會算成 1，導致中文
    儲存格看起來比英文儲存格窄一半。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _insert_table(widget: tk.Text, header: list[str], rows: list[list[str]],
                   base_tags: tuple[str, ...]) -> None:
    ncols = len(header)
    widths = [_display_width(header[i]) for i in range(ncols)]
    for row in rows:
        for i in range(ncols):
            widths[i] = max(widths[i], _display_width(row[i] if i < len(row) else ""))

    def _line(cells: list[str]) -> str:
        padded = []
        for i in range(ncols):
            cell = cells[i] if i < len(cells) else ""
            padded.append(cell + " " * max(0, widths[i] - _display_width(cell)))
        return " | ".join(padded)

    table_tags = base_tags + ("md_table",)
    widget.insert(tk.INSERT, _line(header) + "\n", table_tags)
    widget.insert(tk.INSERT, "-+-".join("-" * w for w in widths) + "\n", table_tags)
    for row in rows:
        widget.insert(tk.INSERT, _line(row) + "\n", table_tags)


def _insert_inline(widget: tk.Text, line: str, base_tags: tuple[str, ...]) -> None:
    pos = 0
    for m in _INLINE_RE.finditer(line):
        if m.start() > pos:
            widget.insert(tk.INSERT, line[pos:m.start()], base_tags)
        if m.group("bold") is not None:
            content = m.group("bold_a") if m.group("bold_a") is not None else m.group("bold_b")
            widget.insert(tk.INSERT, content, base_tags + ("md_bold",))
        elif m.group("code") is not None:
            widget.insert(tk.INSERT, m.group("code_txt"), base_tags + ("md_code_inline",))
        elif m.group("italic") is not None:
            content = m.group("italic_a") if m.group("italic_a") is not None else m.group("italic_b")
            widget.insert(tk.INSERT, content, base_tags + ("md_italic",))
        pos = m.end()
    if pos < len(line):
        widget.insert(tk.INSERT, line[pos:], base_tags)


def insert_markdown(widget: tk.Text, text: str, base_tag: str = "ai_text") -> None:
    """把 text 當 markdown 解析，在 widget 目前的 INSERT 位置逐段插入渲染後的
    內容（呼叫端自己負責 widget.configure(state="normal"/"disabled") 前後包住）。
    """
    base = (base_tag,)
    lines = text.split("\n")
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if _FENCE_RE.match(line):
            i += 1
            code_lines = []
            while i < n and not _FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳過收尾的 ```（如果原文沒收尾，i 這時已經等於 n，不會出錯）
            widget.insert(tk.INSERT, "\n".join(code_lines) + "\n", base + ("md_code_block",))
            continue

        if _TABLE_ROW_RE.match(line) and i + 1 < n and _is_table_separator(lines[i + 1]):
            header = _split_table_row(line)
            i += 2
            rows = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                rows.append(_split_table_row(lines[i]))
                i += 1
            _insert_table(widget, header, rows, base)
            continue

        header_match = _HEADER_RE.match(line)
        if header_match:
            level = len(header_match.group(1))
            _insert_inline(widget, header_match.group(2), base + (f"md_h{level}",))
            widget.insert(tk.INSERT, "\n", base)
            i += 1
            continue

        quote_match = _QUOTE_RE.match(line)
        if quote_match:
            _insert_inline(widget, quote_match.group(1), base + ("md_quote",))
            widget.insert(tk.INSERT, "\n", base)
            i += 1
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            widget.insert(tk.INSERT, "  •  ", base + ("md_bullet",))
            _insert_inline(widget, bullet_match.group(1), base)
            widget.insert(tk.INSERT, "\n", base)
            i += 1
            continue

        numbered_match = _NUMBERED_RE.match(line)
        if numbered_match:
            widget.insert(tk.INSERT, f"  {numbered_match.group(1)}  ", base + ("md_bullet",))
            _insert_inline(widget, numbered_match.group(2), base)
            widget.insert(tk.INSERT, "\n", base)
            i += 1
            continue

        if line.strip() == "":
            widget.insert(tk.INSERT, "\n", base)
            i += 1
            continue

        _insert_inline(widget, line, base)
        widget.insert(tk.INSERT, "\n", base)
        i += 1
