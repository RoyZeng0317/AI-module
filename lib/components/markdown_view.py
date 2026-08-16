"""極簡 Markdown → Tk Text 渲染器。

不是完整的 CommonMark 實作，只涵蓋 sinco 回覆會用到的語法（標題、粗體、斜體、
行內程式碼、程式碼區塊、清單、引言），目的跟 CLI 那邊用 rich.markdown.Markdown
（見 lib/components/cli.py）一樣：讓聊天視窗顯示的是排版後的「預覽」畫面，
而不是原始的 "**粗體**"、"# 標題" 這些符號本身。Tkinter 的 Text 元件本身沒有
markdown 渲染能力，所以這裡手動解析、逐段插入不同 tag。

各 md_* tag 的字型/顏色由呼叫端（lib/main.py）用 chat_display.tag_configure()
設定，這裡只負責解析文字、決定該用哪個 tag 名稱——跟這個檔案裡其餘 tag
（ai_text、trace...）的設定方式一致，維持「main.py 管主題，其他模組管邏輯」
的分工。
"""

import re
import tkinter as tk

_INLINE_RE = re.compile(
    r"(?P<bold>\*\*(?P<bold_a>.+?)\*\*|__(?P<bold_b>.+?)__)"
    r"|(?P<code>`(?P<code_txt>[^`]+)`)"
    r"|(?P<italic>(?<!\*)\*(?P<italic_a>[^*\n]+?)\*(?!\*)|(?<!_)_(?P<italic_b>[^_\n]+?)_(?!_))"
)

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)")
_QUOTE_RE = re.compile(r"^>\s?(.*)")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)")
_NUMBERED_RE = re.compile(r"^\s*(\d+\.)\s+(.*)")
_FENCE_RE = re.compile(r"^\s*```")


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
