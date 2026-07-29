"""Small file/text utilities shared across the desktop app.

No Tkinter dependency here on purpose — this module is the base layer that
command.py and conversation.py both import from.
"""

import re
from pathlib import Path

import markdown

# 支援所有檔案格式；markdown 檔案先轉成純文字再交給模型
MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def read_as_chat_content(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    if path.suffix.lower() in MARKDOWN_EXTENSIONS:
        html = markdown.markdown(raw_text)
        return re.sub(r"<[^>]+>", "", html).strip()
    return raw_text.strip()
