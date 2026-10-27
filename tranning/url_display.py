"""中文網址只解碼顯示標籤；點擊與下載仍使用原網址。"""
import re, unicodedata
from urllib.parse import urlsplit


def readable_url(url: str) -> str:
    def decode(match):
        parts = re.findall(r"%[0-9a-fA-F]{2}", match[0])
        output, i = [], 0
        while i < len(parts):
            first = int(parts[i][1:], 16)
            count = 2 if 0xC2 <= first <= 0xDF else 3 if 0xE0 <= first <= 0xEF else 4 if 0xF0 <= first <= 0xF4 else 0
            try:
                char = bytes(int(p[1:], 16) for p in parts[i:i + count]).decode("utf-8") if count else ""
            except UnicodeDecodeError:
                char = ""
            if len(char) == 1 and not unicodedata.category(char).startswith("C"):
                output.append(char)
                i += count
            else:
                output.append(parts[i])
                i += 1
        return "".join(output)
    return re.sub(r"(?:%[0-9a-fA-F]{2})+", decode, url)


def markdown_label(text: str) -> str:
    text = " ".join(str(text).split()).replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\\[\]*_`])", r"\\\1", text)


def source_link(url: str, label: str | None = None) -> str:
    """純 ASCII 來源沿用獨立網址行；中文則顯示可讀標籤。"""
    if not isinstance(url, str) or any(c.isspace() or c in "<>\\" for c in url):
        return ""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""
    except ValueError:
        return ""
    readable = readable_url(url)
    if label is None and readable == url:
        return url
    return f"[{markdown_label(label if label is not None else readable)}](<{url}>)"
