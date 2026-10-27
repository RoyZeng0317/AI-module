"""中文連結顯示與原始目標分離，避免破壞 query／簽名。"""
from io import StringIO
from urllib.parse import quote

import pytest
from rich.console import Console
from rich.markdown import Markdown
from url_display import readable_url, source_link
import tools


def test_chinese_url_display_and_rich_rendering_keep_raw_target():
    raw = "https://zh.wikipedia.org/wiki/" + quote("周柯宇")
    reply = tools._with_source("人物介紹", raw)
    assert "https://zh.wikipedia.org/wiki/周柯宇" in reply and f"(<{raw}>)" in reply
    stream = StringIO()
    Console(file=stream, width=160, color_system=None).print(Markdown(reply))
    assert "周柯宇" in stream.getvalue() and "%E5" not in stream.getvalue()


def test_reserved_sequences_signature_and_double_encoding_are_preserved():
    raw = "https://example.com/%E5%91%A8%2F%E6%9F%AF?q=%E5%AE%87%26a%3Db&sig=x%2Fy%2Bz%3D%25&x=%25E5%2591%25A8"
    display = readable_url(raw)
    assert display == "https://example.com/周%2F柯?q=宇%26a%3Db&sig=x%2Fy%2Bz%3D%25&x=%25E5%2591%25A8"
    assert f"(<{raw}>)" in source_link(raw)


@pytest.mark.parametrize("encoded", ["%FF", "%E5%91", "%ED%A0%80", "%E2%80%AE", "%0A", "%25E5"])
def test_invalid_utf8_and_controls_are_not_decoded(encoded):
    assert readable_url(encoded) == encoded


@pytest.mark.parametrize("url", [None, "javascript:alert(1)", "file:///a", "https://a.example/\nsecret", "https://a.example/<x>"])
def test_unsafe_source_is_not_rendered(url):
    assert source_link(url) == ""


def test_ascii_compatibility_and_markdown_escaping():
    assert tools._with_source("text", "https://example.com/path") == "text\nhttps://example.com/path"
    raw = "https://example.com/" + quote("周") + "(a)?x=%26"
    rendered = source_link(raw, "[周]* (a)")
    assert rendered.startswith(r"[\[周\]\* (a)]") and f"(<{raw}>)" in rendered
