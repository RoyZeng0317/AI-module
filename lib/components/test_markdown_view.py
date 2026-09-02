"""markdown_view.insert_markdown() 的渲染測試——重點驗證 CLAUDE.md 需求
「模型輸出要有 markdown preview，不能讓 '##'、表格管線符號這些原始語法
原封不動露出來」：data/pairs.json 裡從 md-chat 匯入的真實回覆含有四級以上
標題（"####"）跟表格，這支測試確保這兩種語法都真的被解析掉，不是只驗證
h1~h3、清單這些原本就有的語法。
"""

import tkinter as tk

import pytest

from lib.components.markdown_view import insert_markdown


@pytest.fixture
def widget():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    text = tk.Text(root)
    for tag in ("ai_text", "md_h1", "md_h2", "md_h3", "md_h4", "md_h5", "md_h6",
                "md_bold", "md_italic", "md_code_inline", "md_code_block",
                "md_quote", "md_bullet", "md_table"):
        text.tag_configure(tag)
    yield text
    root.destroy()


def _rendered(widget) -> str:
    return widget.get("1.0", tk.END)


def test_h1_to_h3_headers_stripped(widget):
    insert_markdown(widget, "# 一級\n## 二級\n### 三級")
    content = _rendered(widget)
    assert "#" not in content
    assert "一級" in content and "二級" in content and "三級" in content


def test_h4_to_h6_headers_stripped(widget):
    """既有 regex 只到 h3，data/pairs.json 的真實回覆裡有 "####" 這種
    四級標題——沒修就會原封不動露出 "####" 字面符號。"""
    insert_markdown(widget, "#### 四級\n##### 五級\n###### 六級")
    content = _rendered(widget)
    assert "#" not in content
    assert "四級" in content and "五級" in content and "六級" in content


def test_h4_header_uses_its_own_tag(widget):
    insert_markdown(widget, "#### 標題文字")
    ranges = widget.tag_ranges("md_h4")
    assert ranges
    assert widget.get(ranges[0], ranges[1]) == "標題文字"


def test_table_pipes_and_separator_row_not_left_raw(widget):
    """data/pairs.json 裡有 81 行表格列；沒有表格解析時 "| A | B |" 跟
    "| --- | --- |" 這種原始 GFM 表格語法會整行照原樣顯示。"""
    table_md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    insert_markdown(widget, table_md)
    content = _rendered(widget)
    # 分隔列本身的 "---" 不該原封不動出現（改成依欄寬計算的分隔線）
    assert "| --- | --- |" not in content
    assert "A" in content and "B" in content and "1" in content and "2" in content


def test_table_cells_land_in_md_table_tag(widget):
    table_md = "| Name | Score |\n| --- | --- |\n| Roy | 100 |"
    insert_markdown(widget, table_md)
    ranges = widget.tag_ranges("md_table")
    assert ranges
    table_text = widget.get(ranges[0], ranges[-1])
    assert "Name" in table_text and "Roy" in table_text and "100" in table_text


def test_plain_text_without_markdown_still_renders_unchanged(widget):
    insert_markdown(widget, "普通句子，沒有語法。")
    assert _rendered(widget).strip() == "普通句子，沒有語法。"
