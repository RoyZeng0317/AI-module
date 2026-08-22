"""Unit tests for code_snippet_import.py — 把 GitHub 專案的原始碼檔案
（Python/C/C++/C#）轉成 markdown、再轉成 chats.py 程式碼 checkpoint 吃的
{"prompt", "reply"} 訓練配對，不用手動整理 data/code_pairs.json。
"""

import json

from code_snippet_import import (
    build_prompt,
    collect_code_pairs,
    extract_c_family_units,
    extract_python_units,
    extract_source_tree,
    parse_markdown_units,
)


def test_extract_python_units_prefers_docstring_first_line():
    text = '''
def bubble_sort(arr):
    """氣泡排序法"""
    return sorted(arr)


def no_doc(x):
    return x + 1
'''
    units = extract_python_units(text)
    headings = dict(units)
    assert "氣泡排序法" in headings
    assert "def bubble_sort(arr):" in headings["氣泡排序法"]
    assert "no_doc" in headings


def test_extract_python_units_captures_class_methods():
    text = '''
class Stack:
    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
'''
    units = extract_python_units(text)
    headings = [h for h, _ in units]
    assert "Stack.push" in headings
    assert "Stack.pop" in headings


def test_extract_python_units_returns_empty_on_syntax_error():
    assert extract_python_units("def broken(:\n") == []


def test_extract_c_family_units_finds_function_and_body():
    text = """
// 費氏數列第 n 項
int fibonacci(int n) {
    if (n < 2) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"""
    units = extract_c_family_units(text)
    assert len(units) == 1
    heading, code = units[0]
    assert heading == "費氏數列第 n 項"
    assert code.startswith("int fibonacci(int n) {")
    assert code.rstrip().endswith("}")
    assert code.count("{") == code.count("}")


def test_extract_c_family_units_falls_back_to_name_without_comment():
    text = "int add(int a, int b) {\n    return a + b;\n}\n"
    units = extract_c_family_units(text)
    assert units == [("add", "int add(int a, int b) {\n    return a + b;\n}")]


def test_extract_c_family_units_ignores_braces_inside_strings_and_comments():
    text = (
        'int weird(void) {\n'
        '    // a stray brace in a comment }\n'
        '    const char *s = "also a stray brace }";\n'
        '    return 1;\n'
        '}\n'
    )
    units = extract_c_family_units(text)
    assert len(units) == 1
    assert units[0][1].rstrip().endswith("return 1;\n}")


def test_extract_c_family_units_does_not_misfire_on_else_if():
    text = "int classify(int x) {\n    if (x > 0) {\n        return 1;\n    } else if (x < 0) {\n        return -1;\n    }\n    return 0;\n}\n"
    units = extract_c_family_units(text)
    names = [h for h, _ in units]
    assert names == ["classify"]


def test_extract_source_tree_mirrors_directory_and_writes_markdown(tmp_path):
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "math_utils.py").write_text(
        'def add(a, b):\n    """兩數相加"""\n    return a + b\n', encoding="utf-8"
    )
    (source / "empty.h").write_text("#define FOO 1\n", encoding="utf-8")

    out = tmp_path / "md"
    summary = extract_source_tree(source, out)

    assert summary["scanned"] == 2
    assert len(summary["written"]) == 1
    assert "empty.h" in summary["empty"][0]

    md_path = out / "pkg" / "math_utils.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "## 兩數相加" in content
    assert "```python" in content
    assert "def add(a, b):" in content


def test_parse_markdown_units_extracts_heading_lang_code():
    text = "## bubble_sort\n\n```python\ndef bubble_sort(arr):\n    return sorted(arr)\n```\n"
    units = parse_markdown_units(text)
    assert units == [("bubble_sort", "python", "def bubble_sort(arr):\n    return sorted(arr)")]


def test_build_prompt_templates_non_chinese_heading_with_language():
    assert build_prompt("bubble_sort", "python") == "用 Python 寫一個 bubble_sort"
    assert build_prompt("fibonacci", "cpp") == "用 C++ 寫一個 fibonacci"
    assert build_prompt("Add", "csharp") == "用 C# 寫一個 Add"


def test_build_prompt_reuses_chinese_heading_without_double_templating():
    assert build_prompt("氣泡排序法", "python") == "用 Python 氣泡排序法"
    assert build_prompt("用 C++ 寫一個費氏數列", "cpp") == "用 C++ 寫一個費氏數列"


def test_collect_code_pairs_reads_all_markdown_recursively(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text(
        "## bubble_sort\n\n```python\ndef bubble_sort(arr):\n    return sorted(arr)\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "two.md").write_text(
        "## 費氏數列\n\n```cpp\nint fib(int n) { return n < 2 ? n : fib(n-1)+fib(n-2); }\n```\n",
        encoding="utf-8",
    )

    entries = collect_code_pairs(tmp_path)

    assert {"prompt": "用 Python 寫一個 bubble_sort", "reply": "def bubble_sort(arr):\n    return sorted(arr)"} in entries
    assert {"prompt": "用 C++ 費氏數列", "reply": "int fib(int n) { return n < 2 ? n : fib(n-1)+fib(n-2); }"} in entries
    assert len(entries) == 2


def test_extract_then_pairs_round_trip_matches_code_pairs_json_shape(tmp_path):
    """extract -> pairs 兩階段串起來，輸出格式要跟現有 data/code_pairs.json
    的 {"prompt", "reply"} 配對格式完全相容（可以直接追加進同一份檔案）。"""
    from dataset_import import write_manifest

    source = tmp_path / "src"
    source.mkdir()
    (source / "sort.py").write_text(
        'def bubble_sort(arr):\n    """氣泡排序法"""\n    return sorted(arr)\n', encoding="utf-8"
    )

    md_out = tmp_path / "md"
    extract_source_tree(source, md_out)
    entries = collect_code_pairs(md_out)

    out = tmp_path / "code_pairs.json"
    write_manifest(entries, out, val_ratio=0.0)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == [{
        "prompt": "用 Python 氣泡排序法",
        "reply": 'def bubble_sort(arr):\n    """氣泡排序法"""\n    return sorted(arr)',
    }]
