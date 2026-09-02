"""Unit tests for build_code_corpus.py — data/program/ 底下真實全端專案檔案
掃描成 transformer_chat.py pretrain() 用的純文字語料，含資料清理邏輯
（排除第三方依賴目錄、超大檔案、已壓縮檔案、重複內容）。
"""

from pathlib import Path

from build_code_corpus import build_corpus, collect_code_files, is_vendor_dir


def test_is_vendor_dir_matches_known_and_renamed_vendor_folders():
    assert is_vendor_dir("node_modules")
    assert is_vendor_dir("node_modules copy")  # 真的在 data/program 裡出現過的改名副本
    assert is_vendor_dir(".venv")
    assert is_vendor_dir("venv")
    assert is_vendor_dir(".git")
    assert is_vendor_dir("__pycache__")
    assert is_vendor_dir("dist-royhomenas")  # 帶後綴的 hashed build 輸出資料夾
    assert is_vendor_dir("build")
    assert is_vendor_dir("vendor")
    assert is_vendor_dir("some_pkg.egg-info")
    assert not is_vendor_dir("src")
    assert not is_vendor_dir("distinct-feature")  # "dist" 前綴不可以誤傷正常命名


def test_collect_code_files_filters_extension_and_skips_empty(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "style.css").write_text("body { color: red; }", encoding="utf-8")
    (tmp_path / "main.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "app.ts").write_text("const x = 1", encoding="utf-8")
    (tmp_path / "query.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.css").write_text("", encoding="utf-8")

    files, skipped = collect_code_files(tmp_path)

    names = {p.name for p in files}
    assert names == {"index.html", "style.css", "main.py", "app.ts", "query.sql"}
    assert skipped["empty"] == ["empty.css"]


def test_collect_code_files_prunes_vendor_directories_without_descending(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("console.log(1)", encoding="utf-8")
    nm = tmp_path / "node_modules" / "somepkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}", encoding="utf-8")
    venv = tmp_path / ".venv" / "Lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("# stdlib copy", encoding="utf-8")

    files, _ = collect_code_files(tmp_path)

    rels = {p.relative_to(tmp_path).as_posix() for p in files}
    assert rels == {"src/app.js"}


def test_collect_code_files_skips_oversized_and_minified(tmp_path):
    (tmp_path / "normal.js").write_text("const a = 1;", encoding="utf-8")
    (tmp_path / "huge.js").write_text("x" * 200, encoding="utf-8")
    (tmp_path / "lib.min.js").write_text("!function(){}()", encoding="utf-8")

    files, skipped = collect_code_files(tmp_path, max_file_size=100)

    names = {p.name for p in files}
    assert names == {"normal.js"}
    assert skipped["oversized"] == ["huge.js"]
    assert skipped["minified"] == ["lib.min.js"]


def test_collect_code_files_recurses_subfolders_and_is_sorted(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "z.html").write_text("<p>b</p>", encoding="utf-8")
    (tmp_path / "a" / "y.css").write_text("p { margin: 0; }", encoding="utf-8")

    files, _ = collect_code_files(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["a/y.css", "b/z.html"]


def test_build_corpus_writes_file_with_language_appropriate_header(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    (source / "index.html").write_text("<div>hi</div>", encoding="utf-8")
    (source / "style.css").write_text("body { color: blue; }", encoding="utf-8")
    (source / "main.py").write_text("print('hi')", encoding="utf-8")
    (source / "query.sql").write_text("SELECT * FROM t;", encoding="utf-8")
    out_path = tmp_path / "out" / "corpus.txt"

    summary = build_corpus(source, out_path)

    assert summary["files_written"] == 4
    text = out_path.read_text(encoding="utf-8")
    assert "<!-- index.html -->" in text and "<div>hi</div>" in text
    assert "/* style.css */" in text and "body { color: blue; }" in text
    assert "# main.py" in text and "print('hi')" in text
    assert "-- query.sql" in text and "SELECT * FROM t;" in text


def test_build_corpus_skips_whitespace_only_file(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    (source / "blank.css").write_text("   \n\n  ", encoding="utf-8")
    (source / "real.css").write_text("a { top: 0; }", encoding="utf-8")
    out_path = tmp_path / "corpus.txt"

    summary = build_corpus(source, out_path)

    assert summary["files_written"] == 1
    assert "blank.css" in summary["skipped"]["empty"]


def test_build_corpus_dedupes_identical_content_keeping_first_by_path(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    (source / "a_first.css").write_text("body { color: red; }", encoding="utf-8")
    (source / "b_copy.css").write_text("body { color: red; }", encoding="utf-8")
    out_path = tmp_path / "corpus.txt"

    summary = build_corpus(source, out_path)

    assert summary["files_written"] == 1
    assert summary["skipped"]["duplicate"] == ["b_copy.css"]
    text = out_path.read_text(encoding="utf-8")
    assert "a_first.css" in text
    assert "b_copy.css" not in text


def test_build_corpus_is_reproducible_across_runs(tmp_path):
    source = tmp_path / "program"
    source.mkdir()
    (source / "b.html").write_text("<b>2</b>", encoding="utf-8")
    (source / "a.html").write_text("<a>1</a>", encoding="utf-8")

    out1, out2 = tmp_path / "c1.txt", tmp_path / "c2.txt"
    build_corpus(source, out1)
    build_corpus(source, out2)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_build_corpus_on_real_data_program(tmp_path):
    """跑在真正的 data/program/ 底下，確保實際的全端專案（含 node_modules/
    .venv/超大檔案等雜訊）掃描不會噴例外、也不會把第三方依賴掃進語料裡。
    """
    real_source = Path(__file__).resolve().parent.parent / "data" / "program"
    if not real_source.exists():
        return
    out_path = tmp_path / "real_corpus.txt"

    summary = build_corpus(real_source, out_path)

    assert summary["files_written"] > 0
    assert summary["total_chars"] > 0
    text = out_path.read_text(encoding="utf-8")
    # 只檢查「檔案邊界標頭」裡沒有 node_modules 路徑（標頭是我們自己組的一行
    # 路徑字串），不檢查整份語料的任意子字串——某個使用者自己寫的檔案內容
    # 本身提到 "node_modules" 這個詞（例如處理路徑的程式碼字面字串）是合理
    # 的，不代表第三方依賴目錄真的被掃進來了。
    header_lines = [line for line in text.splitlines()
                     if line.startswith(("<!-- ", "/* ", "// ", "# ", "-- "))]
    assert header_lines  # 確保真的有掃到檔案，不是空的比對
    assert not any("node_modules" in line for line in header_lines)
