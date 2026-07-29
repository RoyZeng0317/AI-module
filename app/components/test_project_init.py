"""Unit tests for project_init.py — the deterministic file-tree scan behind
/init (CLAUDE.md 需求 #01).
"""

from project_init import END_MARK, START_MARK, build_summary, detect_stack, write_claude_md


def test_detect_stack_finds_marker_files(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")

    stack = detect_stack(tmp_path)
    assert "Node.js / JavaScript / TypeScript" in stack
    assert "Python" in stack


def test_detect_stack_reports_unknown_when_no_markers(tmp_path):
    assert detect_stack(tmp_path) == ["未偵測到已知的語言/框架標記檔案"]


def test_build_summary_lists_top_level_entries_and_ignores_noise_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".git").mkdir()

    summary = build_summary(tmp_path)

    assert "src/" in summary
    assert "README.md" in summary
    assert "node_modules" not in summary
    assert ".git" not in summary
    assert summary.startswith(START_MARK)
    assert summary.rstrip().endswith(END_MARK)


def test_write_claude_md_creates_new_file(tmp_path):
    path = write_claude_md(tmp_path)

    assert path == tmp_path / "CLAUDE.md"
    content = path.read_text(encoding="utf-8")
    assert START_MARK in content and END_MARK in content


def test_write_claude_md_preserves_user_content_outside_markers(tmp_path):
    existing = tmp_path / "CLAUDE.md"
    existing.write_text("# 我自己寫的規則\n\n這段不可以被 /init 動到\n", encoding="utf-8")

    write_claude_md(tmp_path)

    content = existing.read_text(encoding="utf-8")
    assert "我自己寫的規則" in content
    assert "這段不可以被 /init 動到" in content
    assert START_MARK in content and END_MARK in content


def test_write_claude_md_is_idempotent_and_only_replaces_marked_block(tmp_path):
    write_claude_md(tmp_path)
    first = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")

    (tmp_path / "extra_file.txt").write_text("new", encoding="utf-8")
    write_claude_md(tmp_path)
    second = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")

    assert first.count(START_MARK) == 1
    assert second.count(START_MARK) == 1  # 沒有重複疊加區塊
    assert "extra_file.txt" in second
