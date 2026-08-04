"""Unit tests for cli.py's SlashCommandCompleter — the Tab-completion logic
behind "跟 GUI 一樣可以 tab 快速鍵入完整指令". This only tests the pure
get_completions() generator (no real terminal/TTY needed); prompt_toolkit's
own Tab-key wiring and the actual live-typing experience still need your own
manual test in a real terminal, same honesty convention as this project's
other GUI-interaction features (see CLAUDE.md to-do #09/#12).

Also covers run_memory() — /memory ported from the GUI's CommandPalette
(command.py) so the CLI can edit persistent memory too, not just chat.
Every test monkeypatches memory_store.MEMORY_PATH to an isolated tmp_path
file (same pattern as app/components/test_command.py) so these never touch
the real project's memory/memory.json.
"""

from types import SimpleNamespace

import cli
import memory_store


def _completions(text: str) -> list[str]:
    document = SimpleNamespace(text_before_cursor=text)
    return [item.text for item in cli.SlashCommandCompleter().get_completions(document, None)]


def test_no_completions_without_leading_slash():
    assert _completions("hello") == []
    assert _completions("") == []


def test_completes_partial_builtin_command_name():
    assert _completions("/he") == ["help"]
    # 比對跟 get_completions() 一樣的大小寫不敏感規則（"/c" 也會吃到
    # "CodeReview" 這種指令），不是只比對字面開頭一致的子集合。
    assert _completions("/c") == [c for c in cli._slash_commands() if c.lower().startswith("c")]


def test_completes_markdown_command_names(monkeypatch):
    monkeypatch.setattr(cli, "_markdown_commands", lambda: ["rules", "release"])
    assert _completions("/rel") == ["release"]


def test_full_command_name_still_matches_itself():
    assert "help" in _completions("/help")


def test_unknown_command_prefix_yields_no_completions():
    assert _completions("/doesnotexist") == []


def test_character_argument_suggestions_use_character_names(monkeypatch):
    monkeypatch.setattr(cli, "_character_names", lambda: ["周柯宇", "周興哲"])
    assert _completions("/character ") == ["周柯宇", "周興哲"]
    assert _completions("/character 周柯") == ["周柯宇"]


def test_no_argument_suggestions_for_commands_without_registered_args(monkeypatch):
    monkeypatch.setattr(cli, "_markdown_commands", lambda: ["rules"])
    assert _completions("/rules ") == []


def test_completion_replaces_only_the_partially_typed_text():
    completions = list(cli.SlashCommandCompleter().get_completions(
        SimpleNamespace(text_before_cursor="/he"), None,
    ))
    assert completions[0].text == "help"
    assert completions[0].start_position == -2  # 取代 "he"，不是整個 "/he"


def test_memory_argument_suggestions_are_the_subcommands():
    assert cli._arg_suggestions("memory") == ["list", "add", "del"]


# ---------------------------------------------------------------------------
# run_memory() — 見 CLAUDE.md：使用者在 CLI 問「我要怎麼編輯你的記憶?」，
# 這句沒有訓練資料可以接住、掉回 GRU 亂答，但這其實是個真的能回答的問題——
# GUI 早就有 /memory，只是 CLI 沒有移植過來。這裡不重新設計，直接複用
# memory_store.py，跟 test_command.py 的 palette 測試同一套隔離方式。
# ---------------------------------------------------------------------------

def _isolate_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MEMORY_PATH", tmp_path / "memory.json")


def _printed(capsys) -> str:
    return capsys.readouterr().out


def test_run_memory_add_then_list_round_trips(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("add project 這是一筆測試記憶")
    added = _printed(capsys)
    assert "已新增記憶" in added
    assert "這是一筆測試記憶" in added

    cli.run_memory("")
    listed = _printed(capsys)
    assert "這是一筆測試記憶" in listed
    assert "(Project)" in listed


def test_run_memory_list_filters_by_category(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("add project 專案記憶")
    cli.run_memory("add coding_style 風格記憶")
    capsys.readouterr()

    cli.run_memory("list coding_style")
    listed = _printed(capsys)
    assert "風格記憶" in listed
    assert "專案記憶" not in listed


def test_run_memory_add_rejects_unknown_category(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("add not_a_real_category 內容")
    assert "未知分類" in _printed(capsys)
    assert memory_store.list_memories() == []


def test_run_memory_delete_removes_entry(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("add project 要被刪除的記憶")
    entry_id = memory_store.list_memories()[0]["id"]
    capsys.readouterr()

    cli.run_memory(f"del {entry_id}")
    assert "已刪除記憶" in _printed(capsys)
    assert memory_store.list_memories() == []


def test_run_memory_delete_unknown_id_reports_not_found(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("del doesnotexist")
    assert "找不到記憶" in _printed(capsys)


def test_run_memory_unknown_subcommand_shows_usage(tmp_path, monkeypatch, capsys):
    _isolate_memory(tmp_path, monkeypatch)
    cli.run_memory("frobnicate")
    assert "用法" in _printed(capsys)
