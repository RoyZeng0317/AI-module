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

import lib.components.cli as cli
import lib.components.memory_store as memory_store
import lib.components.session_store as session_store


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


# ---------------------------------------------------------------------------
# run_model() — /model 移植自 GUI 的 CommandPalette._run_model（見
# lib/components/test_command.py 的對應測試），跟 /memory 一樣是 CLI 之前
# 沒有的內建指令，靠 state["force_mode"] 覆蓋 smart_reply_traced() 的自動
# chat/code 判斷。
# ---------------------------------------------------------------------------

def _state(**overrides) -> dict:
    base = {"out_dir": cli.DEFAULT_OUT_DIR, "persona": cli.DEFAULT_PERSONA, "force_mode": "auto", "history": []}
    base.update(overrides)
    return base


def test_model_argument_suggestions_are_the_modes():
    assert cli._arg_suggestions("model") == cli.MODEL_MODES


def test_run_model_no_arg_reports_current_mode(capsys):
    state = _state()
    cli.run_model("", state)
    assert "目前模式：auto" in _printed(capsys)
    assert state["force_mode"] == "auto"


def test_run_model_switches_force_mode(capsys):
    state = _state()
    cli.run_model("code", state)
    assert state["force_mode"] == "code"
    assert "已切換模式：code" in _printed(capsys)


def test_run_model_rejects_unknown_mode_without_changing_state(capsys):
    state = _state()
    cli.run_model("not-a-mode", state)
    assert state["force_mode"] == "auto"
    assert "未知模式" in _printed(capsys)


# ---------------------------------------------------------------------------
# run_resume() — /resume 回到電腦未關機前記錄下的對話（session_store.py），
# 跟 lib/components/test_command.py 的 GUI 對應測試共用同一份 session_store。
# ---------------------------------------------------------------------------

def _isolate_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_PATH", tmp_path / "session.json")


def test_resume_argument_suggestions_are_clear():
    assert cli._arg_suggestions("resume") == ["clear"]


def test_run_resume_without_recorded_session_reports_empty(tmp_path, monkeypatch, capsys):
    _isolate_session(tmp_path, monkeypatch)
    cli.run_resume("", _state())
    assert "沒有記錄下的對話" in _printed(capsys)


def test_run_resume_replays_recorded_turns(tmp_path, monkeypatch, capsys):
    _isolate_session(tmp_path, monkeypatch)
    session_store.record_turn("你好", "哈囉，我是 sinco", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)

    cli.run_resume("", _state())

    printed = _printed(capsys)
    assert "你好" in printed
    assert "哈囉，我是 sinco" in printed


def test_run_resume_restores_history_for_nvidia_context(tmp_path, monkeypatch, capsys):
    """/resume 重播只是唸給人看，不會送進 sinco；但要把 state["history"] 補回
    去，讓 /model nvidia 能接上這段還原的歷史當多輪對話上下文（見
    chats.smart_reply_traced() 的說明）——這是 NVIDIA 模式看起來「沒有還原
    對話紀錄」的根因修正。"""
    _isolate_session(tmp_path, monkeypatch)
    session_store.record_turn("你好", "哈囉，我是 sinco", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)
    state = _state()

    cli.run_resume("", state)

    assert state["history"] == [("你好", "哈囉，我是 sinco")]


def test_run_resume_clear_wipes_recorded_session(tmp_path, monkeypatch, capsys):
    _isolate_session(tmp_path, monkeypatch)
    session_store.record_turn("你好", "哈囉", path=session_store.SESSION_PATH)

    cli.run_resume("clear", _state())

    assert "已清除" in _printed(capsys)
    assert session_store.load_session(path=session_store.SESSION_PATH) == []


def test_ask_model_records_turn_to_session(tmp_path, monkeypatch, capsys):
    _isolate_session(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "smart_reply_traced", lambda *a, **k: ("· trace", "回覆內容"))

    cli.ask_model("測試訊息", _state())

    entries = session_store.load_session(path=session_store.SESSION_PATH)
    assert len(entries) == 1
    assert entries[0]["user"] == "測試訊息"
    assert entries[0]["reply"] == "回覆內容"


def test_ask_model_passes_state_history_to_smart_reply_traced(tmp_path, monkeypatch, capsys):
    """同一個 process 內，state["history"] 要原樣轉交給 smart_reply_traced()，
    /model nvidia 才接得到之前幾輪當上下文（sinco/code 模式會忽略這個參數，
    見 chats.smart_reply_traced() 的說明，這裡只驗證有傳到，不驗證各模式怎麼用）。"""
    _isolate_session(tmp_path, monkeypatch)
    seen = {}

    def fake_smart_reply_traced(message, out_dir, force_mode, history):
        seen["history"] = list(history)  # 快照——ask_model() 接下來會就地 append 同一個 list
        return "· trace", "回覆內容"

    monkeypatch.setattr(cli, "smart_reply_traced", fake_smart_reply_traced)
    state = _state(history=[("之前的問題", "之前的回覆")])

    cli.ask_model("測試訊息", state)

    assert seen["history"] == [("之前的問題", "之前的回覆")]
    assert state["history"] == [("之前的問題", "之前的回覆"), ("測試訊息", "回覆內容")]
