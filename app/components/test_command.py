"""Unit tests for command.py — CommandPalette's /model, /memory, /init,
/character dispatch and the "/指令 " argument-suggestion mechanism
(CLAUDE.md 需求 #01, #02). tk.Entry/tk.Text need a live Tcl interpreter,
so these spin up one hidden (withdrawn) root window, same pattern as
tranning/test_train_gui.py.
"""

import sys
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

# character_browser.py（/character 指令的目標）匯入 chats.py，需要 tranning/
# 在 sys.path 上——正常執行時是 home_screen.py 負責這件事，這裡的測試獨立
# 執行，所以自己補上，跟 home_screen.py 用的是同一個相對路徑寫法。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tranning"))

import command
import memory_store
from chats import DEFAULT_OUT_DIR
from command import ARG_SUGGESTIONS, CommandPalette


@pytest.fixture(scope="module")
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def palette(root, tmp_path, monkeypatch):
    # 每個測試各自的 memory.json，不去動真正的專案記憶檔
    monkeypatch.setattr(memory_store, "MEMORY_PATH", tmp_path / "memory.json")

    message = tk.Entry(root)
    chat_display = tk.Text(root)
    chat_display.tag_configure("system")
    conversation = SimpleNamespace(
        force_mode="auto", persona="sinco", out_dir=DEFAULT_OUT_DIR, ask=lambda *a, **k: None,
    )
    return CommandPalette(root, message, chat_display, conversation)


def _chat_text(palette: CommandPalette) -> str:
    return palette.chat_display.get("1.0", tk.END)


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------

def test_model_no_arg_reports_current_mode(palette):
    palette.run("model")
    assert "目前模式：auto" in _chat_text(palette)


def test_model_switches_conversation_force_mode(palette):
    palette.run("model code")
    assert palette.conversation.force_mode == "code"
    assert "已切換模式：code" in _chat_text(palette)


def test_model_rejects_unknown_mode_without_changing_state(palette):
    palette.run("model not-a-mode")
    assert palette.conversation.force_mode == "auto"
    assert "未知模式" in _chat_text(palette)


def test_model_arg_suggestions_registered():
    assert ARG_SUGGESTIONS["model"] == ["auto", "sinco", "code"]


def test_update_suggestions_offers_model_arg_options(palette):
    palette.message.insert(0, "/model ")
    palette.update_suggestions()
    assert palette.current_matches == ["auto", "sinco", "code"]
    assert palette._suggesting_arg_for == "model"


def test_update_suggestions_filters_model_arg_by_prefix(palette):
    palette.message.insert(0, "/model c")
    palette.update_suggestions()
    assert palette.current_matches == ["code"]


def test_choose_suggestion_completes_argument_not_command_name(palette):
    palette.message.insert(0, "/model ")
    palette.update_suggestions()
    palette.choose_suggestion("code")
    assert palette.message.get() == "/model code"


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------

def test_memory_list_when_empty(palette):
    palette.run("memory")
    assert "沒有記憶" in _chat_text(palette)


def test_memory_add_then_list_round_trip(palette):
    palette.run("memory add project sinco 是自建 seq2seq 聊天模型")
    assert "已新增記憶" in _chat_text(palette)

    palette.chat_display.configure(state="normal")
    palette.chat_display.delete("1.0", tk.END)
    palette.chat_display.configure(state="disabled")

    palette.run("memory")
    assert "sinco 是自建 seq2seq 聊天模型" in _chat_text(palette)


def test_memory_add_rejects_unknown_category(palette):
    palette.run("memory add not-a-category 某些內容")
    assert "未知分類" in _chat_text(palette)


def test_memory_delete_unknown_id_reports_not_found(palette):
    palette.run("memory del does-not-exist")
    assert "找不到記憶" in _chat_text(palette)


def test_memory_delete_existing_entry(palette):
    entries_before = memory_store.add_memory("project", "待刪除", path=memory_store.MEMORY_PATH)
    palette.run(f"memory del {entries_before['id']}")
    assert "已刪除記憶" in _chat_text(palette)
    assert memory_store.list_memories(path=memory_store.MEMORY_PATH) == []


def test_memory_unknown_subcommand_shows_usage(palette):
    palette.run("memory not-a-subcommand")
    assert "用法" in _chat_text(palette)


# ---------------------------------------------------------------------------
# /init
# ---------------------------------------------------------------------------

def test_init_with_explicit_folder_writes_claude_md(palette, tmp_path):
    target = tmp_path / "some_project"
    target.mkdir()

    palette.run(f"init {target}")

    assert (target / "CLAUDE.md").exists()
    assert "已產生/更新" in _chat_text(palette)


def test_init_with_missing_folder_reports_error(palette, tmp_path):
    missing = tmp_path / "does_not_exist"
    palette.run(f"init {missing}")
    assert "找不到資料夾" in _chat_text(palette)


def test_init_without_arg_opens_folder_dialog(palette, monkeypatch, tmp_path):
    target = tmp_path / "picked"
    target.mkdir()
    monkeypatch.setattr(command.filedialog, "askdirectory", lambda **kwargs: str(target))

    palette.run("init")

    assert (target / "CLAUDE.md").exists()


def test_init_without_arg_cancelled_dialog_does_nothing(palette, monkeypatch):
    monkeypatch.setattr(command.filedialog, "askdirectory", lambda **kwargs: "")
    palette.run("init")
    assert _chat_text(palette).strip() == ""


# ---------------------------------------------------------------------------
# /character
# ---------------------------------------------------------------------------

def test_character_opens_browser_with_windows_and_conversation(palette, monkeypatch):
    captured = {}

    def fake_open(parent, conversation):
        captured["parent"] = parent
        captured["conversation"] = conversation
        return None

    monkeypatch.setattr("character_browser.open_character_browser", fake_open)

    palette.run("character")

    assert captured["parent"] is palette.windows
    assert captured["conversation"] is palette.conversation


# ---------------------------------------------------------------------------
# 既有行為沒有被新指令弄壞：未知指令 / .md 指令仍照舊運作
# ---------------------------------------------------------------------------

def test_unknown_command_still_reports_error(palette):
    palette.run("this-command-does-not-exist")
    assert "未知指令" in _chat_text(palette)


def test_clear_still_wipes_chat_display(palette):
    palette.run("memory")  # 先寫一些東西進去
    assert _chat_text(palette).strip() != ""
    palette.run("clear")
    assert _chat_text(palette).strip() == ""
