"""Unit tests for command.py — CommandPalette's /model, /memory, /init,
/character, /learn dispatch and the "/指令 " argument-suggestion mechanism
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

import auto_learn
import lib.components.command as command
import lib.components.conversation_store as conversation_store
import lib.components.memory_store as memory_store
import lib.components.session_store as session_store
from chats import DEFAULT_OUT_DIR
from lib.components.command import ARG_SUGGESTIONS, CommandPalette


@pytest.fixture(scope="module")
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def palette(root, tmp_path, monkeypatch):
    # 每個測試各自的 memory.json / auto_learn 候選檔，不去動真正的專案資料
    monkeypatch.setattr(memory_store, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(auto_learn, "CANDIDATES_PATH", tmp_path / "auto_learn_candidates.json")
    monkeypatch.setattr(auto_learn, "PAIRS_PATH", tmp_path / "pairs.json")
    monkeypatch.setattr(session_store, "SESSION_PATH", tmp_path / "session.json")
    monkeypatch.setattr(conversation_store, "CONVERSATIONS_PATH", tmp_path / "conversations.json")

    message = tk.Entry(root)
    chat_display = tk.Text(root)
    chat_display.tag_configure("system")
    conversation = SimpleNamespace(
        force_mode="auto", persona="sinco", out_dir=DEFAULT_OUT_DIR,
        conversation_id=None, history=[],
        ask=lambda *a, **k: conversation.ask_calls.append((a, k)),
        send_message=lambda text: conversation.send_calls.append(text),
        ask_calls=[],
        send_calls=[],
    )
    return CommandPalette(root, message, chat_display, conversation)


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    # @ 檔案附加一律以 PROJECT_ROOT 為根，測試改指到 tmp_path，才能自己擺
    # 一份乾淨的假檔案樹，不會動到真正的專案內容。
    monkeypatch.setattr(command, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _set_input(palette: CommandPalette, text: str, cursor=None):
    palette.message.delete(0, tk.END)
    palette.message.insert(0, text)
    palette.message.icursor(tk.END if cursor is None else cursor)


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
    assert ARG_SUGGESTIONS["model"] == ["auto", "sinco", "code", "nvidia"]


def test_update_suggestions_offers_model_arg_options(palette):
    palette.message.insert(0, "/model ")
    palette.update_suggestions()
    assert palette.current_matches == ["auto", "sinco", "code", "nvidia"]
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
# /learn
# ---------------------------------------------------------------------------

def test_learn_list_when_empty(palette):
    palette.run("learn")
    assert "沒有待審核" in _chat_text(palette)


def test_learn_list_shows_saved_candidate(palette):
    auto_learn.save_candidate("你會微積分嗎", "微積分是...", "微積分", path=auto_learn.CANDIDATES_PATH)
    palette.run("learn list")
    assert "你會微積分嗎" in _chat_text(palette)
    assert "微積分是..." in _chat_text(palette)


def test_learn_approve_writes_to_pairs_and_clears_candidate(palette):
    auto_learn.PAIRS_PATH.write_text('[{"prompt": "hi", "reply": "hi there"}]', encoding="utf-8")
    entry = auto_learn.save_candidate("你會微積分嗎", "微積分是...", "微積分", path=auto_learn.CANDIDATES_PATH)

    palette.run(f"learn approve {entry['id']}")

    assert "已核准" in _chat_text(palette)
    assert auto_learn.list_candidates(path=auto_learn.CANDIDATES_PATH) == []

    import json
    pairs = json.loads(auto_learn.PAIRS_PATH.read_text(encoding="utf-8"))
    assert {"prompt": "你會微積分嗎", "reply": "微積分是..."} in pairs


def test_learn_approve_unknown_id_reports_not_found(palette):
    palette.run("learn approve does-not-exist")
    assert "找不到候選" in _chat_text(palette)


def test_learn_reject_removes_candidate(palette):
    entry = auto_learn.save_candidate("你會微積分嗎", "微積分是...", "微積分", path=auto_learn.CANDIDATES_PATH)
    palette.run(f"learn reject {entry['id']}")
    assert "已捨棄候選" in _chat_text(palette)
    assert auto_learn.list_candidates(path=auto_learn.CANDIDATES_PATH) == []


def test_learn_unknown_subcommand_shows_usage(palette):
    palette.run("learn not-a-subcommand")
    assert "用法" in _chat_text(palette)


# ---------------------------------------------------------------------------
# /resume — 回到電腦未關機前記錄下的對話（session_store.py）
# ---------------------------------------------------------------------------

def test_resume_without_recorded_session_reports_empty(palette):
    palette.run("resume")
    assert "沒有記錄下的對話" in _chat_text(palette)


def test_resume_replays_recorded_turns_and_restores_history(palette):
    session_store.record_turn("你好", "哈囉，我是 sinco", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)
    session_store.record_turn("再見", "掰掰", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)

    palette.run("resume")

    text = _chat_text(palette)
    assert "你好" in text and "哈囉，我是 sinco" in text
    assert "再見" in text and "掰掰" in text
    assert palette.conversation.history == [("你好", "哈囉，我是 sinco"), ("再見", "掰掰")]


def test_resume_clear_wipes_recorded_session(palette):
    session_store.record_turn("你好", "哈囉", path=session_store.SESSION_PATH)

    palette.run("resume clear")

    assert "已清除" in _chat_text(palette)
    assert session_store.load_session(path=session_store.SESSION_PATH) == []


def test_resume_arg_suggestions_registered():
    assert ARG_SUGGESTIONS["resume"] == ["clear"]


def test_resume_replay_includes_subject(palette):
    session_store.record_turn("幫我訓練模型", "好的", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)
    session_store.record_turn("訓練模型完成了嗎", "還沒", persona="sinco", mode="auto",
                               path=session_store.SESSION_PATH)

    palette.run("resume")

    assert "主旨" in _chat_text(palette)


# ---------------------------------------------------------------------------
# /chat — 把記錄下的對話下載成 Markdown 檔案（session_store.py）
# ---------------------------------------------------------------------------

def test_chat_with_explicit_path_saves_without_dialog(palette, tmp_path):
    session_store.record_turn("你好", "哈囉，我是 sinco", path=session_store.SESSION_PATH)
    target = tmp_path / "exported.md"

    palette.run(f"chat {target}")

    assert target.exists()
    assert "你好" in target.read_text(encoding="utf-8")
    assert "已下載對話紀錄" in _chat_text(palette)


def test_chat_without_arg_opens_save_dialog(palette, monkeypatch, tmp_path):
    session_store.record_turn("你好", "哈囉", path=session_store.SESSION_PATH)
    target = tmp_path / "picked.md"
    monkeypatch.setattr(command.filedialog, "asksaveasfilename", lambda **kwargs: str(target))

    palette.run("chat")

    assert target.exists()
    assert "已下載對話紀錄" in _chat_text(palette)


def test_chat_without_arg_cancelled_dialog_does_nothing(palette, monkeypatch, tmp_path):
    monkeypatch.setattr(command.filedialog, "asksaveasfilename", lambda **kwargs: "")
    palette.run("chat")
    assert _chat_text(palette).strip() == ""


def test_chat_export_handles_empty_session(palette, tmp_path):
    target = tmp_path / "empty.md"
    palette.run(f"chat {target}")
    assert "沒有記錄下的對話" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# /conversations — 跟 CLI／網頁共用的多筆對話紀錄（conversation_store.py）
# ---------------------------------------------------------------------------

def test_conversations_list_reports_empty(palette):
    palette.run("conversations")
    assert "沒有任何對話紀錄" in _chat_text(palette)


def test_conversations_new_creates_and_switches_current_conversation(palette):
    palette.run("conversations new")

    assert palette.conversation.conversation_id is not None
    assert conversation_store.list_conversations()[0]["id"] == palette.conversation.conversation_id
    assert "已建立新對話" in _chat_text(palette)


def test_conversations_list_marks_current_conversation(palette):
    palette.run("conversations new")
    conv_id = palette.conversation.conversation_id

    palette.run("conversations list")

    assert f"→ [{conv_id}]" in _chat_text(palette)


def test_conversations_open_restores_history_from_shared_store(palette):
    conv = conversation_store.create_conversation()
    conversation_store.append_message(conv["id"], "user", "你好", persona="sinco", mode="auto")
    conversation_store.append_message(conv["id"], "assistant", "哈囉，我是 sinco", persona="sinco", mode="auto")

    palette.run(f"conversations open {conv['id']}")

    assert palette.conversation.conversation_id == conv["id"]
    assert palette.conversation.history == [("你好", "哈囉，我是 sinco")]
    text = _chat_text(palette)
    assert "你好" in text and "哈囉，我是 sinco" in text


def test_conversations_open_unknown_id_reports_error(palette):
    palette.run("conversations open not-a-real-id")
    assert "找不到對話" in _chat_text(palette)


def test_conversations_arg_suggestions_registered():
    assert ARG_SUGGESTIONS["conversations"] == ["list", "new", "open"]


def test_gui_and_web_writes_share_the_same_conversation(palette):
    # GUI（command.py）跟網頁（web/backend/conversation_store.py）各自維護一份
    # 幾乎相同的模組，但只要指向同一個實體檔案，讀寫就該完全互通——這裡直接
    # 用 conversation_store（GUI 這邊 import 的模組）模擬「網頁那邊建立/回覆過
    # 一則對話」，確認 GUI 的 /conversations 能看到它。
    conv = conversation_store.create_conversation()
    conversation_store.append_message(conv["id"], "user", "從網頁送出的訊息")

    palette.run("conversations list")

    assert conv["id"] in _chat_text(palette)


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


# ---------------------------------------------------------------------------
# @ 檔案附加：mention_matches()（候選清單）
# ---------------------------------------------------------------------------

def test_mention_matches_lists_root_folders_before_files(project_root):
    (project_root / "app").mkdir()
    (project_root / "readme.txt").write_text("hi", encoding="utf-8")

    assert command.mention_matches("") == ["app/", "readme.txt"]


def test_mention_matches_filters_by_prefix(project_root):
    (project_root / "readme.txt").write_text("hi", encoding="utf-8")
    (project_root / "recipe.txt").write_text("hi", encoding="utf-8")
    (project_root / "notes.txt").write_text("hi", encoding="utf-8")

    assert command.mention_matches("re") == ["readme.txt", "recipe.txt"]


def test_mention_matches_browses_subfolder_given_slash(project_root):
    sub = project_root / "app"
    sub.mkdir()
    (sub / "command.py").write_text("code", encoding="utf-8")
    (sub / "other.py").write_text("code", encoding="utf-8")

    assert command.mention_matches("app/comm") == ["app/command.py"]


def test_mention_matches_blocks_escaping_project_root(project_root):
    assert command.mention_matches("../") == []
    assert command.mention_matches("../secrets") == []


def test_mention_matches_excludes_hidden_and_ignored_entries(project_root):
    (project_root / ".git").mkdir()
    (project_root / "__pycache__").mkdir()
    (project_root / ".env").write_text("SECRET=1", encoding="utf-8")
    (project_root / "app.py").write_text("code", encoding="utf-8")

    assert command.mention_matches("") == ["app.py"]


def test_mention_matches_caps_result_count(project_root):
    for i in range(command.MAX_FILE_SUGGESTIONS + 5):
        (project_root / f"file{i:02d}.txt").write_text("x", encoding="utf-8")

    assert len(command.mention_matches("")) == command.MAX_FILE_SUGGESTIONS


# ---------------------------------------------------------------------------
# @ 檔案附加：磁碟絕對路徑（跳出專案資料夾讀檔，見 resolve_mention()）
# ---------------------------------------------------------------------------

def test_is_external_path_detects_windows_unc_and_home_paths():
    assert command._is_external_path("C:/Users/Roy/Downloads/file.py")
    assert command._is_external_path("C:\\Users\\Roy\\Downloads\\file.py")
    assert command._is_external_path("\\\\server\\share\\file.py")
    assert command._is_external_path("~/Downloads/file.py")
    assert not command._is_external_path("readme.txt")
    assert not command._is_external_path("../secrets")


def test_mention_matches_browses_external_absolute_folder(project_root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "gemini-code-123.py").write_text("print(1)", encoding="utf-8")
    (outside / "notes.txt").write_text("hi", encoding="utf-8")

    norm_outside = str(outside).replace("\\", "/")
    assert command.mention_matches(f"{outside}/gemini") == [f"{norm_outside}/gemini-code-123.py"]


def test_mention_matches_still_blocks_relative_escape_when_not_absolute(project_root):
    assert command.mention_matches("../") == []
    assert command.mention_matches("../secrets") == []


def test_resolve_mention_allows_external_absolute_file(project_root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "gemini-code-123.py"
    outside.write_text("print(1)", encoding="utf-8")

    assert command.resolve_mention(str(outside)) == outside


def test_resolve_mention_rejects_external_path_that_is_a_directory(project_root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    assert command.resolve_mention(str(outside)) is None


def test_resolve_mention_still_blocks_relative_escape(project_root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("機密", encoding="utf-8")

    assert command.resolve_mention(f"../{outside.parent.name}/secret.txt") is None


# ---------------------------------------------------------------------------
# @ 檔案附加：_current_mention()（目前游標左邊在打的 @提及）
# ---------------------------------------------------------------------------

def test_current_mention_detects_token_at_cursor(palette):
    _set_input(palette, "請看 @read")
    assert palette._current_mention() == (3, 8, "read")


def test_current_mention_none_without_at_symbol(palette):
    _set_input(palette, "hello world")
    assert palette._current_mention() is None


def test_current_mention_none_when_whitespace_breaks_token(palette):
    _set_input(palette, "@readme.txt 已經打完了")
    assert palette._current_mention() is None


def test_current_mention_empty_token_right_after_at(palette):
    _set_input(palette, "@")
    assert palette._current_mention() == (0, 1, "")


# ---------------------------------------------------------------------------
# @ 檔案附加：update_suggestions() 接上 mention_matches()
# ---------------------------------------------------------------------------

def test_update_suggestions_shows_file_matches_for_at_token(palette, project_root):
    (project_root / "readme.txt").write_text("hi", encoding="utf-8")
    _set_input(palette, "請看 @re")

    palette.update_suggestions()

    assert palette.current_matches == ["readme.txt"]
    assert palette._file_mention_span == (3, 6)


def test_update_suggestions_falls_back_to_slash_commands_without_at(palette):
    _set_input(palette, "/mo")
    palette.update_suggestions()
    assert palette.current_matches == ["model"]
    assert palette._file_mention_span is None


# ---------------------------------------------------------------------------
# @ 檔案附加：choose_suggestion() 針對 @提及 的插入行為
# ---------------------------------------------------------------------------

def test_choose_suggestion_inserts_file_mention_with_trailing_space(palette, project_root):
    (project_root / "readme.txt").write_text("hi", encoding="utf-8")
    _set_input(palette, "請看 @re")
    palette.update_suggestions()

    palette.choose_suggestion("readme.txt")

    assert palette.message.get() == "請看 @readme.txt "
    assert palette.message.index(tk.INSERT) == len("請看 @readme.txt ")


def test_choose_suggestion_inserts_folder_mention_without_trailing_space(palette, project_root):
    (project_root / "app").mkdir()
    _set_input(palette, "@a")
    palette.update_suggestions()

    palette.choose_suggestion("app/")

    assert palette.message.get() == "@app/"
    assert palette.message.index(tk.INSERT) == len("@app/")


def test_choose_suggestion_keeps_text_after_mention_intact(palette, project_root):
    (project_root / "readme.txt").write_text("hi", encoding="utf-8")
    _set_input(palette, "請看 @re 這個檔案", cursor=6)  # 游標停在 "@re" 之後

    palette.update_suggestions()
    palette.choose_suggestion("readme.txt")

    assert palette.message.get() == "請看 @readme.txt  這個檔案"


# ---------------------------------------------------------------------------
# @ 檔案附加：file()（送出訊息時展開 @提及 成檔案內容）
# ---------------------------------------------------------------------------

def test_file_without_mentions_delegates_to_send_message(palette):
    palette.file("hello sinco")
    assert palette.conversation.send_calls == ["hello sinco"]
    assert palette.conversation.ask_calls == []


def test_file_with_mention_reads_content_and_calls_ask(palette, project_root):
    (project_root / "notes.txt").write_text("這是筆記內容", encoding="utf-8")

    palette.file("幫我看 @notes.txt 這是什麼")

    assert palette.conversation.ask_calls
    (header, model_input), kwargs = palette.conversation.ask_calls[0]
    assert header == "幫我看 @notes.txt 這是什麼"
    assert "這是筆記內容" in model_input
    assert "幫我看 @notes.txt 這是什麼" in model_input
    assert kwargs["record_as"] == "幫我看 @notes.txt 這是什麼"


def test_file_with_multiple_mentions_attaches_each_once(palette, project_root):
    (project_root / "a.txt").write_text("內容A", encoding="utf-8")
    (project_root / "b.txt").write_text("內容B", encoding="utf-8")

    palette.file("比較 @a.txt 跟 @b.txt 跟 @a.txt")

    (header, model_input), kwargs = palette.conversation.ask_calls[0]
    assert model_input.count("內容A") == 1
    assert model_input.count("內容B") == 1


def test_file_with_unknown_mention_reports_error_and_sends_nothing(palette, project_root):
    palette.file("看看 @does-not-exist.txt")

    assert "未知檔案" in _chat_text(palette)
    assert palette.conversation.ask_calls == []
    assert palette.conversation.send_calls == []


def test_file_mention_escaping_project_root_is_rejected(palette, project_root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("機密", encoding="utf-8")

    palette.file(f"看看 @../{outside.parent.name}/secret.txt")

    assert "未知檔案" in _chat_text(palette)
    assert palette.conversation.ask_calls == []


def test_file_with_external_absolute_mention_reads_content(palette, project_root, tmp_path_factory):
    # 對應真實情境：使用者從檔案總管複製一個專案資料夾以外的檔案路徑，貼進
    # @ 附加——這應該要能讀到內容並送給模型（例如 /model nvidia），不再被
    # 「一律限制在專案資料夾內」擋下來。
    outside = tmp_path_factory.mktemp("outside") / "gemini-code-123.py"
    outside.write_text("print('hi')", encoding="utf-8")

    palette.file(f"幫我看 @{outside} 這支程式")

    assert palette.conversation.ask_calls
    (header, model_input), kwargs = palette.conversation.ask_calls[0]
    assert "print('hi')" in model_input


# ---------------------------------------------------------------------------
# @ 檔案附加：候選清單旁的檔案內容預覽面板（仿 Claude Code 打 @ 的體驗）
# ---------------------------------------------------------------------------

def test_update_suggestions_populates_preview_for_first_match(palette, project_root):
    (project_root / "readme.txt").write_text("第一行\n第二行", encoding="utf-8")
    _set_input(palette, "請看 @re")

    palette.update_suggestions()

    assert palette.preview_text is not None
    assert "第一行" in palette.preview_text.get("1.0", tk.END)


def test_update_suggestions_no_preview_pane_for_slash_commands(palette):
    _set_input(palette, "/mo")
    palette.update_suggestions()
    assert palette.preview_text is None


def test_update_suggestions_no_preview_pane_for_model_arg(palette):
    _set_input(palette, "/model ")
    palette.update_suggestions()
    assert palette.preview_text is None


def test_preview_shows_placeholder_for_folder_candidate(palette, project_root):
    (project_root / "app").mkdir()
    _set_input(palette, "@a")

    palette.update_suggestions()

    assert "資料夾" in palette.preview_text.get("1.0", tk.END)


def test_move_selection_updates_preview_to_selected_file(palette, project_root):
    (project_root / "a.txt").write_text("內容A", encoding="utf-8")
    (project_root / "b.txt").write_text("內容B", encoding="utf-8")
    _set_input(palette, "@")

    palette.update_suggestions()
    assert "內容A" in palette.preview_text.get("1.0", tk.END)

    palette.move_selection(1)  # 第一次按下鍵：反白第一個候選（跟一開始的預設預覽一樣還是 a.txt）
    palette.move_selection(1)  # 第二次才移到 b.txt
    assert "內容B" in palette.preview_text.get("1.0", tk.END)


def test_hide_suggestions_clears_preview_state(palette, project_root):
    (project_root / "a.txt").write_text("內容A", encoding="utf-8")
    _set_input(palette, "@a")
    palette.update_suggestions()
    assert palette.preview_text is not None

    palette.hide_suggestions()

    assert palette.preview_text is None
    assert palette._match_paths is None
