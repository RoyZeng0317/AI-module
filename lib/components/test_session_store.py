"""Unit tests for session_store.py — the autosave backing /resume (回到電腦
未關機前記錄下的對話). Each test uses its own tmp_path file so runs don't
touch the project's real memory/session.json, same isolation pattern as
test_memory_store.py.
"""

from lib.components.session_store import (
    MAX_SESSION_TURNS,
    clear_session,
    default_export_path,
    export_markdown,
    load_session,
    record_turn,
    save_chat_export,
    session_subject,
)


def test_record_turn_persists_and_load_reads_it_back(tmp_path):
    path = tmp_path / "session.json"
    entry = record_turn("你好", "哈囉，我是 sinco", persona="sinco", mode="auto", path=path)

    assert entry["user"] == "你好"
    assert entry["reply"] == "哈囉，我是 sinco"
    assert entry["persona"] == "sinco"
    assert entry["mode"] == "auto"
    assert "id" in entry and "created_at" in entry

    entries = load_session(path=path)
    assert entries == [entry]


def test_record_turn_appends_in_order(tmp_path):
    path = tmp_path / "session.json"
    record_turn("第一句", "回覆一", path=path)
    record_turn("第二句", "回覆二", path=path)

    entries = load_session(path=path)
    assert [e["user"] for e in entries] == ["第一句", "第二句"]


def test_load_session_empty_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert load_session(path=path) == []


def test_record_turn_caps_history_length(tmp_path):
    path = tmp_path / "session.json"
    for i in range(MAX_SESSION_TURNS + 10):
        record_turn(f"訊息{i}", f"回覆{i}", path=path)

    entries = load_session(path=path)
    assert len(entries) == MAX_SESSION_TURNS
    # 保留的是最新的那些，不是最舊的
    assert entries[-1]["user"] == f"訊息{MAX_SESSION_TURNS + 9}"
    assert entries[0]["user"] == f"訊息{10}"


def test_clear_session_empties_the_file(tmp_path):
    path = tmp_path / "session.json"
    record_turn("你好", "哈囉", path=path)

    clear_session(path=path)

    assert load_session(path=path) == []


# ---------------------------------------------------------------------------
# session_subject() — /resume 顯示用的主旨，現算不持久化
# ---------------------------------------------------------------------------

def test_session_subject_empty_without_entries():
    assert session_subject([]) == ""


def test_session_subject_picks_repeated_keyword():
    entries = [
        {"user": "幫我訓練模型", "reply": "..."},
        {"user": "訓練模型要多久", "reply": "..."},
        {"user": "訓練模型完成了嗎", "reply": "..."},
    ]
    assert "訓練" in session_subject(entries) or "練模" in session_subject(entries)


def test_session_subject_falls_back_to_first_message_when_no_keyword_survives():
    entries = [{"user": "嗨", "reply": "你好"}]
    assert session_subject(entries) == "嗨"


def test_session_subject_regenerates_after_clear(tmp_path):
    path = tmp_path / "session.json"
    record_turn("訓練模型進度", "...", path=path)
    record_turn("訓練模型完成", "...", path=path)
    first_subject = session_subject(load_session(path=path))
    assert first_subject != ""

    clear_session(path=path)
    assert session_subject(load_session(path=path)) == ""

    record_turn("查詢台北天氣", "...", path=path)
    record_turn("查詢台北天氣預報", "...", path=path)
    second_subject = session_subject(load_session(path=path))
    assert second_subject != "" and second_subject != first_subject


# ---------------------------------------------------------------------------
# export_markdown() / save_chat_export() — /chat 下載對話紀錄成 Markdown
# ---------------------------------------------------------------------------

def test_export_markdown_reports_empty_session():
    assert "沒有記錄下的對話" in export_markdown([])


def test_export_markdown_includes_turns_and_subject():
    entries = [
        {"user": "你好", "reply": "哈囉，我是 sinco", "persona": "sinco",
         "created_at": "2026-08-28T00:00:00+00:00"},
    ]
    text = export_markdown(entries)
    assert "你好" in text
    assert "哈囉，我是 sinco" in text
    assert "主旨" in text


def test_default_export_path_lands_under_given_base_dir(tmp_path):
    path = default_export_path(base_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("chat_") and path.suffix == ".md"


def test_save_chat_export_writes_file_and_returns_path(tmp_path):
    entries = [
        {"user": "你好", "reply": "哈囉", "persona": "sinco",
         "created_at": "2026-08-28T00:00:00+00:00"},
    ]
    target = tmp_path / "my_chat.md"

    saved = save_chat_export(entries, path=target)

    assert saved == target
    assert target.exists()
    assert "你好" in target.read_text(encoding="utf-8")


def test_save_chat_export_creates_missing_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "dir" / "chat.md"
    saved = save_chat_export([], path=target)
    assert saved.exists()
