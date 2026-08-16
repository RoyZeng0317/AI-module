"""Unit tests for session_store.py — the autosave backing /resume (回到電腦
未關機前記錄下的對話). Each test uses its own tmp_path file so runs don't
touch the project's real memory/session.json, same isolation pattern as
test_memory_store.py.
"""

from lib.components.session_store import (
    MAX_SESSION_TURNS,
    clear_session,
    load_session,
    record_turn,
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
