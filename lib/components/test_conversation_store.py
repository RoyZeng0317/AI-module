"""Unit tests for conversation_store.py — the multi-conversation store shared
by the desktop GUI, terminal CLI, and web frontend (memory/conversations.json).
Each test uses its own tmp_path file so runs don't touch the project's real
conversations.json, same isolation pattern as test_session_store.py.
"""

from lib.components.conversation_store import (
    DEFAULT_TITLE,
    append_message,
    clear_messages,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)


def test_create_conversation_starts_empty_with_default_title(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)

    assert conv["title"] == DEFAULT_TITLE
    assert conv["messages"] == []
    assert "id" in conv and "created_at" in conv and "updated_at" in conv


def test_list_conversations_sorted_newest_first(tmp_path):
    path = tmp_path / "conversations.json"
    first = create_conversation(path=path)
    second = create_conversation(path=path)
    append_message(second["id"], "user", "second 對話後來又動了一下", path=path)

    ids_in_order = [c["id"] for c in list_conversations(path=path)]
    assert ids_in_order[0] == second["id"]
    assert first["id"] in ids_in_order


def test_append_message_auto_titles_from_first_user_message(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)

    updated = append_message(conv["id"], "user", "幫我看一下這段程式碼", path=path)
    assert updated["title"] == "幫我看一下這段程式碼"
    assert updated["messages"] == [{"role": "user", "content": "幫我看一下這段程式碼", "ts": updated["messages"][0]["ts"]}]


def test_append_message_stores_persona_and_mode_when_given(tmp_path):
    # persona/mode 是給 GUI/CLI 用的（記錄當時的 /character 人格、/model 模式）
    # ——網頁端不帶這兩個引數，訊息物件應該維持精簡形狀（見上一項測試）。
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)

    updated = append_message(conv["id"], "assistant", "哈囉，我是 sinco", persona="sinco", mode="auto", path=path)

    assert updated["messages"][0]["persona"] == "sinco"
    assert updated["messages"][0]["mode"] == "auto"


def test_append_message_truncates_long_title(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)
    long_text = "一二三四五六七八九十" * 5  # 50 字，超過 24 字上限

    updated = append_message(conv["id"], "user", long_text, path=path)
    assert updated["title"].endswith("…")
    assert len(updated["title"]) == 25  # 24 字 + 省略號


def test_append_message_does_not_retitle_after_first_message(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)
    append_message(conv["id"], "user", "第一句話", path=path)
    updated = append_message(conv["id"], "user", "第二句話不應該變成標題", path=path)

    assert updated["title"] == "第一句話"
    assert len(updated["messages"]) == 2


def test_append_message_returns_none_for_unknown_conversation(tmp_path):
    path = tmp_path / "conversations.json"
    assert append_message("not-a-real-id", "user", "hi", path=path) is None


def test_get_conversation_round_trips_messages(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)
    append_message(conv["id"], "user", "哈囉", persona="sinco", mode="auto", path=path)
    append_message(conv["id"], "assistant", "你好", persona="sinco", mode="auto", path=path)

    fetched = get_conversation(conv["id"], path=path)
    assert [m["role"] for m in fetched["messages"]] == ["user", "assistant"]


def test_rename_conversation_updates_title(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)
    renamed = rename_conversation(conv["id"], "自訂標題", path=path)

    assert renamed["title"] == "自訂標題"
    assert get_conversation(conv["id"], path=path)["title"] == "自訂標題"


def test_rename_conversation_returns_none_for_unknown_id(tmp_path):
    path = tmp_path / "conversations.json"
    assert rename_conversation("not-a-real-id", "x", path=path) is None


def test_clear_messages_empties_but_keeps_conversation_id(tmp_path):
    path = tmp_path / "conversations.json"
    conv = create_conversation(path=path)
    append_message(conv["id"], "user", "會被清掉的訊息", path=path)

    cleared = clear_messages(conv["id"], path=path)
    assert cleared["id"] == conv["id"]
    assert cleared["messages"] == []
    assert cleared["title"] == DEFAULT_TITLE


def test_delete_conversation_removes_matching_id_only(tmp_path):
    path = tmp_path / "conversations.json"
    keep = create_conversation(path=path)
    remove = create_conversation(path=path)

    assert delete_conversation(remove["id"], path=path) is True
    remaining_ids = [c["id"] for c in list_conversations(path=path)]
    assert remaining_ids == [keep["id"]]


def test_delete_conversation_returns_false_for_unknown_id(tmp_path):
    path = tmp_path / "conversations.json"
    create_conversation(path=path)
    assert delete_conversation("not-a-real-id", path=path) is False


def test_load_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert list_conversations(path=path) == []
