"""Unit tests for memory_store.py — the persistent memory backing /memory
(CLAUDE.md 需求 #01). Each test uses its own tmp_path file so runs don't
touch the project's real memory/memory.json.
"""

import pytest

from memory_store import (
    CATEGORIES,
    add_memory,
    delete_memory,
    format_memories,
    list_memories,
)


def test_add_memory_rejects_unknown_category(tmp_path):
    with pytest.raises(ValueError):
        add_memory("not-a-real-category", "some text", path=tmp_path / "memory.json")


def test_add_memory_rejects_empty_text(tmp_path):
    with pytest.raises(ValueError):
        add_memory("project", "   ", path=tmp_path / "memory.json")


def test_add_memory_persists_and_list_reads_it_back(tmp_path):
    path = tmp_path / "memory.json"
    entry = add_memory("project", "sinco 是自建 seq2seq 聊天模型", path=path)

    assert entry["category"] == "project"
    assert entry["text"] == "sinco 是自建 seq2seq 聊天模型"
    assert "id" in entry and "created_at" in entry

    entries = list_memories(path=path)
    assert entries == [entry]


def test_list_memories_filters_by_category(tmp_path):
    path = tmp_path / "memory.json"
    add_memory("project", "專案筆記", path=path)
    add_memory("coding_style", "PEP8", path=path)

    assert [e["category"] for e in list_memories("coding_style", path=path)] == ["coding_style"]
    assert len(list_memories(path=path)) == 2


def test_delete_memory_removes_matching_id_only(tmp_path):
    path = tmp_path / "memory.json"
    keep = add_memory("project", "保留這筆", path=path)
    remove = add_memory("project", "刪掉這筆", path=path)

    assert delete_memory(remove["id"], path=path) is True
    remaining = list_memories(path=path)
    assert remaining == [keep]


def test_delete_memory_returns_false_for_unknown_id(tmp_path):
    path = tmp_path / "memory.json"
    add_memory("project", "存在的一筆", path=path)
    assert delete_memory("not-a-real-id", path=path) is False


def test_format_memories_empty_and_nonempty(tmp_path):
    assert "沒有記憶" in format_memories([])

    path = tmp_path / "memory.json"
    entry = add_memory("developer_preference", "喜歡簡潔的回覆", path=path)
    text = format_memories([entry])
    assert entry["id"] in text
    assert "喜歡簡潔的回覆" in text


def test_all_declared_categories_are_accepted(tmp_path):
    path = tmp_path / "memory.json"
    for category in CATEGORIES:
        entry = add_memory(category, f"{category} 測試", path=path)
        assert entry["category"] == category
