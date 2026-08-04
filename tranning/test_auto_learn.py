"""Tests for auto_learn.py — the review queue between tools.py's live web
lookups and data/pairs.json. Every test uses its own tmp_path files so runs
don't touch the project's real candidate/pairs files.
"""

import auto_learn


def test_save_candidate_creates_entry_with_expected_fields(tmp_path):
    path = tmp_path / "candidates.json"
    entry = auto_learn.save_candidate("你會微積分嗎", "微積分是數學的一個分支...", "微積分", path=path)

    assert entry["prompt"] == "你會微積分嗎"
    assert entry["reply"] == "微積分是數學的一個分支..."
    assert entry["topic"] == "微積分"
    assert entry["source"] == "duckduckgo"
    assert "id" in entry and "created_at" in entry

    assert auto_learn.list_candidates(path=path) == [entry]


def test_save_candidate_dedupes_by_topic(tmp_path):
    path = tmp_path / "candidates.json"
    first = auto_learn.save_candidate("你會微積分嗎", "答案A", "微積分", path=path)
    second = auto_learn.save_candidate("什麼是微積分", "答案B", "微積分", path=path)

    assert first == second
    assert len(auto_learn.list_candidates(path=path)) == 1


def test_approve_candidate_moves_it_into_pairs_and_removes_from_queue(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text('[{"prompt": "hi", "reply": "hi there"}]', encoding="utf-8")

    entry = auto_learn.save_candidate("你會微積分嗎", "微積分是...", "微積分", path=candidates_path)
    ok = auto_learn.approve_candidate(entry["id"], candidates_path=candidates_path, pairs_path=pairs_path)

    assert ok is True
    assert auto_learn.list_candidates(path=candidates_path) == []

    import json
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    assert {"prompt": "你會微積分嗎", "reply": "微積分是..."} in pairs
    assert {"prompt": "hi", "reply": "hi there"} in pairs  # 原本就有的資料沒被動到


def test_approve_candidate_returns_false_for_unknown_id(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    pairs_path = tmp_path / "pairs.json"
    assert auto_learn.approve_candidate("not-a-real-id", candidates_path=candidates_path,
                                          pairs_path=pairs_path) is False


def test_reject_candidate_removes_matching_id_only(tmp_path):
    path = tmp_path / "candidates.json"
    keep = auto_learn.save_candidate("A?", "reply A", "topicA", path=path)
    remove = auto_learn.save_candidate("B?", "reply B", "topicB", path=path)

    assert auto_learn.reject_candidate(remove["id"], path=path) is True
    assert auto_learn.list_candidates(path=path) == [keep]


def test_reject_candidate_returns_false_for_unknown_id(tmp_path):
    path = tmp_path / "candidates.json"
    auto_learn.save_candidate("A?", "reply A", "topicA", path=path)
    assert auto_learn.reject_candidate("not-a-real-id", path=path) is False


def test_format_candidates_empty_and_nonempty(tmp_path):
    assert "沒有待審核" in auto_learn.format_candidates([])

    path = tmp_path / "candidates.json"
    entry = auto_learn.save_candidate("你會微積分嗎", "微積分是...", "微積分", path=path)
    text = auto_learn.format_candidates([entry])
    assert entry["id"] in text
    assert "你會微積分嗎" in text
    assert "微積分是..." in text
