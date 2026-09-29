"""Unit tests for news_keyword_import.py — 用合成假新聞文字驗證「讀資料夾 ->
jieba+TFIDF 抽關鍵字 -> chats.py 用的 prompt/reply pairs」這條管線本身能跑完，
不代表真實新聞的關鍵字品質（跟專案其餘 test_*.py 慣例一致）。
"""

import json

from news_keyword_import import (
    build_pairs,
    extract_keywords_per_doc,
    load_stopwords,
    read_news_folder,
)


def test_load_stopwords_reads_lines_and_skips_blank(tmp_path):
    path = tmp_path / "stopwords.txt"
    path.write_text("的\n了\n\n是\n", encoding="utf-8")
    assert load_stopwords(path) == {"的", "了", "是"}


def test_load_stopwords_missing_file_returns_empty_set(tmp_path):
    assert load_stopwords(tmp_path / "missing.txt") == set()


def test_read_news_folder_sorted_by_filename(tmp_path):
    (tmp_path / "b.txt").write_text("台灣的半導體產業持續成長", encoding="utf-8")
    (tmp_path / "a.txt").write_text("中央銀行宣布升息半碼", encoding="utf-8")
    articles, filenames = read_news_folder(tmp_path)
    assert filenames == ["a.txt", "b.txt"]
    assert articles[0].startswith("中央銀行")


def test_extract_keywords_per_doc_ranks_distinctive_words_higher(tmp_path):
    articles = [
        "台灣半導體產業今年持續成長，晶片出口創下新高紀錄",
        "中央銀行宣布升息半碼，房貸利率預期同步上升",
        "台灣半導體今年出口動能強勁，廠商樂觀看待後市",
    ]
    keywords = extract_keywords_per_doc(articles, stopwords=set(), topk=3)
    assert len(keywords) == 3
    for doc_keywords in keywords:
        assert 0 < len(doc_keywords) <= 3
        weights = [w for _, w in doc_keywords]
        assert weights == sorted(weights, reverse=True)
    bank_doc_words = {w for w, _ in keywords[1]}
    assert "升息" in bank_doc_words or "央行" in bank_doc_words or "利率" in bank_doc_words


def test_build_pairs_creates_prompt_reply_and_skips_empty_keyword_docs():
    articles = ["台灣半導體出口成長", ""]
    keywords_per_doc = [[("半導體", 0.8), ("出口", 0.6)], []]
    pairs = build_pairs(articles, keywords_per_doc)
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "這篇新聞的關鍵字是什麼：\n台灣半導體出口成長"
    assert pairs[0]["reply"] == "半導體、出口"


def test_end_to_end_writes_valid_pairs_json(tmp_path):
    source = tmp_path / "news"
    source.mkdir()
    (source / "a.txt").write_text("台灣半導體產業出口創新高", encoding="utf-8")
    (source / "b.txt").write_text("中央銀行宣布升息半碼影響房貸族", encoding="utf-8")

    articles, _ = read_news_folder(source)
    keywords_per_doc = extract_keywords_per_doc(articles, stopwords=set(), topk=3)
    pairs = build_pairs(articles, keywords_per_doc)

    out = tmp_path / "pairs_news_keywords_draft.json"
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert all("prompt" in e and "reply" in e for e in loaded)
