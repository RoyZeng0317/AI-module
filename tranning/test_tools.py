"""Tests for tools.route_reply()'s calculus routing (quiz / reveal-last /
free-form solve) and the implicit "do you know about X" web-lookup fallback.

Weather/existing-search routing isn't covered here (those need a real
network call, same reason chats.py/test_chats.py never exercises them
either) — the calculus tests are pure local computation
(calculus_generator.py / calculus_solver.py) and need no network. The new
knowledge-question fallback tests below DO need web_search() to return
something, so they monkeypatch tools.web_search() instead of hitting
DuckDuckGo for real — same reasoning as tools.py's own docstring: the
network call itself isn't sinco's logic, the routing decision is.

route_reply() caches the most recently posed quiz question in a module-level
global (tools._last_calculus_problem) — see tools.py's comment on why a
plain global is enough for this single-user desktop app. Because it's
module-level state shared across tests, every test that depends on it
starting empty resets it explicitly first, rather than relying on test
execution order.
"""

import json

import tools
from tools import _calculus_topic_if_requested, _is_solve_last_request, route_reply


def _reset_last_problem():
    tools._last_calculus_problem = None


def test_generic_calculus_request_detected_as_random_topic():
    assert _calculus_topic_if_requested("出一題微積分") == "random"


def test_specific_subtopic_keywords_detected():
    assert _calculus_topic_if_requested("出一題微分") == "derivative"
    assert _calculus_topic_if_requested("出一題導數") == "derivative"
    assert _calculus_topic_if_requested("出一題積分") == "integral"
    assert _calculus_topic_if_requested("出一題極限") == "limit"
    assert _calculus_topic_if_requested("give me a problem on derivative") == "derivative"
    assert _calculus_topic_if_requested("quiz me on integral") == "integral"


def test_specific_subtopic_inside_generic_phrase_overrides_random():
    # "積分" is a literal substring of "微積分" — the specific mention should
    # still win over treating this as a generic "pick any topic" request.
    assert _calculus_topic_if_requested("出一題微積分裡的積分題") == "integral"


def test_no_request_hint_returns_none_even_with_topic_word():
    assert _calculus_topic_if_requested("我今天要交微積分作業") is None
    assert _calculus_topic_if_requested("微積分") is None


def test_no_topic_word_returns_none_even_with_request_hint():
    assert _calculus_topic_if_requested("考我英文單字") is None


def test_is_solve_last_request_exact_match_only():
    assert _is_solve_last_request("解題")
    assert _is_solve_last_request("解題！")
    assert _is_solve_last_request("Show Answer")
    assert not _is_solve_last_request("幫我解 3x^2 的微分")  # contains "解" but isn't a bare reveal request


def test_route_reply_quiz_request_returns_question_only_and_stores_state():
    _reset_last_problem()
    result = route_reply("出一題微分")
    assert result is not None
    reason, reply = result
    assert "出題請求" in reason
    assert "sinco 微積分出題" in reply
    assert "題目" in reply
    assert tools._last_calculus_problem is not None
    assert tools._last_calculus_problem["topic"] == "derivative"
    # two-step flow: the actual worked answer/steps are not revealed yet
    assert tools._last_calculus_problem["answer"] not in reply
    for step in tools._last_calculus_problem["steps"]:
        assert step not in reply


def test_route_reply_solve_last_reveals_stored_problem():
    _reset_last_problem()
    route_reply("出一題積分")  # pose a question, storing it
    reason, reply = route_reply("解題")
    assert "解題」請求" in reason
    assert "答案" in reply
    assert "詳解" in reply


def test_route_reply_solve_last_without_prior_quiz_returns_friendly_message():
    _reset_last_problem()
    reason, reply = route_reply("解題")
    assert "沒有已出的題目" in reason
    assert "出一題微積分" in reply


def test_route_reply_calculus_quiz_does_not_fall_through_to_search():
    _reset_last_problem()
    reason, _reply = route_reply("出一題微積分")
    assert "搜尋" not in reason


def test_route_reply_plain_calculus_mention_falls_back_to_none():
    _reset_last_problem()
    assert route_reply("我今天要交微積分作業") is None


def test_route_reply_freeform_solve_derivative():
    _reset_last_problem()
    reason, reply = route_reply("3x^2+5x 的微分")
    assert "自由輸入" in reason
    assert "sinco 解題" in reply
    assert "答案" in reply
    assert tools._last_calculus_problem["topic"] == "derivative"


def test_route_reply_freeform_solve_integral():
    _reset_last_problem()
    reason, reply = route_reply("∫ x^2 dx")
    assert "自由輸入" in reason
    assert "積分" in reply


def test_route_reply_freeform_solve_limit():
    _reset_last_problem()
    reason, reply = route_reply("lim(x->0) sin(x)/x")
    assert "自由輸入" in reason
    assert "= 1" in reply


# ---------------------------------------------------------------------------
# 隱含知識詢問句型的網路查詢備援（"你會微積分嗎" 這類，CLAUDE.md to-do）
# ---------------------------------------------------------------------------

def _write_pairs(tmp_path, prompts: list[str]):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([{"prompt": p, "reply": "已訓練過的固定回覆"} for p in prompts], ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_knowledge_question_fallback_fires_for_you_know_x_pattern(tmp_path, monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, ["hi"]))
    saved = {}
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: saved.update(kw))
    monkeypatch.setattr(tools, "web_search", lambda subject: f"{subject} 是數學的一個分支。")

    reason, reply = route_reply("你會微積分嗎")

    assert "微積分" in reason
    assert "微積分 是數學的一個分支。" == reply
    assert saved == {"prompt": "你會微積分嗎", "reply": reply, "topic": "微積分", "source": "duckduckgo"}


def test_knowledge_question_fallback_fires_for_what_is_x_pattern(tmp_path, monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, ["hi"]))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    monkeypatch.setattr(tools, "web_search", lambda subject: f"{subject} 的說明")

    reason, reply = route_reply("什麼是牛頓第二運動定律")

    assert "牛頓第二運動定律" in reason
    assert reply == "牛頓第二運動定律 的說明"


def test_knowledge_question_fallback_skips_messages_matching_trained_prompt(tmp_path, monkeypatch):
    # "你會說英文嗎" 已經是 data/pairs.json 裡固定訓練過的一筆（回覆
    # "Sure, no problem!"），句型跟 "你會 X 嗎" 一模一樣，不該被這裡的規則
    # 搶走去查「說英文」——這是新規則跟既有訓練資料最容易撞在一起的案例。
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, ["你會說英文嗎"]))
    called = []
    monkeypatch.setattr(tools, "web_search", lambda subject: called.append(subject) or "不該被呼叫")

    assert route_reply("你會說英文嗎") is None
    assert called == []


def test_knowledge_question_fallback_returns_none_without_apology_when_nothing_found(tmp_path, monkeypatch):
    # 跟明講要搜尋的 _SEARCH_PATTERNS 不同：這裡是猜測性的，找不到就安靜地
    # 退回一般聊天模型，不回一句「抱歉沒找到」。DuckDuckGo 跟中文維基百科都
    # 要模擬成查無結果，不然 _lookup() 真的會打去 zh.wikipedia.org。
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, []))
    monkeypatch.setattr(tools, "web_search", lambda subject: None)
    monkeypatch.setattr(tools, "wikipedia_search", lambda subject: None)

    assert route_reply("你懂量子力學嗎") is None


def test_lookup_falls_back_to_wikipedia_when_duckduckgo_has_nothing(monkeypatch):
    # 這是修「你會微積分嗎」實際查不到資料那個案例的核心：DuckDuckGo 對中文
    # 詞條常常查無結果（實測「微積分」「台北101」都是），中文維基百科查得到。
    monkeypatch.setattr(tools, "web_search", lambda subject: None)
    monkeypatch.setattr(tools, "wikipedia_search", lambda subject: f"{subject} 是數學的一個分支。")

    result, source = tools._lookup("微積分")

    assert source == "wikipedia"
    assert result == "微積分 是數學的一個分支。"


def test_knowledge_question_fallback_uses_wikipedia_and_records_its_source(tmp_path, monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, ["hi"]))
    monkeypatch.setattr(tools, "web_search", lambda subject: None)
    monkeypatch.setattr(tools, "wikipedia_search", lambda subject: f"{subject} 是數學的一個分支。")
    saved = {}
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: saved.update(kw))

    reason, reply = route_reply("你會微積分嗎")

    assert "中文維基百科" in reason
    assert reply == "微積分 是數學的一個分支。"
    assert saved["source"] == "wikipedia"


def test_knowledge_question_fallback_excludes_self_reference_subject(tmp_path, monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, []))
    called = []
    monkeypatch.setattr(tools, "web_search", lambda subject: called.append(subject) or "不該被呼叫")

    assert route_reply("你懂你嗎") is None
    assert called == []


# ---------------------------------------------------------------------------
# 圖片辨識路由（"辨識圖片 <路徑或網址>"，延伸現有 YOLO 偵測器到任意圖片）
# ---------------------------------------------------------------------------

def test_image_source_detected_for_each_prefix():
    from tools import _image_source_if_requested
    assert _image_source_if_requested("辨識圖片 C:/photos/cat.jpg") == "C:/photos/cat.jpg"
    assert _image_source_if_requested("辨識圖像：https://example.com/a.png") == "https://example.com/a.png"
    assert _image_source_if_requested("識別圖片 a.jpg") == "a.jpg"
    assert _image_source_if_requested("看看這張圖片 a.jpg") == "a.jpg"
    assert _image_source_if_requested("recognize image https://example.com/x.jpg") == \
        "https://example.com/x.jpg"


def test_image_source_returns_none_without_prefix():
    from tools import _image_source_if_requested
    assert _image_source_if_requested("這是一張圖片") is None
    assert _image_source_if_requested("我喜歡圖片") is None


def test_image_source_returns_none_without_path_after_prefix():
    from tools import _image_source_if_requested
    assert _image_source_if_requested("辨識圖片") is None
    assert _image_source_if_requested("辨識圖片   ") is None


def test_route_reply_fires_image_recognition(monkeypatch):
    _reset_last_problem()
    called = []
    monkeypatch.setattr(tools, "recognize_image", lambda source: called.append(source) or "偵測到 1 個物件：貓（信心度 90%）")

    reason, reply = route_reply("辨識圖片 C:/photos/cat.jpg")

    assert "圖片辨識" in reason
    assert reply == "偵測到 1 個物件：貓（信心度 90%）"
    assert called == ["C:/photos/cat.jpg"]


# ---------------------------------------------------------------------------
# 口語連接詞前綴（"那你知道 X 嗎"）不該讓 _SEARCH_PATTERNS/_KNOWLEDGE_QUESTION_
# PATTERNS 的 ^ 錨定失效——實測回報："你知道token嗎" 能正確觸發搜尋，但多了
# 開頭一個「那」字的「那你知道張凌赫嗎」卻比對不到、掉回死記聊天模型亂答。
# ---------------------------------------------------------------------------

def test_strip_leading_filler_removes_common_conversational_prefixes():
    assert tools._strip_leading_filler("那你知道張凌赫嗎") == "你知道張凌赫嗎"
    assert tools._strip_leading_filler("欸那你認識他嗎") == "你認識他嗎"
    assert tools._strip_leading_filler("請問你知道台北101嗎") == "你知道台北101嗎"
    assert tools._strip_leading_filler("你知道token嗎") == "你知道token嗎"


def test_route_reply_search_pattern_fires_with_leading_filler_prefix(monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_lookup", lambda subject: (f"{subject} 是一位演員。", "duckduckgo"))

    reason, reply = route_reply("那你知道張凌赫嗎?")

    assert "張凌赫" in reason
    assert reply == "張凌赫 是一位演員。"


def test_route_reply_knowledge_fallback_fires_with_leading_filler_prefix(tmp_path, monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "_PAIRS_PATH", _write_pairs(tmp_path, ["hi"]))
    monkeypatch.setattr(tools, "web_search", lambda subject: f"{subject} 是數學的一個分支。")

    reason, reply = route_reply("那你懂微積分嗎")

    assert "微積分" in reason
    assert reply == "微積分 是數學的一個分支。"


def test_recognize_image_reports_missing_local_file():
    assert "找不到圖片檔案" in tools.recognize_image("C:/definitely/not/a/real/path/nope.jpg")


def test_recognize_image_reports_download_failure(monkeypatch):
    class _FakeSession:
        def get(self, *a, **kw):
            raise tools.requests.RequestException("boom")

    monkeypatch.setattr(tools.requests, "get", _FakeSession().get)
    assert "下載失敗" in tools.recognize_image("https://example.com/nope.jpg")


# ---------------------------------------------------------------------------
# web_search() 的歧義詞展開（DuckDuckGo 對有歧義的詞回傳空 AbstractText +
# 自己就先截斷成 "..." 的 RelatedTopics 預覽，見 CLAUDE.md — "Tokenization"
# 這個真實案例：AbstractText 是空字串，RelatedTopics 給的是「referred to...」
# 這種斷尾預覽，FirstURL 帶著消歧義後的完整詞條名）
# ---------------------------------------------------------------------------

def test_web_search_returns_abstract_text_directly_when_present(monkeypatch):
    calls = []

    def fake_fetch(query):
        calls.append(query)
        return {"AbstractText": "完整說明。", "RelatedTopics": []}

    monkeypatch.setattr(tools, "_fetch_abstract", fake_fetch)
    assert tools.web_search("已知詞") == "完整說明。"
    assert calls == ["已知詞"]


def test_web_search_expands_truncated_related_topic_via_first_url(monkeypatch):
    calls = []

    def fake_fetch(query):
        calls.append(query)
        if query == "Tokenization":
            return {
                "AbstractText": "",
                "RelatedTopics": [
                    {
                        "FirstURL": "https://duckduckgo.com/Tokenization_(data_security)",
                        "Text": "Tokenization (data security) The process of substituting...",
                    }
                ],
            }
        assert query == "Tokenization (data security)"
        return {"AbstractText": "完整的 Tokenization 說明，沒有被截斷。"}

    monkeypatch.setattr(tools, "_fetch_abstract", fake_fetch)
    result = tools.web_search("Tokenization")
    assert result == "完整的 Tokenization 說明，沒有被截斷。"
    assert calls == ["Tokenization", "Tokenization (data security)"]


def test_web_search_falls_back_to_truncated_text_when_expansion_fails(monkeypatch):
    def fake_fetch(query):
        if query == "Tokenization":
            return {
                "AbstractText": "",
                "RelatedTopics": [
                    {
                        "FirstURL": "https://duckduckgo.com/Tokenization_(data_security)",
                        "Text": "Tokenization (data security) The process of substituting...",
                    }
                ],
            }
        return {"AbstractText": ""}  # 消歧義後重查依然沒有結果

    monkeypatch.setattr(tools, "_fetch_abstract", fake_fetch)
    result = tools.web_search("Tokenization")
    assert result == "Tokenization (data security) The process of substituting..."


def test_web_search_returns_related_topic_text_unchanged_when_not_truncated(monkeypatch):
    def fake_fetch(query):
        return {
            "AbstractText": "",
            "RelatedTopics": [{"FirstURL": "https://duckduckgo.com/Foo", "Text": "完整的一句話，沒有省略號"}],
        }

    monkeypatch.setattr(tools, "_fetch_abstract", fake_fetch)
    assert tools.web_search("foo") == "完整的一句話，沒有省略號"


# ---------------------------------------------------------------------------
# 影片搜尋（"找...的影片"/"搜尋影片 X"/"search video X"）——實測 YouTube 搜尋
# 結果頁沒有像百度百科那樣被反爬蟲擋下來，直接解析內嵌的 ytInitialData JSON。
# 這裡的測試不打真實網路（同 web_search 的測試慣例），用假的 HTML/JSON 驗證
# 解析邏輯本身。
# ---------------------------------------------------------------------------

def test_video_query_detected_with_prefix_and_suffix_phrasing():
    assert tools._video_query_if_requested("找台灣夜市的影片") == "台灣夜市"
    assert tools._video_query_if_requested("搜尋影片 貓咪") == "貓咪"
    assert tools._video_query_if_requested("search video python tutorial") == "python tutorial"


def test_video_query_returns_none_without_both_hint_and_keyword():
    # 只有「影片/video」關鍵字，沒有找/搜尋類請求動詞 -> 不該誤觸發
    assert tools._video_query_if_requested("這部影片很好看") is None
    # 只有請求動詞，沒有影片/視頻/video/youtube 關鍵字 -> 不該誤觸發
    assert tools._video_query_if_requested("幫我找一下資料") is None
    # "電影" 不包含 "影片" 這個完整關鍵字，不該誤觸發
    assert tools._video_query_if_requested("我剛看了一部很好看的電影") is None


def _fake_youtube_html(videos: list[dict]) -> str:
    data = {"contents": {"videoRenderers": [{"videoRenderer": v} for v in videos]}}
    return f"<html><script>var ytInitialData = {json.dumps(data)};</script></html>"


def test_youtube_search_formats_title_channel_duration_views_and_link(monkeypatch):
    video = {
        "videoId": "abc123",
        "title": {"runs": [{"text": "測試影片標題"}]},
        "ownerText": {"runs": [{"text": "測試頻道"}]},
        "lengthText": {"simpleText": "10:30"},
        "viewCountText": {"simpleText": "觀看次數：1,000次"},
        "detailedMetadataSnippets": [{"snippetText": {"runs": [{"text": "這是"}, {"text": "重點摘要"}]}}],
    }

    class _FakeResp:
        status_code = 200
        text = _fake_youtube_html([video])

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _FakeResp())

    result = tools.youtube_search("測試查詢")

    assert "測試影片標題" in result
    assert "測試頻道" in result
    assert "10:30" in result
    assert "1,000次" in result
    assert "https://www.youtube.com/watch?v=abc123" in result
    assert "這是重點摘要" in result


def test_youtube_search_returns_none_when_no_videos_found(monkeypatch):
    class _FakeResp:
        status_code = 200
        text = _fake_youtube_html([])

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _FakeResp())
    assert tools.youtube_search("查無結果的查詢") is None


def test_youtube_search_returns_none_on_request_failure(monkeypatch):
    def raise_error(*a, **kw):
        raise tools.requests.RequestException("boom")

    monkeypatch.setattr(tools.requests, "get", raise_error)
    assert tools.youtube_search("任何查詢") is None


def test_route_reply_fires_video_search(monkeypatch):
    _reset_last_problem()
    called = []
    monkeypatch.setattr(tools, "youtube_search", lambda query, **kw: called.append(query) or "《影片》\n連結：...")

    reason, reply = route_reply("找貓咪的影片")

    assert "貓咪" in reason
    assert reply == "《影片》\n連結：..."
    assert called == ["貓咪"]


def test_route_reply_video_search_reports_no_results_without_apology_confusion(monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "youtube_search", lambda query, **kw: None)

    reason, reply = route_reply("搜尋影片 xyzxyzxyz不存在的東西")

    assert "影片搜尋" in reason
    assert "沒有找到" in reply


# ---------------------------------------------------------------------------
# 台股加權指數（TAIEX）—— 資料源是 TWSE 自己的 mis.twse.com.tw 即時行情 API，
# 回應格式是真實 API 打過一次拿到的樣本（見 tools.py get_taiex() 註解）。
# ---------------------------------------------------------------------------

def _fake_taiex_response(z="39933.30", y="40039.18", d="20260730", n="發行量加權股價指數"):
    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"msgArray": [{"z": z, "y": y, "d": d, "n": n}]}

    return _FakeResp()


def test_get_taiex_formats_index_change_and_percentage(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _fake_taiex_response())

    result = tools.get_taiex()

    assert "發行量加權股價指數" in result
    assert "2026/07/30" in result
    assert "39933.30" in result
    assert "-105.88" in result  # 39933.30 - 40039.18
    assert "-0.26%" in result


def test_get_taiex_formats_positive_change_with_leading_plus_sign(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _fake_taiex_response(z="40100.00", y="40000.00"))

    result = tools.get_taiex()

    assert "+100.00" in result
    assert "+0.25%" in result


def test_get_taiex_raises_when_no_rows_returned(monkeypatch):
    class _EmptyResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"msgArray": []}

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _EmptyResp())

    try:
        tools.get_taiex()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_taiex_raises_when_price_fields_are_not_numeric(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: _fake_taiex_response(z="-", y="-"))

    try:
        tools.get_taiex()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_route_reply_fires_stock_query(monkeypatch):
    _reset_last_problem()
    monkeypatch.setattr(tools, "get_taiex", lambda: "發行量加權股價指數（2026/07/30）：39933.30 點，較前一交易日 -105.88 點（-0.26%）")

    reason, reply = route_reply("台股表現怎麼樣?")

    assert "台股" in reason
    assert "39933.30" in reply


def test_route_reply_stock_query_reports_failure_without_crashing(monkeypatch):
    _reset_last_problem()

    def raise_error():
        raise tools.requests.RequestException("boom")

    monkeypatch.setattr(tools, "get_taiex", raise_error)

    reason, reply = route_reply("大盤現在幾點?")

    assert "大盤" in reason
    assert "暫時失敗" in reply
