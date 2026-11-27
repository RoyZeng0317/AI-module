"""Offline search intent, source, and provider failure regression tests."""
import pytest, requests
import tools


class Response:
    status_code = 200
    def __init__(self, text="", data=None):
        self.text, self.content, self.data = text, text.encode(), data
    def raise_for_status(self):
        pass
    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


def test_capability_and_typo_do_not_fetch(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: pytest.fail("unexpected network"))
    assert tools.route_reply("你懂得股票投資嗎?") is None
    assert tools.route_reply("你會查天氣嗎？") is None
    assert "交叉熵" in tools.route_reply("什麼是交叉商損失?")[1]
    typo_reply = tools.route_reply("什麼是交叉商損失?")[1]
    assert "不能確定" in typo_reply and "若是" in typo_reply
    assert "−log(p)" in typo_reply and "https://docs.pytorch.org/" in typo_reply
    assert "我再" not in typo_reply


def test_rag_explanation_context_and_expanded_subject(monkeypatch):
    topics = []
    monkeypatch.setattr(tools, "_is_known_trained_prompt", lambda m: False)
    monkeypatch.setattr(tools, "_lookup", lambda q, **kw: (topics.append(q) or "source text", "duckduckgo"))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    reply = tools.route_reply("名詞解釋什麼是RAG")[1]
    assert "AI" in reply and "其他縮寫" in reply
    assert not topics  # exact reviewed definition needs no live lookup
    tools.route_reply("什麼是遊戲中的RAG")
    assert topics[0] == "遊戲中的RAG"


def test_explicit_fresh_search_never_uses_encyclopedia(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda q: pytest.fail("stale abstract"))
    monkeypatch.setattr(tools, "wikipedia_search", lambda *a, **kw: pytest.fail("encyclopedia"))
    monkeypatch.setattr(tools, "general_web_search", lambda q, **kw: None)
    assert tools._lookup("latest news", fresh=True) == (None, "general_search")


def test_no_key_public_html_sources(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    html = '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Title</a><span class="result__snippet">Snippet</span></div>'
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: Response(html))
    result = tools.general_web_search("Title")
    assert "Snippet" in result and "https://example.com/" in result.splitlines()


def test_challenge_falls_back_to_rss(monkeypatch):
    responses = iter([Response('<form id="challenge-form"></form>'), Response('<rss><channel><item><title>Official</title><link>https://example.org/</link><description>Text</description></item></channel></rss>')])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: next(responses))
    assert "https://example.org/" in tools.public_web_search("Official")


def test_all_public_providers_timeout_returns_none(monkeypatch):
    def fail(*a, **kw):
        raise requests.Timeout()
    monkeypatch.setattr(tools.requests, "get", fail)
    assert tools.public_web_search("q") is None


@pytest.mark.parametrize("data", [[], 42, {"organic": [42, {"title": [], "snippet": {}, "link": 3}]}])
def test_malformed_serper_json_is_not_answer(monkeypatch, data):
    monkeypatch.setenv("SERPER_API_KEY", "test")
    monkeypatch.setattr(tools.requests, "post", lambda *a, **kw: Response(data=data))
    assert tools.general_web_search("q") is None


def test_malformed_knowledge_providers_do_not_escape_or_fall_through(monkeypatch):
    monkeypatch.setattr(tools, "_is_known_trained_prompt", lambda m: False)
    monkeypatch.setattr(tools, "web_search", lambda q: (_ for _ in ()).throw(ValueError("bad JSON")))
    monkeypatch.setattr(tools, "wikipedia_search", lambda *a, **kw: None)
    monkeypatch.setattr(tools, "general_web_search", lambda *a, **kw: None)
    reply = tools.route_reply("什麼是非典型未知名詞")[1]
    assert "可靠網路資料" in reply


def test_english_questions_and_search_language(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_lookup", lambda q, **kw: (seen.append((q, kw)) or "An English source summary.", "duckduckgo"))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    assert tools.route_reply("Can you explain stock investing?") is None
    for question in ("Explain RAG", "What is cross entropy loss?"):
        reply = tools.route_reply(question)[1]
        assert not tools._CJK_RE.search(reply)
    assert all(kwargs["lang"] == "en" for _, kwargs in seen)


def test_explicit_stock_news_uses_search_not_index(monkeypatch):
    monkeypatch.setattr(tools, "get_taiex", lambda: pytest.fail("not an index quote"))
    seen = []
    monkeypatch.setattr(tools, "_lookup", lambda q, **kw: (seen.append((q, kw)) or "News source", "general_search"))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    assert "News source" in tools.route_reply("上網搜尋台股最新新聞")[1]
    assert seen[0][1]["fresh"] is True


@pytest.mark.parametrize("question", ["台股的意思是什麼？", "天氣的意思是什麼？", "什麼是台股？", "什麼是天氣？", "天氣是什麼？"])
def test_definition_intent_precedes_live_tools(monkeypatch, question):
    monkeypatch.setattr(tools, "get_taiex", lambda: pytest.fail("definition is not a quote"))
    monkeypatch.setattr(tools, "get_weather", lambda *a, **kw: pytest.fail("definition is not weather"))
    monkeypatch.setattr(tools, "_is_known_trained_prompt", lambda m: False)
    monkeypatch.setattr(tools, "_lookup", lambda q, **kw: ("Definition source", "duckduckgo"))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    assert tools.route_reply(question)[1] == "Definition source"


def test_polite_english_search_executes_not_capability(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_lookup", lambda q, **kw: (seen.append((q, kw)) or "News results", "general_search"))
    monkeypatch.setattr(tools.auto_learn, "save_candidate", lambda **kw: None)
    assert tools.route_reply("Can you search the latest Taiwan news?")[1] == "News results"
    assert seen[0][1] == {"lang": "en", "fresh": True}
    assert "Taiwan news" in seen[0][0]
    assert tools.route_reply("Can you explain stock investing?") is None


def test_current_weather_and_stock_still_use_live_tools(monkeypatch):
    monkeypatch.setattr(tools, "get_taiex", lambda: "Current index")
    monkeypatch.setattr(tools, "get_weather", lambda place, **kw: "Current weather")
    monkeypatch.setattr(tools, "_lookup", lambda *a, **kw: pytest.fail("live data is not a definition"))
    assert tools.route_reply("台股今天行情")[1] == "Current index"
    assert "Current weather" in tools.route_reply("台北現在天氣")[1]


@pytest.mark.parametrize("query", ["RAG retrieval augmented generation 檢索增強生成", "PyTorch CrossEntropyLoss documentation"])
def test_unrelated_public_search_snippets_are_rejected(monkeypatch, query):
    html = '<div class="result"><a class="result__a" href="https://example.org/person">政治人物</a><span class="result__snippet">沈伯洋 Facebook 貼文</span></div>'
    rss = '<rss><channel><item><title>Political profile</title><link>https://example.org/person</link><description>A person biography</description></item></channel></rss>'
    responses = iter([Response(html), Response(rss)])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: next(responses))
    assert tools.public_web_search(query) is None


def test_rag_lookup_rejects_unrelated_abstracts(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda q: "沈伯洋 biography https://example.org/person")
    monkeypatch.setattr(tools, "wikipedia_search", lambda *a, **kw: "Political profile")
    monkeypatch.setattr(tools, "general_web_search", lambda *a, **kw: None)
    assert tools._lookup("RAG retrieval augmented generation 檢索增強生成")[0] is None


def test_exact_reviewed_terms_are_local_and_other_rag_context_is_not(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: pytest.fail("local definitions need no network"))
    for question in ["名詞解釋什麼是RAG", "什麼是交叉熵損失?", "Explain RAG", "What is cross entropy loss?"]:
        reason, reply = tools.route_reply(question)
        assert "本機已審核詞條" in reason and "2026-10-18" in reason
        assert "https://" in reply
    assert tools._technical_definition("什麼是遊戲中的RAG", "zh") is None
