"""圖片搜尋路由、資料邊界、YOLO 目標篩選與誠實候選狀態。"""
import html, io, json
from types import SimpleNamespace

import pytest, requests
from PIL import Image
import image_search as search
import tools


@pytest.mark.parametrize("message,query", [
    ("你尋找有關周柯宇的圖片", "周柯宇"), ("幫我搜尋周柯宇的照片", "周柯宇"),
    ("搜尋圖片：周柯宇", "周柯宇"), ("請幫我找貓的圖片", "貓"),
    ("find images of Daniel Zhou", "Daniel Zhou"), ("Can you find cat photos?", "cat"),
    ("search for pictures of cats", "cats"),
])
def test_explicit_query(message, query):
    assert search.image_query(message) == query


@pytest.mark.parametrize("message", ["圖片是什麼", "不要搜尋圖片", "不要搜尋周柯宇的照片", "我喜歡周柯宇的圖片", "辨識圖片 x.png", "search images", "find photos", "照片"])
def test_not_an_image_search(message):
    assert search.image_query(message) is None


@pytest.mark.parametrize("message,lang", [("你尋找有關周柯宇的圖片", "zh"), ("Can you find cat photos?", "en")])
def test_shared_route_before_chat_and_capability_guard(monkeypatch, message, lang):
    calls = []
    monkeypatch.setattr(tools, "search_images", lambda query, **kw: calls.append((query, kw)) or "image reply")
    reason, reply = tools.route_reply(message)
    assert reply == "image reply"
    assert ("Image search" if lang == "en" else "圖片搜尋") in reason
    if lang == "en":
        assert not tools._CJK_RE.search(reason)
    assert calls[0][1] == {"lang": lang}


def test_bing_parser_requires_real_direct_image_and_source():
    item = {"murl": "https://image.example/a.jpg", "purl": "https://source.example/a", "t": "周柯宇"}
    anchor = f'<a class="iusc" m="{html.escape(json.dumps(item))}"></a>'
    bad = '<a class="iusc" m="bad"></a><a class="iusc" m="[]"></a>'
    assert search.parse_candidates(anchor + anchor + bad) == [{"image_url": item["murl"], "source": item["purl"], "title": item["t"]}]
    assert search.parse_candidates('<img src="logo.jpg">') == []


def test_360_provider_filters_deduplicates_and_strips_title_markup(monkeypatch):
    data = {"list": [{"img": "https://img.example/a.jpg", "link": "https://source.example", "title": "<b>周柯宇</b>"},
                     {"img": "https://img.example/a.jpg", "link": "https://source.example"},
                     {"img": "file:///secret", "link": "https://source.example"}, None]}
    monkeypatch.setattr(search, "_fetch", lambda *a, **kw: json.dumps(data).encode())
    assert search.search_candidates("周柯宇") == [{"image_url": "https://img.example/a.jpg", "source": "https://source.example", "title": "周柯宇"}]


def test_provider_failure_can_use_bing_fixture(monkeypatch):
    item = {"murl": "https://img.example/a.jpg", "purl": "https://source.example/a", "t": "cat"}
    def fetch(url, *a, **kw):
        if "so.com" in url:
            raise requests.Timeout()
        return f'<a class="iusc" m="{html.escape(json.dumps(item))}"></a>'.encode()
    monkeypatch.setattr(search, "_fetch", fetch)
    assert search.search_candidates("cat")[0]["title"] == "cat"


class Response:
    def __init__(self, chunks=(b"hello",), headers=None):
        self.headers = headers or {"Content-Type": "image/png"}
        self.chunks = chunks
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, size): return iter(self.chunks)


@pytest.mark.parametrize("headers,chunks", [
    ({"Content-Type": "text/html"}, [b"x"]),
    ({"Content-Type": "image/png", "Content-Length": "100"}, [b"x"]),
    ({"Content-Type": "image/png"}, [b"12345", b"67890"]),
])
def test_stream_limits_and_content_type(monkeypatch, headers, chunks):
    monkeypatch.setattr(search.requests, "get", lambda *a, **kw: Response(chunks, headers))
    with pytest.raises(ValueError):
        search._fetch("https://example.com/image", 8, image=True)


def test_download_checks_pixels_before_conversion(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="PNG")
    monkeypatch.setattr(search, "_fetch", lambda *a, **kw: buffer.getvalue())
    monkeypatch.setattr(search, "MAX_PIXELS", 20)
    with pytest.raises(ValueError, match="pixel"):
        search.download_image("https://example.com/image.png")


def test_corrupt_image_rejected(monkeypatch):
    monkeypatch.setattr(search, "_fetch", lambda *a, **kw: b"not-an-image")
    with pytest.raises(OSError):
        search.download_image("https://example.com/image.png")


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "CACHE_DIR", tmp_path)
    candidates = [{"image_url": f"https://img.example/{i}.jpg", "source": f"https://source.example/{i}", "title": "周柯宇"} for i in range(4)]
    monkeypatch.setattr(search, "search_candidates", lambda query: candidates)
    monkeypatch.setattr(search, "character_model", lambda: SimpleNamespace(names={0: "周柯宇", 1: "其他人物"}))
    monkeypatch.setattr(search, "download_image", lambda url: Image.new("RGB", (10, 10)))
    return tmp_path, candidates


def test_only_matching_predictions_become_confirmed_attachments(monkeypatch, pipeline):
    directory, candidates = pipeline
    scores = iter([None, 0.72, None, 0.60])
    monkeypatch.setattr(search, "predict_target", lambda *a: next(scores))
    reply = search.search_images("周柯宇")
    assert reply.count("![") == 2 and "YOLO 72.0%" in reply and "YOLO 60.0%" in reply
    assert "不代表身份驗證" in reply and "未確認" not in reply
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
    assert {row["image_url"] for row in metadata} == {candidates[1]["image_url"], candidates[3]["image_url"]}
    assert all(row["state"] == "yolo" for row in metadata)


def test_no_match_still_shows_explicitly_unconfirmed_search_candidates(monkeypatch, pipeline):
    directory, _ = pipeline
    monkeypatch.setattr(search, "predict_target", lambda *a: None)
    reply = search.search_images("周柯宇")
    assert "匹配 0 張" in reply and "未經人物模型確認" in reply and reply.count("![") == 3
    assert all(json.loads(p.read_text(encoding="utf-8"))["score"] is None for p in directory.glob("*.json"))


def test_unknown_target_never_calls_identity_inference(monkeypatch, pipeline):
    monkeypatch.setattr(search, "predict_target", lambda *a: pytest.fail("unsupported target"))
    reply = search.search_images("cats", lang="en")
    assert "no class" in reply and "not checked" in reply and reply.count("![") == 3


def test_missing_model_is_distinct_from_unsupported_query(monkeypatch, pipeline):
    def unavailable(): raise ValueError("missing")
    monkeypatch.setattr(search, "character_model", unavailable)
    reply = search.search_images("周柯宇")
    assert "人物模型目前無法使用" in reply and "未經人物模型確認" in reply


def test_prediction_failure_not_reported_as_zero_matches(monkeypatch, pipeline):
    def fail(*a): raise RuntimeError("bad inference")
    monkeypatch.setattr(search, "predict_target", fail)
    reply = search.search_images("周柯宇")
    assert "未能完成辨識" in reply and "匹配 0 張" not in reply


def test_all_downloads_fail_has_no_fake_attachments(monkeypatch, pipeline):
    def fail(url): raise requests.Timeout()
    monkeypatch.setattr(search, "download_image", fail)
    reply = search.search_images("周柯宇")
    assert "無法下載" in reply and "![" not in reply


def test_search_failure_returns_readable_search_link(monkeypatch):
    monkeypatch.setattr(search, "search_candidates", lambda q: [])
    reply = search.search_images("周柯宇")
    assert "暫時無法取得結果" in reply and "q=周柯宇" in reply and "![" not in reply


def test_real_target_filter_uses_class_not_person_presence():
    box1, box2 = SimpleNamespace(cls=[1], conf=[0.99]), SimpleNamespace(cls=[0], conf=[0.51])
    model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(names={0: "周柯宇", 1: "其他人物"}, boxes=[box1, box2])])
    assert search.predict_target(model, Image.new("RGB", (10, 10)), "周柯宇") == 0.51


def test_missing_weight_never_downloads_model(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "WEIGHTS", tmp_path / "missing.pt")
    search.character_model.cache_clear()
    with pytest.raises(ValueError, match="missing"):
        search.character_model()
