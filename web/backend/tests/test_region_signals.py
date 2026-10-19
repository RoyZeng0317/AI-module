"""規則契約與反例；不宣稱所在地分類準確度。"""
import ast
from pathlib import Path
from unittest.mock import patch

import pytest
from web.backend import region_signals as signals


@pytest.mark.parametrize("timezone,country,status", [
    ("Asia/Shanghai", "CN", "cn_timezone_and_ip"),
    ("Asia/Shanghai", "MY", "malaysia_timezone_conflict"),
    ("Asia/Kuala_Lumpur", "MY", "malaysia_ip"),
    ("Asia/Taipei", "TW", "other_country_ip"),
    ("Asia/Singapore", "SG", "other_country_ip"),
    ("UTC+8", None, "unknown"),
    (None, "CN", "cn_ip_only"),
    ("Asia/Shanghai", None, "beijing_timezone_only"),
    ("Asia/Shanghai", "TW", "other_country_timezone_conflict"),
])
def test_separate_signals(timezone, country, status):
    with patch.object(signals, "lookup_country", return_value=country):
        result = signals.assess_region("这句话，那咋啦", timezone, "unused")
    assert result["status"] == status
    assert result["ip_country"] == country
    assert result["action"] == "observe_only"


@pytest.mark.parametrize("message,key", [("这", "simplified_sample"), ("，", "punctuation"), ("？", "punctuation"), ("、", "punctuation"), ("。", "punctuation"), ("那咋啦", "phrases")])
def test_text_alone_never_confirms_region(message, key):
    with patch.object(signals, "lookup_country", return_value=None):
        result = signals.assess_region(message)
    assert result["status"] == "unknown"
    assert result["text_signals"][key]


def test_traditional_shared_characters_not_simplified_marker():
    with patch.object(signals, "lookup_country", return_value="TW"):
        result = signals.assess_region("皇后走了三公里，詩云。", "Asia/Taipei")
    assert not result["text_signals"]["simplified_sample"]
    assert "U+FF0C" in result["text_signals"]["punctuation"]
    assert result["status"] == "other_country_ip"


@pytest.mark.parametrize("timezone,country,status", [
    (None, None, "unknown"),
    ("Asia/Taipei", "TW", "other_country_ip"),
    ("Asia/Kuala_Lumpur", "MY", "malaysia_ip"),
    ("Asia/Shanghai", "MY", "malaysia_timezone_conflict"),
])
def test_punctuation_is_one_weak_category_without_repetition_weight(timezone, country, status):
    with patch.object(signals, "lookup_country", return_value=country):
        once = signals.assess_region("，？、。", timezone)
        repeated = signals.assess_region("，？、。" * 100, timezone)
    assert once == repeated
    assert once["status"] == status
    assert once["text_signals"]["punctuation"] == ["U+3001", "U+3002", "U+FF0C", "U+FF1F"]
    assert once["action"] == "observe_only"


def test_ascii_punctuation_is_not_a_configured_marker():
    with patch.object(signals, "lookup_country", return_value=None):
        result = signals.assess_region(",?.")
    assert result["text_signals"]["punctuation"] == []
    assert result["status"] == "unknown"


@pytest.mark.parametrize("address", [None, "bad", "127.0.0.1", "192.168.1.1", "::1", "2001:db8::1"])
def test_invalid_private_ip_unknown(address):
    assert signals.lookup_country(address) is None


def test_missing_and_corrupt_database_unknown(tmp_path):
    assert signals.lookup_country("8.8.8.8", tmp_path / "missing") is None
    broken = tmp_path / "broken.mmdb"
    broken.write_bytes(b"not a database")
    assert signals.lookup_country("8.8.8.8", broken) is None


def test_public_ipv4_and_ipv6_lookup_boundary(tmp_path, monkeypatch):
    path = tmp_path / "db"
    path.touch()
    class Reader:
        def get(self, ip):
            assert ip in ("8.8.8.8", "2606:4700:4700::1111")
            return {"country": {"iso_code": "MY"}}
    monkeypatch.setattr(signals, "_reader", lambda *args: Reader())
    for ip in ("8.8.8.8", "2606:4700:4700::1111"):
        assert signals.lookup_country(ip, path) == "MY"


def test_browser_timezone_failure_and_success():
    source = Path("web/admin/frontend/src/components/action.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_client_timezone")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "action.py", "exec"), namespace)
    assert namespace["_client_timezone"]() is None  # Intl absent: chat can continue.
    from types import SimpleNamespace
    formatter = SimpleNamespace(resolvedOptions=lambda: SimpleNamespace(timeZone="Asia/Kuala_Lumpur"))
    namespace["window"] = SimpleNamespace(Intl=SimpleNamespace(DateTimeFormat=SimpleNamespace(new=lambda: formatter)))
    assert namespace["_client_timezone"]() == "Asia/Kuala_Lumpur"


def test_chat_keeps_assessment_internal_and_ignores_spoofed_ip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from web.backend import app as module
    monkeypatch.setattr(module.usage_store, "USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(module.session_store, "SESSION_PATH", tmp_path / "session.json")
    module._chat_request_log.clear()
    module.app.state.region_assessments.clear()
    with patch.object(module, "smart_reply_traced", return_value=("trace", "hello")), patch.object(module.region_signals, "lookup_country", return_value="MY") as lookup:
        response = TestClient(module.app).post("/api/chat", json={"message": "这，那咋啦", "client_timezone": "Asia/Shanghai", "ip": "114.114.114.114", "country": "CN"}, headers={"X-Forwarded-For": "114.114.114.114"})
    assert response.status_code == 200
    assert set(response.json()) == {"reply", "trace", "conversation_id"}
    lookup.assert_called_once_with("testclient")
    recent = module.app.state.region_assessments
    assert recent[-1]["status"] == "malaysia_timezone_conflict"
    assert "message" not in recent[-1] and "ip" not in recent[-1]
    assert recent.maxlen == 200
    with patch.object(module, "smart_reply_traced", return_value=("trace", "hello")):
        response = TestClient(module.app).post("/api/chat", json={"message": "hi", "client_timezone": "x" * 129})
    assert response.status_code == 422
