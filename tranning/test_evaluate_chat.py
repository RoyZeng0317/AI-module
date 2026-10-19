import json
from types import SimpleNamespace
import pytest
from evaluate_chat import main, evaluate_cases


def test_offline_disables_router_and_skips_network():
    calls = []
    engine = SimpleNamespace(route_reply=lambda message: pytest.fail("network router called"))
    def reply(message, **kwargs):
        calls.append(kwargs)
        assert engine.route_reply(message) is None
        return "fake model", "review me"
    engine.smart_reply_traced = reply
    records = list(evaluate_cases([
        dict(id="a", prompt="hi", history=[["earlier", "answer"]]),
        dict(id="b", prompt="news", requires_network=True),
    ], engine))
    assert records[0]["status"] == "recorded"
    assert records[0]["semantic_score"] is None
    assert records[0]["latency_ms"] >= 0
    assert calls == [dict(force_mode="auto", history=[["earlier", "answer"]])]
    assert records[1]["status"] == "skipped_network"


def test_cli_requires_run_and_writes_review_records(tmp_path):
    cases, output = tmp_path / "cases.json", tmp_path / "out.jsonl"
    cases.write_text(json.dumps([dict(id="a", prompt="hello", language="en", expected_intent="greeting", review_criteria=["relevant"])]))
    args = ["--cases", str(cases), "--output", str(output)]
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()
    engine = SimpleNamespace(route_reply=lambda message: None, smart_reply_traced=lambda *args, **kwargs: ("fake", "Hello"))
    main(args + ["--run"], engine=engine)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["reply"] == "Hello"
    assert record["semantic_review"] == "pending_human_review"
    assert record["tools_enabled"] is False


def test_network_only_enabled_explicitly():
    engine = SimpleNamespace(route_reply=lambda message: "network", smart_reply_traced=lambda *args, **kwargs: ("tool", "source"))
    record = next(evaluate_cases([dict(prompt="news", requires_network=True)], engine, allow_network=True))
    assert record["status"] == "recorded" and record["tools_enabled"] is True
