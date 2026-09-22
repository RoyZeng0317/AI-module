"""HTTP-level tests for app.py — catches wiring bugs that a direct call to
detector.detect() wouldn't (routing, static file exposure, query params).
"""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.backend import app as app_module
from web.backend.app import app

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_public_assets_served():
    # script.js was the planned Firebase-Hosting variant of the frontend
    # (見 前端部屬scp.md) — it was never actually committed to this repo, so
    # asserting on it here was testing a file that doesn't exist. The real
    # PyScript frontend (index.html/style.css/action.py) lives under
    # frontend/src/components/ and is what FRONTEND_DIR now points at.
    for name in ("style.css", "action.py"):
        r = client.get(f"/{name}")
        assert r.status_code == 200


def test_public_assets_are_not_browser_cached():
    # 手機瀏覽器（例如三星瀏覽器）沒有桌面 DevTools 的 disable-cache 選項，
    # 改完 action.py 之後使用者沒辦法方便地手動清快取，靠這個標頭讓瀏覽器
    # 每次都重新驗證，避免又吃到修正前的舊版程式碼跑在瀏覽器裡。
    for path in ("/", "/action.py"):
        r = client.get(path)
        assert r.headers["cache-control"] == "no-cache"


def test_source_and_data_dirs_not_exposed():
    for path in ("/src/.env", "/src/components/machine_learning.py", "/data/basic_data.sql"):
        r = client.get(path)
        assert r.status_code == 404, f"{path} should not be servable, got {r.status_code}"


def test_chat_and_usage_share_client_id(tmp_path, monkeypatch):
    # /api/chat 累計的用量要能被 /api/usage 讀到——代表兩個端點確實共用
    # _resolve_client_id() 算出的同一把 key（連線 IP）。
    monkeypatch.setattr("web.backend.app.usage_store.USAGE_PATH", tmp_path / "usage.json")
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")
    with patch("web.backend.app.smart_reply_traced", return_value=("trace", "hi")):
        client.post("/api/chat", json={"message": "hello"})
    usage = client.get("/api/usage").json()
    assert usage["chars_used"] == len("hello") + len("hi")


def test_chat_endpoint_rejects_empty_message():
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 400


def test_chat_endpoint_returns_reply_from_self_built_model(tmp_path, monkeypatch):
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")
    with patch("web.backend.app.smart_reply_traced", return_value=("trace", "hi there")) as mock_smart_reply:
        r = client.post("/api/chat", json={"message": "hello"})
    mock_smart_reply.assert_called_once_with(
        "hello", out_dir=app_module.DEFAULT_OUT_DIR, force_mode="auto", history=None
    )
    assert r.status_code == 200
    assert r.json() == {"reply": "hi there", "conversation_id": None}


def test_chat_endpoint_without_conversation_id_does_not_persist(tmp_path, monkeypatch):
    # 舊的呼叫方式（沒帶 conversation_id）要維持無狀態行為——不能因為新增
    # 了對話紀錄功能，就逼著沒更新的呼叫端也得先建立一筆對話才能用 /api/chat。
    monkeypatch.setattr("web.backend.app.convo_store.CONVERSATIONS_PATH", tmp_path / "conversations.json")
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")
    with patch("web.backend.app.smart_reply_traced", return_value=("trace", "hi there")):
        client.post("/api/chat", json={"message": "hello"})
    assert app_module.convo_store.list_conversations("testclient") == []


def test_chat_endpoint_rejects_unknown_conversation_id():
    r = client.post("/api/chat", json={"message": "hello", "conversation_id": "not-a-real-id"})
    assert r.status_code == 404


def test_conversation_lifecycle_create_chat_get_delete(tmp_path, monkeypatch):
    monkeypatch.setattr("web.backend.app.convo_store.CONVERSATIONS_PATH", tmp_path / "conversations.json")
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")

    created = client.post("/api/conversations").json()
    conv_id = created["id"]
    assert created["title"] == "新對話"

    with patch("web.backend.app.smart_reply_traced", return_value=("trace", "嗨，我是 sinco")):
        chat_resp = client.post("/api/chat", json={"message": "你好", "conversation_id": conv_id})
    assert chat_resp.status_code == 200
    assert chat_resp.json() == {"reply": "嗨，我是 sinco", "conversation_id": conv_id}

    fetched = client.get(f"/api/conversations/{conv_id}").json()
    assert [m["role"] for m in fetched["messages"]] == ["user", "assistant"]
    assert fetched["title"] == "你好"

    listed = client.get("/api/conversations").json()
    assert any(c["id"] == conv_id for c in listed)

    renamed = client.patch(f"/api/conversations/{conv_id}", json={"title": "重新命名"}).json()
    assert renamed["title"] == "重新命名"

    cleared = client.post(f"/api/conversations/{conv_id}/clear").json()
    assert cleared["messages"] == []

    assert client.delete(f"/api/conversations/{conv_id}").status_code == 200
    assert client.get(f"/api/conversations/{conv_id}").status_code == 404


def test_commands_endpoint_lists_cli_equivalent_commands():
    data = client.get("/api/commands").json()
    for name in ("help", "clear", "open", "preview", "character", "memory",
                 "model", "resume", "chat", "conversations", "learn"):
        assert name in data["commands"]
    assert data["arg_suggestions"]["model"] == app_module.MODEL_MODES
    assert data["arg_suggestions"]["memory"] == ["list", "add", "del"]
    assert data["model_modes"] == app_module.MODEL_MODES


def test_chat_endpoint_open_command_feeds_file_content_to_model(tmp_path, monkeypatch):
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")
    sample = tmp_path / "note.txt"
    sample.write_text("檔案內容", encoding="utf-8")

    with patch("web.backend.app.smart_reply_traced", return_value=("trace", "讀到了")) as mock_smart_reply:
        r = client.post("/api/chat", json={"message": f"/open {sample}"})
    assert r.status_code == 200
    assert r.json()["reply"] == "讀到了"
    mock_smart_reply.assert_called_once_with(
        "檔案內容", out_dir=app_module.DEFAULT_OUT_DIR, force_mode="auto", history=None
    )


def test_chat_endpoint_open_command_missing_file_does_not_call_model():
    with patch("web.backend.app.smart_reply_traced") as mock_smart_reply:
        r = client.post("/api/chat", json={"message": "/open no-such-file.txt"})
    mock_smart_reply.assert_not_called()
    assert "找不到檔案" in r.json()["reply"]


def test_chat_endpoint_preview_command_returns_raw_file_without_calling_model(tmp_path):
    sample = tmp_path / "note.md"
    sample.write_text("# 標題", encoding="utf-8")
    with patch("web.backend.app.smart_reply_traced") as mock_smart_reply:
        r = client.post("/api/chat", json={"message": f"/preview {sample}"})
    mock_smart_reply.assert_not_called()
    assert "# 標題" in r.json()["reply"]


def test_chat_endpoint_resume_command_round_trips_session_store(tmp_path, monkeypatch):
    monkeypatch.setattr("web.backend.app.session_store.SESSION_PATH", tmp_path / "session.json")
    empty = client.post("/api/chat", json={"message": "/resume"}).json()
    assert "目前沒有記錄下的對話" in empty["reply"]

    app_module.session_store.record_turn("你好", "嗨", persona="sinco", mode="auto")
    replay = client.post("/api/chat", json={"message": "/resume"}).json()
    assert "你好" in replay["reply"] and "嗨" in replay["reply"]

    cleared = client.post("/api/chat", json={"message": "/resume clear"}).json()
    assert "已清除" in cleared["reply"]
    assert app_module.session_store.load_session() == []


def test_chat_endpoint_frontend_only_command_is_not_sent_to_model():
    # /model、/character、/clear、/chat、/conversations、/help 由前端直接
    # 處理，不會打進模型——直接呼叫 API 的情況下也不能被誤送進 smart_reply_traced()。
    with patch("web.backend.app.smart_reply_traced") as mock_smart_reply:
        r = client.post("/api/chat", json={"message": "/model nvidia"})
    mock_smart_reply.assert_not_called()
    assert "前端介面" in r.json()["reply"]


def test_chat_endpoint_unknown_command_is_not_sent_to_model():
    with patch("web.backend.app.smart_reply_traced") as mock_smart_reply:
        r = client.post("/api/chat", json={"message": "/not-a-real-command"})
    mock_smart_reply.assert_not_called()
    assert "未知指令" in r.json()["reply"]


def test_conversation_endpoints_404_for_unknown_id():
    assert client.get("/api/conversations/not-a-real-id").status_code == 404
    assert client.patch("/api/conversations/not-a-real-id", json={"title": "x"}).status_code == 404
    assert client.post("/api/conversations/not-a-real-id/clear").status_code == 404
    assert client.delete("/api/conversations/not-a-real-id").status_code == 404


def test_detect_endpoint_returns_real_detections():
    import ultralytics
    sample = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
    with open(sample, "rb") as f:
        r = client.post("/api/detect?conf=0.35", files={"frame": ("bus.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    labels = {d["label"] for d in data["detections"]}
    assert "bus" in labels
    assert "person" in labels
    assert data["width"] > 0 and data["height"] > 0


def test_detect_ws_endpoint_streams_detections_over_one_connection():
    import ultralytics
    sample = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
    frame_bytes = sample.read_bytes()

    with client.websocket_connect("/ws/detect?conf=0.35") as ws:
        # 送兩張畫面驗證同一條連線可以重複收送，不是連一次只能偵測一次。
        ws.send_bytes(frame_bytes)
        first = ws.receive_json()
        ws.send_bytes(frame_bytes)
        second = ws.receive_json()

    for data in (first, second):
        labels = {d["label"] for d in data["detections"]}
        assert "bus" in labels
        assert "person" in labels
        assert data["width"] > 0 and data["height"] > 0


def test_detect_ws_endpoint_reports_bad_frame_without_closing():
    with client.websocket_connect("/ws/detect?conf=0.35") as ws:
        ws.send_bytes(b"not a jpeg")
        data = ws.receive_json()
        assert data["error"] == "could not decode image"
        assert data["detections"] == []


def test_local_command_see_usage_and_missing(tmp_path):
    from app import _run_local_command
    assert _run_local_command("/see") == ("reply", "用法：/see <圖片路徑>")
    kind, text = _run_local_command(f"/see {tmp_path / 'nope.png'}")
    assert kind == "reply" and "找不到檔案" in text
