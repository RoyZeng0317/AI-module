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


def test_resolve_client_id_uses_google_uid_when_token_valid():
    with patch("web.backend.app.verify_id_token", return_value={"sub": "abc123"}):
        r = client.get("/api/usage", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 200


def test_chat_and_usage_share_client_id_from_verified_token(tmp_path, monkeypatch):
    # 有效 token 時，/api/chat 累計的用量要能被同一個帳號的 /api/usage 讀到
    # ——代表兩個端點確實共用 _resolve_client_id() 算出的同一把 key（"google:<uid>"），
    # 而不是各自退回不同的 IP。
    monkeypatch.setattr("web.backend.app.usage_store.USAGE_PATH", tmp_path / "usage.json")
    headers = {"Authorization": "Bearer whatever"}
    with patch("web.backend.app.verify_id_token", return_value={"sub": "uid-1"}):
        with patch("web.backend.app.smart_reply", return_value="hi"):
            client.post("/api/chat", json={"message": "hello"}, headers=headers)
        usage = client.get("/api/usage", headers=headers).json()
    assert usage["chars_used"] == len("hello") + len("hi")


def test_chat_endpoint_falls_back_to_ip_without_token(tmp_path, monkeypatch):
    # Authorization header 沒帶、或驗證失敗，行為要跟改動前完全一樣——不能
    # 逼著沒更新的舊呼叫端（或這次還沒串接登入的另一份前端）先登入才能用。
    monkeypatch.setattr("web.backend.app.usage_store.USAGE_PATH", tmp_path / "usage.json")
    with patch("web.backend.app.verify_id_token", return_value=None):
        with patch("web.backend.app.smart_reply", return_value="hi"):
            r = client.post("/api/chat", json={"message": "hello"})
    assert r.status_code == 200


def test_chat_endpoint_rejects_empty_message():
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 400


def test_chat_endpoint_returns_reply_from_self_built_model():
    with patch("web.backend.app.smart_reply", return_value="hi there") as mock_smart_reply:
        r = client.post("/api/chat", json={"message": "hello"})
    mock_smart_reply.assert_called_once_with("hello")
    assert r.status_code == 200
    assert r.json() == {"reply": "hi there", "conversation_id": None}


def test_chat_endpoint_without_conversation_id_does_not_persist(tmp_path, monkeypatch):
    # 舊的呼叫方式（沒帶 conversation_id）要維持無狀態行為——不能因為新增
    # 了對話紀錄功能，就逼著沒更新的呼叫端也得先建立一筆對話才能用 /api/chat。
    monkeypatch.setattr("web.backend.app.convo_store.CONVERSATIONS_PATH", tmp_path / "conversations.json")
    with patch("web.backend.app.smart_reply", return_value="hi there"):
        client.post("/api/chat", json={"message": "hello"})
    assert app_module.convo_store.list_conversations() == []


def test_chat_endpoint_rejects_unknown_conversation_id():
    r = client.post("/api/chat", json={"message": "hello", "conversation_id": "not-a-real-id"})
    assert r.status_code == 404


def test_conversation_lifecycle_create_chat_get_delete(tmp_path, monkeypatch):
    monkeypatch.setattr("web.backend.app.convo_store.CONVERSATIONS_PATH", tmp_path / "conversations.json")

    created = client.post("/api/conversations").json()
    conv_id = created["id"]
    assert created["title"] == "新對話"

    with patch("web.backend.app.smart_reply", return_value="嗨，我是 sinco"):
        chat_resp = client.post(
            "/api/chat", json={"message": "你好", "conversation_id": conv_id}
        )
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
