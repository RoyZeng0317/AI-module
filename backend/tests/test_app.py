"""HTTP-level tests for app.py — catches wiring bugs that a direct call to
detector.detect() wouldn't (routing, static file exposure, query params).
"""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_public_assets_served():
    for name in ("script.js", "style.css"):
        r = client.get(f"/{name}")
        assert r.status_code == 200


def test_source_and_data_dirs_not_exposed():
    for path in ("/src/.env", "/src/components/machine_learning.py", "/data/basic_data.sql"):
        r = client.get(path)
        assert r.status_code == 404, f"{path} should not be servable, got {r.status_code}"


def test_chat_endpoint_rejects_empty_message():
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 400


def test_chat_endpoint_returns_reply_from_self_built_model():
    with patch("app.chat_reply", return_value="hi there") as mock_chat_reply:
        r = client.post("/api/chat", json={"message": "hello"})
    mock_chat_reply.assert_called_once_with("hello")
    assert r.status_code == 200
    assert r.json() == {"reply": "hi there"}


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
