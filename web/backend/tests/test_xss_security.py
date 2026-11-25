"""Stored payloads remain JSON data; frontend replay tests cover DOM execution."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from web.backend import app as backend
from web.backend.security_headers import PYSCRIPT_CSP

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = '<img src=x onerror="window.__xss=1"> <script>window.__xss=2</script> 中文 `x < 3`'


@pytest.fixture
def client(monkeypatch):
    # Keep all persistence within this project and never read the real ledgers.
    import tempfile
    directory = ROOT / 'output/xss-tests'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        for module, key, name in ((backend.convo_store, 'CONVERSATIONS_PATH', 'conversations'),
                                  (backend.session_store, 'SESSION_PATH', 'session'),
                                  (backend.usage_store, 'USAGE_PATH', 'usage')):
            monkeypatch.setattr(module, key, Path(temporary) / (name + '.json'))
        monkeypatch.setattr(backend, '_chat_request_log', {})
        monkeypatch.setattr(backend, '_check_chat_rate_limit', lambda identity: None)
        with TestClient(backend.app, raise_server_exceptions=False) as session:
            yield session


def test_persisted_payload_reloaded_as_json(client):
    identity = client.post('/api/conversations').json()['id']
    with patch.object(backend, 'smart_reply_traced', return_value=('test only', PAYLOAD)):
        response = client.post('/api/chat', json={'conversation_id': identity, 'message': PAYLOAD})
    assert response.status_code == 200
    title = '<svg/onload=alert(1)>'
    assert client.patch('/api/conversations/' + identity, json={'title': title}).status_code == 200
    # GET reads the on-disk JSON anew; there is no in-memory conversation cache.
    replay = client.get('/api/conversations/' + identity)
    assert replay.headers['content-type'] == 'application/json'
    conversation = replay.json()
    assert conversation['title'] == title
    assert [(item['role'], item['content']) for item in conversation['messages']] == [('user', PAYLOAD), ('assistant', PAYLOAD)]
    assert client.get('/api/conversations').json()[0]['title'] == title
    stored = json.loads(backend.convo_store.CONVERSATIONS_PATH.read_text(encoding='utf-8'))
    assert stored[0]['messages'][0]['content'] == PAYLOAD
    assert backend.session_store.load_session()[0]['reply'] == PAYLOAD
    (ROOT / 'output/xss-tests/api-replay.json').write_text(json.dumps(conversation, ensure_ascii=False), encoding='utf-8')


@pytest.mark.parametrize('path,status', [('/', 200), ('/action.py', 200), ('/api/health', 200),
                                       ('/unknown', 404), ('/api/conversations/missing', 404)])
def test_headers_on_static_json_and_errors(client, path, status):
    response = client.get(path)
    assert response.status_code == status
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['content-security-policy'] == PYSCRIPT_CSP
    assert response.headers['x-frame-options'] == 'DENY'


def test_internal_error_is_json_and_has_defenses(client):
    with patch.object(backend, 'get_gpu_info', side_effect=RuntimeError('<img onerror=alert(1)>')):
        response = client.get('/api/gpu_info')
    assert response.status_code == 500
    assert response.headers['content-type'] == 'application/json'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['content-security-policy'] == PYSCRIPT_CSP


def test_static_hosting_header_policy_matches_backend():
    config = json.loads((ROOT / 'web/firebase.json').read_text(encoding='utf-8'))
    headers = {item['key']: item['value'] for item in config['hosting']['headers'][0]['headers']}
    assert headers['Content-Security-Policy'] == PYSCRIPT_CSP
    assert headers['X-Content-Type-Options'] == 'nosniff'


def test_mount_prefix_redirect_static_and_error_headers(monkeypatch):
    import importlib.util
    monkeypatch.setenv('MOUNT_PREFIX', '/AI-Module')
    spec = importlib.util.spec_from_file_location('xss_mounted_app', backend.__file__)
    mounted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mounted)
    with TestClient(mounted.app, follow_redirects=False) as session:
        for path, status in (('/', 307), ('/AI-Module/', 200), ('/AI-Module/action.py', 200),
                             ('/AI-Module/api/health', 200), ('/AI-Module/missing', 404)):
            response = session.get(path)
            assert response.status_code == status
            assert response.headers['x-content-type-options'] == 'nosniff'
            assert response.headers['content-security-policy'] == PYSCRIPT_CSP
        assert session.get('/').headers['location'] == '/AI-Module/'
