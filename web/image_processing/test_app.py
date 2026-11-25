import io, json, subprocess, sys, threading, time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from web.image_processing import app as backend
from lib.opencv2 import convolution as conv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'JOBS_DIR', tmp_path / 'jobs')
    backend.JOBS.clear()
    with TestClient(backend.app) as client:
        yield client


def png(size=(5, 5), mode='RGB'):
    image = Image.new(mode, size)
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), ((x*20, y*30, x+y) if mode == 'RGB' else (255, 0, 0, 0)))
    stream = io.BytesIO()
    image.save(stream, 'PNG')
    return stream.getvalue()


def finish(client, response):
    assert response.status_code == 202, response.text
    for _ in range(400):
        job = client.get(response.json()['status_url']).json()
        if job['status'] in ('done', 'error'):
            return job
        time.sleep(.01)
    pytest.fail('工作逾時')


def test_convolution_values_download_and_safety(client):
    data = png()
    job = finish(client, client.post('/api/jobs?operation=convolution&filter=filter3&channel=rgb&padding=valid&display=minmax', content=data,
                                    headers={'X-File-Name': '..%2F.env'}))
    assert job['status'] == 'done'
    assert job['input']['name'] == '../.env'
    raw_file = next(item for item in job['files'] if item['name'] == 'raw.json')
    raw = client.get(raw_file['url']).json()
    channels = conv.read_channels(backend.JOBS_DIR / job['id'] / 'input.png')
    assert raw['channels'] == [conv.convolve(channel, conv.FILTERS['filter3'], 'valid') for channel in channels]
    assert raw['output_width'] == raw['output_height'] == 3
    assert raw['minimum'] < 0
    response = client.get(raw_file['download_url'])
    assert response.headers['content-type'] == 'application/json'
    assert 'attachment' in response.headers['content-disposition']
    assert client.get(f"/api/jobs/{job['id']}/files/.env").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/files/%2e%2e%2f.env").status_code == 404
    assert client.get('/api/jobs/unknown').status_code == 404
    assert client.get('/static/.env').status_code == 404


def test_histogram_counts_and_report(client):
    job = finish(client, client.post('/api/jobs?operation=histogram', content=png()))
    assert job['status'] == 'done'
    assert job['summary']['pixels'] == 25
    assert 'C:' not in job['summary']['source']
    csv_file = next(item for item in job['files'] if item['name'].endswith('.csv'))
    import csv
    rows = list(csv.DictReader(io.StringIO(client.get(csv_file['url']).text)))
    assert len(rows) == 256
    assert sum(int(row['before_count']) for row in rows) == 25
    assert sum(int(row['after_count']) for row in rows) == 25
    assert int(rows[-1]['cdf_count']) == 25
    preview = client.get(job['previews']['before'])
    assert preview.headers['content-type'] == 'image/png'
    assert Image.open(io.BytesIO(preview.content)).size == (5, 5)


@pytest.mark.parametrize('body,expected', [(b'', 400), (b'not an image', 400), (b'x'*(backend.MAX_BYTES+1), 413)], ids=['empty', 'invalid', 'oversize'])
def test_bad_upload(client, body, expected):
    assert client.post('/api/jobs', content=body).status_code == expected


def test_xss_filename_and_generated_mime(client):
    from urllib.parse import quote
    name = '<img src=x onerror="window.__xss=1">.png'
    job = finish(client, client.post('/api/jobs?operation=convolution', content=png(),
                                    headers={'X-File-Name': quote(name)}))
    response = client.get('/api/jobs/' + job['id'])
    assert response.headers['content-type'] == 'application/json'
    assert response.json()['input']['name'] == name
    for file in job['files']:
        result = client.get(file['url'])
        assert result.headers['x-content-type-options'] == 'nosniff'
        assert result.headers['content-type'].split(';')[0] == file['media_type']
        if file['media_type'] == 'image/png':
            assert result.content.startswith(b'\x89PNG\r\n\x1a\n')


def test_html_disguised_as_image_rejected(client):
    response = client.post('/api/jobs', content=b'<html><script>alert(1)</script></html>',
                           headers={'Content-Type': 'image/png', 'X-File-Name': 'picture.png'})
    assert response.status_code == 400
    assert response.headers['content-type'] == 'application/json'


@pytest.mark.parametrize('path', ['/', '/static/app.js', '/static/style.css', '/static/missing', '/api/jobs/missing'])
def test_security_headers(client, path):
    from web.backend.security_headers import IMAGE_CSP
    response = client.get(path)
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['content-security-policy'] == IMAGE_CSP
    assert response.headers['x-frame-options'] == 'DENY'


def test_internal_error_headers(client, monkeypatch):
    monkeypatch.setattr(backend, 'find_job', lambda identity: 1/0)
    with TestClient(backend.app, raise_server_exceptions=False) as session:
        response = session.get('/api/jobs/missing')
    assert response.status_code == 500
    assert response.headers['content-type'] == 'application/json'
    assert response.headers['x-content-type-options'] == 'nosniff'


def test_limits_formats_and_modes(client):
    assert client.post('/api/jobs?operation=convolution&padding=valid&filter=filter2', content=png((2, 2))).status_code == 400
    assert client.post('/api/jobs?padding=wrong', content=png()).status_code == 422
    assert client.post('/api/jobs?operation=other', content=png()).status_code == 422
    stream = io.BytesIO()
    Image.new('RGB', (1025, 1024)).save(stream, 'PNG')
    assert client.post('/api/jobs', content=stream.getvalue()).status_code == 413
    stream = io.BytesIO()
    Image.new('RGB', (2, 2)).save(stream, 'GIF')
    assert client.post('/api/jobs', content=stream.getvalue()).status_code == 415


def test_transparency_manual_white(client):
    job = finish(client, client.post('/api/jobs', content=png(mode='RGBA')))
    response = client.get(job['previews']['original'])
    assert Image.open(io.BytesIO(response.content)).getpixel((0, 0)) == (255, 255, 255)


def test_capacity_and_worker_failure(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def broken(*args):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('private traceback marker')
    monkeypatch.setattr(backend.hist, 'process', broken)
    responses = [client.post('/api/jobs', content=png())]
    assert entered.wait(2)
    responses.extend(client.post('/api/jobs', content=png()) for _ in range(2))
    assert client.post('/api/jobs', content=png()).status_code == 429
    assert client.get(responses[1].json()['status_url']).json()['status'] == 'queued'
    release.set()
    for response in responses:
        job = finish(client, response)
        assert job['status'] == 'error'
        assert 'traceback' not in json.dumps(job)
    assert 'private traceback marker' in (backend.JOBS_DIR.parent / 'errors.log').read_text(encoding='utf-8')


def test_clean_runtime_import():
    script = "import sys; import web.image_processing.app; assert not any(n.split('.')[0] in ('numpy','cv2','scipy','torch') for n in sys.modules)"
    subprocess.run([sys.executable, '-c', script], cwd=backend.ROOT, check=True)

