"""本機純迴圈影像處理 API；啟動：python -m uvicorn web.image_processing.app:app --host 127.0.0.1 --port 8060。"""
import asyncio, io, json, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image, UnidentifiedImageError
from lib.opencv2 import convolution as conv, histogram_equalization as hist
from web.backend.security_headers import IMAGE_CSP, SecurityHeadersMiddleware, security_headers

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / 'static'
JOBS_DIR = ROOT / 'output/image-processing-web/jobs'
MAX_BYTES, MAX_PIXELS = 10 * 1024 * 1024, 1048576
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='image-processing')
SLOTS = threading.BoundedSemaphore(3)
JOBS = {}
app = FastAPI(title='本機影像處理', docs_url=None, redoc_url=None)
app.add_middleware(SecurityHeadersMiddleware, policy=IMAGE_CSP)


@app.exception_handler(Exception)
async def internal_error(request, exc):
    return JSONResponse({'detail': '處理失敗，請稍後重試。'}, status_code=500,
                        headers=security_headers(IMAGE_CSP))


def fail(status, message):
    raise HTTPException(status, message)


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def invalid_options(request, exc):
    return JSONResponse({'detail': '處理參數無效，請重新選擇選項。'}, status_code=422)


def decode(data):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ('PNG', 'JPEG', 'WEBP', 'TIFF', 'BMP'):
                fail(415, '請使用 PNG、JPEG、WebP、TIFF 或 BMP 圖片。')
            if source.width * source.height > MAX_PIXELS:
                fail(413, '圖片超過 1,048,576 像素；請自行縮小後上傳。')
            source.seek(0)
            source.load()
            # 只做容器解碼；透明像素以純迴圈合成白底，不套內建濾鏡。
            rgba = source.convert('RGBA')
            image = Image.new('RGB', rgba.size)
            pixels, target = rgba.load(), image.load()
            for y in range(image.height):
                for x in range(image.width):
                    r, g, b, a = pixels[x, y]
                    target[x, y] = tuple((v*a + 255*(255-a) + 127)//255 for v in (r, g, b))
            return image
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        fail(413, '圖片像素過多。')
    except (UnidentifiedImageError, OSError, ValueError, EOFError):
        fail(400, '圖片無法解碼，請確認檔案完整。')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def run_job(job, image, options):
    started = time.monotonic()
    job.update(status='running', message='正在以 Python 迴圈處理。', started=started)
    directory = JOBS_DIR / job['id']
    try:
        directory.mkdir(parents=True)
        source = directory / 'input.png'
        image.save(source)
        previews = {'original': 'input.png'}
        if options['operation'] == 'histogram':
            summary = hist.process(source, directory)
            summary['source'] = 'uploaded image (first frame)'
            dump(directory / 'input_report.json', summary)
            previews.update(before='input_Y.png', after='input_Y_equalized.png',
                            histogram_before='input_Y_histogram.png', histogram_after='input_Y_equalized_histogram.png')
        else:
            rgb = conv.read_channels(source)
            channels = [conv.grayscale(rgb)] if options['channel'] == 'gray' else rgb
            result = [conv.convolve(channel, conv.FILTERS[options['filter']], options['padding']) for channel in channels]
            conv.save_display(channels, directory / 'before.png')
            conv.save_display(result, directory / 'clipped.png')
            conv.save_display(result, directory / 'minmax.png', minmax=True)
            low, high = float('inf'), float('-inf')
            for channel in result:
                for row in channel:
                    for value in row:
                        low, high = min(low, value), max(high, value)
            summary = dict(options, input_width=image.width, input_height=image.height,
                           output_width=len(result[0][0]), output_height=len(result[0]), minimum=low, maximum=high,
                           kernel=conv.FILTERS[options['filter']], convolution='kernel rotated 180 degrees',
                           display_note='PNG 為顯示映射；JSON 保留帶符號原值。')
            dump(directory / 'raw.json', dict(summary, channels=result))
            dump(directory / 'report.json', summary)
            previews.update(before='before.png', after=options['display']+'.png')
        files = []
        for path in sorted(directory.iterdir()):
            if path.suffix not in ('.png', '.json', '.csv'):
                continue
            media = {'.png': 'image/png', '.json': 'application/json', '.csv': 'text/csv'}[path.suffix]
            url = f"/api/jobs/{job['id']}/files/{path.name}"
            files.append(dict(name=path.name, label=path.name, url=url, download_url=url+'?download=1', media_type=media))
        job.update(status='done', message='處理完成。', summary=summary, files=files,
                   previews={key: f"/api/jobs/{job['id']}/files/{name}" for key, name in previews.items()},
                   elapsed_seconds=round(time.monotonic()-started, 3))
    except Exception:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with (JOBS_DIR.parent / 'errors.log').open('a', encoding='utf-8') as stream:
            import traceback
            stream.write(f"工作 {job['id']}\n" + traceback.format_exc() + '\n')
        job.update(status='error', message='處理失敗，請更換圖片或參數後重試。', elapsed_seconds=round(time.monotonic()-started, 3))
    finally:
        image.close()
        SLOTS.release()


@app.post('/api/jobs', status_code=202)
async def create_job(request: Request, operation: Literal['histogram', 'convolution'] = 'histogram',
                     filter: Literal['filter1', 'filter2', 'filter3'] = 'filter1',
                     padding: Literal['edge', 'zero', 'valid'] = 'edge',
                     channel: Literal['gray', 'rgb'] = 'gray', display: Literal['clipped', 'minmax'] = 'clipped'):
    if not SLOTS.acquire(blocking=False):
        fail(429, '目前已有一項處理及兩項等待工作，請稍後重試。')
    try:
        data = bytearray()
        async for chunk in request.stream():
            if len(data)+len(chunk) > MAX_BYTES:
                fail(413, '檔案超過 10 MiB。')
            data.extend(chunk)
        if not data:
            fail(400, '請先選擇圖片。')
        image = await asyncio.to_thread(decode, bytes(data))
        if operation == 'convolution' and padding == 'valid' and min(image.size) < len(conv.FILTERS[filter]):
            image.close()
            fail(400, 'valid 模式的圖片長寬不可小於卷積核。')
        identity = uuid.uuid4().hex
        name = unquote(request.headers.get('x-file-name', 'uploaded image'))[:240]
        job = dict(id=identity, status='queued', operation=operation, message='等待處理。', elapsed_seconds=0,
                   input=dict(name=name, width=image.width, height=image.height, mode='RGB',
                              frame=0, alpha='白底合成', orientation='保留解碼像素方向，不套 EXIF 旋轉'))
        JOBS[identity] = job
        EXECUTOR.submit(run_job, job, image, dict(operation=operation, filter=filter, padding=padding, channel=channel, display=display))
        return dict(id=identity, status=job['status'], status_url=f'/api/jobs/{identity}')
    except BaseException:
        SLOTS.release()
        raise


def find_job(identity):
    if identity not in JOBS:
        fail(404, '找不到工作；伺服器重啟後請重新上傳。')
    return JOBS[identity]


@app.get('/api/jobs/{identity}')
async def status(identity: str):
    result = dict(find_job(identity))
    started = result.pop('started', None)
    if result['status'] == 'running':
        result['elapsed_seconds'] = round(time.monotonic()-started, 2)
    return result


@app.get('/api/jobs/{identity}/files/{name}')
async def artifact(identity: str, name: str, download: bool = False):
    job = find_job(identity)
    selected = next((file for file in job.get('files', []) if file['name'] == name), None)
    if not selected:
        fail(404, '找不到結果檔案。')
    return FileResponse(JOBS_DIR / identity / name, media_type=selected['media_type'],
                        filename=name, content_disposition_type='attachment' if download else 'inline')


@app.get('/')
async def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/static/{name}')
async def static(name: str):
    if name not in ('app.js', 'style.css'):
        fail(404, '找不到檔案。')
    return FileResponse(STATIC / name)

