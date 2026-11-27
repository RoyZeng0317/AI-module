"""公開網路圖片搜尋與本機人物 YOLO 篩選，不訓練、不呼叫雲端 AI。"""
import hashlib, io, json, re, sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from url_display import markdown_label, source_link

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "output/image-search"
WEIGHTS = ROOT / "tranning/yolo_runs/custom/weights/characters.pt"
MAX_CANDIDATES, MAX_RESULTS = 8, 3
MAX_SEARCH_BYTES, MAX_IMAGE_BYTES, MAX_PIXELS = 2 * 1024 * 1024, 5 * 1024 * 1024, 20_000_000
TIMEOUT = (4, 8)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Nova-local/1.0)"}


def image_query(message: str) -> str | None:
    text = message.strip().strip("。!?！？")
    if re.search(r"不要|別|不必|不用|do not|don't", text, re.I):
        return None
    patterns = (
        r"^(?:(?:請|麻煩|幫我|請幫我|你|可以幫我|能幫我)\s*)?(?:上網)?(?:搜尋|搜索|尋找|找一下|找|查找)\s*(?:有關|關於)?\s*(.+?)\s*的?\s*(?:圖片|照片|相片)$",
        r"^(?:(?:請|幫我|請幫我|你)\s*)?(?:搜尋|搜索|尋找|找)\s*(?:圖片|照片|相片)\s*[:：]?\s*(.+)$",
        r"^(?:(?:please|can you|could you)\s+)?(?:find|search(?: for)?|look for)\s+(?:images?|pictures?|photos?)\s+(?:of\s+)?(.+)$",
        r"^(?:(?:please|can you|could you)\s+)?(?:find|search(?: for)?|look for)\s+(.+?)\s+(?:images?|pictures?|photos?)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            query = match[1].strip().rstrip("的").strip()
            return query[:160] if query else None
    return None


def _http_url(url) -> bool:
    if not isinstance(url, str) or any(c.isspace() or c in "<>\\" for c in url):
        return False
    try:
        parsed = urlsplit(url)
        return parsed.scheme in ("http", "https") and bool(parsed.hostname) and not parsed.username
    except ValueError:
        return False


def _fetch(url, limit, *, params=None, image=False):
    if not _http_url(url):
        raise ValueError("Invalid HTTP URL")
    with requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        if image and not content_type.startswith("image/"):
            raise ValueError("Not an image response")
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("Response exceeds size limit")
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError("Response exceeds size limit")
            chunks.append(chunk)
        return b"".join(chunks)


def parse_candidates(html: str) -> list[dict]:
    rows, seen = [], set()
    for anchor in BeautifulSoup(html, "html.parser").select("a.iusc[m]"):
        try:
            item = json.loads(anchor["m"])
        except (ValueError, TypeError):
            continue
        if not isinstance(item, dict):
            continue
        image_url, source = item.get("murl"), item.get("purl")
        if not _http_url(image_url) or not _http_url(source) or image_url in seen:
            continue
        title = item.get("t")
        rows.append({"image_url": image_url, "source": source,
                     "title": str(title or anchor.get_text(" ", strip=True) or "Image")[:240]})
        seen.add(image_url)
        if len(rows) == MAX_CANDIDATES:
            break
    return rows


def search_candidates(query: str) -> list[dict]:
    # 360 公開圖片資料，僅使用直接圖片與來源連結，不採用搜尋引擎的 AI 答覆。
    try:
        data = json.loads(_fetch("https://image.so.com/j", MAX_SEARCH_BYTES,
                                 params={"q": query, "pn": MAX_CANDIDATES, "sn": 0}).decode("utf-8"))
        rows, seen = [], set()
        items = data.get("list", []) if isinstance(data, dict) else []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            image_url, source = item.get("img"), item.get("link")
            if _http_url(image_url) and _http_url(source) and image_url not in seen:
                title = BeautifulSoup(str(item.get("title") or query), "html.parser").get_text(" ", strip=True)
                rows.append({"image_url": image_url, "source": source, "title": title[:240]})
                seen.add(image_url)
            if len(rows) == MAX_CANDIDATES:
                break
        if rows:
            return rows
    except (requests.RequestException, ValueError):
        pass
    # HTML 是公開資料頁，不是穩定 API；空白／驗證頁不冒充搜尋成功。
    for endpoint, params in (
        ("https://www.bing.com/images/async", {"q": query, "first": 0, "count": 35, "adlt": "moderate"}),
        ("https://www.bing.com/images/search", {"q": query, "form": "HDRSC3"}),
    ):
        try:
            rows = parse_candidates(_fetch(endpoint, MAX_SEARCH_BYTES, params=params).decode("utf-8", errors="replace"))
            if rows:
                return rows
        except (requests.RequestException, ValueError):
            continue
    return []


@lru_cache(maxsize=1)
def character_model():
    if not WEIGHTS.is_file() or WEIGHTS.stat().st_size == 0:
        raise ValueError("characters.pt is missing or empty")
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def download_image(url: str):
    data = _fetch(url, MAX_IMAGE_BYTES, image=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds pixel limit")
        return ImageOps.exif_transpose(image).convert("RGB")


def predict_target(model, image, target: str) -> float | None:
    import numpy as np, torch
    result = model.predict(np.asarray(image)[:, :, ::-1].copy(), conf=0.25, imgsz=640,
                           device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    scores = [float(box.conf[0]) for box in result.boxes if str(result.names[int(box.cls[0])]) == target]
    return max(scores) if scores else None


def _save_image(image, item: dict, query: str, score: float | None, state: str, lang="zh") -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 檔名含本輪內容，重複查詢也不覆寫其他對話正在預覽的來源與狀態。
    metadata = {**item, "query": query, "score": score, "state": state, "lang": lang}
    digest = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8") + image.tobytes()).hexdigest()[:24]
    path = CACHE_DIR / f"{digest}.png"
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    preview.save(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**metadata, "path": path.resolve().as_posix()}


def search_images(query: str, lang="zh") -> str:
    english = lang == "en"
    candidates = search_candidates(query)
    search_url = "https://www.bing.com/images/search?" + requests.compat.urlencode({"q": query})
    if not candidates:
        message = "Image search is not available, or no images were found. Try this search link:" if english else "圖片搜尋暫時無法取得結果，或沒有找到圖片；可開啟搜尋頁重試："
        return f"{message}\n{source_link(search_url)}"
    model, target, model_error = None, None, False
    try:
        model = character_model()
        names = model.names.values() if isinstance(model.names, dict) else model.names
        target = next((str(name) for name in names if str(name).casefold() == query.casefold()), None)
    except Exception:
        model_error = True
    found, fallback, errors, predicted = [], [], 0, 0
    for item in candidates[:MAX_CANDIDATES]:
        try:
            image = download_image(item["image_url"])
            if target and len(fallback) < MAX_RESULTS:
                preview = image.copy()
                preview.thumbnail((1600, 1600))
                fallback.append((preview, item))
            score = predict_target(model, image, target) if target else None
            predicted += bool(target)
            if target and score is None:
                continue
            state = "yolo" if target else "model_error" if model_error else "unverified"
            found.append(_save_image(image, item, query, score, state, lang))
            if len(found) == MAX_RESULTS:
                break
        except Exception:
            errors += 1
    unconfirmed_target = bool(target and not found)
    if unconfirmed_target:
        state = "no_match" if predicted else "inference_error"
        found = [_save_image(image, item, query, None, state, lang) for image, item in fallback]
    if not found:
        message = "Images could not be downloaded or checked." if english else "候選圖片無法下載、解碼或完成模型辨識。"
        return f"{message}\n{source_link(search_url)}"
    if unconfirmed_target and predicted:
        heading = "YOLO found 0 target matches. These are search images only, not confirmed by the person model." if english else "YOLO 目標人物匹配 0 張。以下僅為網路搜尋候選，未經人物模型確認。"
    elif unconfirmed_target:
        heading = "The person model could not check these images. These are search images only." if english else "人物模型未能完成辨識；以下僅為網路搜尋候選，未經人物模型確認。"
    elif target:
        heading = "Local YOLO predictions. These are not proof of identity." if english else "以下為本機 YOLO 預測匹配的圖片，不代表身份驗證。"
    elif model_error:
        heading = "The person model is not available. These search images are not checked." if english else "人物模型目前無法使用；以下僅為網路搜尋候選，未經人物模型確認。"
    else:
        heading = "The person model has no class for this query. These search images are not checked." if english else "人物模型沒有此查詢的姓名類別；以下僅為網路搜尋候選，未經人物模型確認。"
    lines = [heading]
    for i, item in enumerate(found, 1):
        score = f" — YOLO {item['score']:.1%}" if item["score"] is not None else ""
        label = f"{query} {i}{score}" if item["score"] is not None else (f"Search candidate {i}: {query}" if english else f"搜尋候選 {i}：{query}（未確認）")
        lines.extend([f"\n![{markdown_label(label)}](<{item['path']}>)", source_link(item["source"]),
                      source_link(item["image_url"], "Original image" if english else "原始圖片")])
    if errors:
        lines.append("Some candidates could not be read or checked." if english else "部分候選圖片無法讀取或完成辨識，已略過。")
    return "\n\n".join(lines)
