from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2


class ImageEditorError(Exception):
    """Expected, user-correctable image processing error."""


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise ImageEditorError(f"找不到輸入圖片：{path}")
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageEditorError(f"無法讀取圖片或格式不支援：{path}") from exc


def save_image(image: Image.Image, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(output)
    except OSError as exc:
        raise ImageEditorError(f"無法寫入輸出圖片：{output}") from exc
    return output


def image_info(path: str | Path) -> dict:
    im = load_image(path)
    p = Path(path)
    return {
        "path": str(p.resolve()),
        "format": im.format or p.suffix.lstrip(".").upper(),
        "width": im.width,
        "height": im.height,
        "mode": im.mode,
        "bytes": p.stat().st_size,
    }


def enhance_image(
    source: str | Path,
    output: str | Path,
    strength: int = 5,
    sharpen: float = 0.25,
) -> Path:
    if not 0 <= strength <= 20:
        raise ImageEditorError("strength 必須介於 0 到 20。")
    if not 0.0 <= sharpen <= 1.0:
        raise ImageEditorError("sharpen 必須介於 0.0 到 1.0。")

    im = load_image(source)
    original_mode = im.mode
    # OpenCV works in BGR; preserve alpha separately when present.
    rgba = im.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if strength > 0:
        processed = cv2.fastNlMeansDenoisingColored(
            bgr, None, h=strength, hColor=strength, templateWindowSize=7,
            searchWindowSize=21
        )
    else:
        processed = bgr

    if sharpen > 0:
        blurred = cv2.GaussianBlur(processed, (0, 0), sigmaX=1.0)
        processed = cv2.addWeighted(processed, 1.0 + sharpen, blurred, -sharpen, 0)

    out_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    out = np.dstack([out_rgb, alpha])
    result = Image.fromarray(out, mode="RGBA")
    # Keep alpha if source had it; otherwise write RGB.
    if original_mode != "RGBA" and "A" not in original_mode:
        result = result.convert("RGB")
    return save_image(result, output)


def inpaint_image(
    source: str | Path,
    mask_path: str | Path,
    output: str | Path,
    radius: float = 5.0,
) -> Path:
    if not 1.0 <= radius <= 20.0:
        raise ImageEditorError("radius 必須介於 1 到 20。")

    im = load_image(source).convert("RGB")
    mask_im = load_image(mask_path).convert("L")

    if im.size != mask_im.size:
        raise ImageEditorError(
            f"遮罩尺寸 {mask_im.size} 必須與原圖尺寸 {im.size} 相同。"
        )

    rgb = np.array(im)
    mask = np.array(mask_im)
    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    if not np.any(binary_mask):
        raise ImageEditorError("遮罩是空白的；請用白色標記要修復的區域。")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, binary_mask, radius, cv2.INPAINT_TELEA)
    result = Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
    return save_image(result, output)


def cutout_image(source: str | Path, output: str | Path) -> Path:
    try:
        from rembg import remove
    except ImportError as exc:
        raise ImageEditorError(
            '尚未安裝 rembg。請執行：pip install -e ".[subject]"'
        ) from exc

    im = load_image(source).convert("RGBA")
    try:
        result = remove(im)
        if not isinstance(result, Image.Image):
            result = Image.open(result).convert("RGBA")
        return save_image(result.convert("RGBA"), output)
    except Exception as exc:
        raise ImageEditorError(f"主體擷取失敗：{exc}") from exc
