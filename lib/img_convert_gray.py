"""圖片轉灰階 → 找特徵（邊緣/輪廓切出區域）→ 依特徵上色 → 文字分析報告。

純 OpenCV 古典影像處理，不用任何 AI 模型或外部 API（Rule 06）。

流程：
  1. 讀圖並轉灰階
  2. 高斯模糊去雜訊 → Canny 邊緣 → 膨脹/閉運算把斷掉的邊接起來 → 找外輪廓 = 區域
  3. 每個區域算特徵：形狀、面積、位置、灰階亮度、紋理（邊緣密度）、原圖主色
  4. 上色（在灰階底圖上）：
       --mode original：每個區域填回原圖該區域的平均顏色（灰階轉回彩色）
       --mode shape   ：依形狀類別上色（圓=紅、方=綠、三角=藍、其他=黃）
       --mode brightness：依區域亮度上色（暗=藍、中=綠、亮=紅）
  5. 印出文字分析報告，並把灰階/邊緣/上色結果存到輸出資料夾

用法：
    python lib/img_convert_gray.py                      # 預設用 lib/opencv2/koala.jpg
    python lib/img_convert_gray.py photo.jpg --mode shape --out output/gray_features
    python lib/img_convert_gray.py photo.jpg --show     # 另外開視窗顯示
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "opencv2" / "koala.jpg"
DEFAULT_OUT = HERE.parent / "output" / "gray_features"

# BGR 顏色
SHAPE_COLORS = {
    "圓形": (0, 0, 255),
    "正方形": (0, 200, 0),
    "長方形": (0, 200, 0),
    "三角形": (255, 0, 0),
}
OTHER_COLOR = (0, 220, 255)
BRIGHTNESS_COLORS = {"暗": (255, 0, 0), "中": (0, 200, 0), "亮": (0, 0, 255)}


# ---------- 讀寫（用 imdecode/imencode，路徑有中文也不會讀不到） ----------

def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"無法讀取 {path}，請確認檔案路徑及圖片格式")
    return img


def write_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"無法寫入 {path}")
    buf.tofile(str(path))


# ---------- 1. 灰階 ----------

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------- 2. 特徵：邊緣 → 區域 ----------

def find_edges(gray, sigma=0.33):
    """Canny 邊緣，門檻依畫面亮度中位數自動調整（暗圖、亮圖都抓得到）。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    low = int(max(0, (1.0 - sigma) * med))
    high = int(min(255, max(low + 20, (1.0 + sigma) * med)))
    return cv2.Canny(blur, low, high)


def quantize_gray(gray, levels=4):
    """用 k-means 把灰階亮度分成幾層（每層 = 亮度相近的像素），回傳每個像素的層編號。"""
    blur = cv2.medianBlur(gray, 5)
    data = blur.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)  # 固定隨機種子，同一張圖每次結果一樣
    _, labels, _ = cv2.kmeans(data, levels, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(gray.shape)


def find_regions(gray, levels=4, min_area_ratio=0.002, max_regions=30):
    """灰階分層後，每一層裡連在一起的像素就是一個區域。
    回傳面積由大到小的區域遮罩清單（濾掉雜點）。"""
    labels = quantize_gray(gray, levels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_area = gray.size * min_area_ratio
    masks = []
    for level in range(levels):
        layer = np.uint8(labels == level) * 255
        layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(layer, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                masks.append(np.uint8(comp == i) * 255)
    masks.sort(key=cv2.countNonZero, reverse=True)
    return masks[:max_regions]


def is_background(mask):
    """面積大、而且碰到畫面 3 個以上邊界的區域視為背景。"""
    h, w = mask.shape
    touches = sum(bool(mask[edge].any()) for edge in
                  (np.s_[0, :], np.s_[h - 1, :], np.s_[:, 0], np.s_[:, w - 1]))
    return touches >= 3 and cv2.countNonZero(mask) > 0.25 * mask.size


# ---------- 3. 每個區域的特徵 ----------

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri == 0:
        return "不規則"
    # 實心度 = 面積 / 凸包面積；照片裡凹凸不平的色塊實心度低，不硬套幾何形狀
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0 or area / hull_area < 0.9:
        return "不規則"
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    circularity = 4 * np.pi * area / (peri * peri)
    n = len(approx)
    if n == 3:
        return "三角形"
    if n == 4:
        _, _, w, h = cv2.boundingRect(approx)
        return "正方形" if 0.9 <= w / float(h) <= 1.1 else "長方形"
    if circularity > 0.8:
        return "圓形"
    if 5 <= n <= 8:
        return f"{n}邊形"
    return "不規則"


def color_name(bgr):
    """把平均 BGR 轉成中文顏色名稱（HSV 色相分段）。"""
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    if v < 50:
        return "黑"
    if s < 40:
        return "白" if v > 200 else "灰"
    h = int(h) * 2  # OpenCV 色相 0~179 → 0~358 度
    for limit, name in ((15, "紅"), (40, "橙"), (70, "黃"), (160, "綠"),
                        (200, "青"), (260, "藍"), (300, "紫"), (340, "粉紅")):
        if h < limit:
            return name
    return "紅"


def brightness_level(value):
    return "暗" if value < 85 else ("中" if value < 170 else "亮")


def position_name(cx, cy, w, h):
    row = "上" if cy < h / 3 else ("下" if cy > 2 * h / 3 else "中")
    col = "左" if cx < w / 3 else ("右" if cx > 2 * w / 3 else "中")
    return "正中" if row == col == "中" else f"{row}{col}"


def region_features(img, gray, edges, mask, index):
    h, w = gray.shape
    pixels = max(int(cv2.countNonZero(mask)), 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

    mean_gray = cv2.mean(gray, mask=mask)[0]
    mean_bgr = tuple(int(round(c)) for c in cv2.mean(img, mask=mask)[:3])
    # 紋理：輪廓範圍內（含內部細節）的邊緣像素比例
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, filled)) / max(cv2.countNonZero(filled), 1)
    background = is_background(mask)

    return {
        "id": index,
        "contour": contour,
        "mask": mask,
        "background": background,
        "shape": "背景" if background else classify_shape(contour),
        "area": float(cv2.contourArea(contour)),
        "area_ratio": pixels / float(h * w),
        "bbox": cv2.boundingRect(contour),
        "center": (int(cx), int(cy)),
        "position": position_name(cx, cy, w, h),
        "mean_gray": mean_gray,
        "brightness": brightness_level(mean_gray),
        "edge_density": edge_density,
        "texture": "紋理多" if edge_density > 0.15 else ("有些紋理" if edge_density > 0.05 else "平滑"),
        "mean_bgr": mean_bgr,
        "color": color_name(mean_bgr),
    }


def analyze(img):
    gray = to_gray(img)
    edges = find_edges(gray)
    regions = [region_features(img, gray, edges, m, i + 1)
               for i, m in enumerate(find_regions(gray))]
    return gray, edges, regions


# ---------- 4. 依特徵上色 ----------

def recolor(gray, regions, mode="original"):
    """在灰階底圖上，依 mode 把每個區域上色，並標上編號（對應報告）。"""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 大的先畫、小的後畫，避免小區域被大區域蓋掉
    for r in sorted(regions, key=lambda r: r["area"], reverse=True):
        if mode == "original":
            color = r["mean_bgr"]
        elif mode == "shape":
            if r["background"]:
                continue  # 背景保持灰階，只替物體上色
            color = SHAPE_COLORS.get(r["shape"], OTHER_COLOR)
        elif mode == "brightness":
            color = BRIGHTNESS_COLORS[r["brightness"]]
        else:
            raise ValueError(f"未知的上色模式：{mode}")
        layer = out.copy()
        layer[r["mask"] > 0] = color
        # 原本的灰階明暗保留一部分，看得出細節
        out = np.where(r["mask"][..., None] > 0,
                       cv2.addWeighted(layer, 0.6, out, 0.4, 0), out)
    for r in regions:
        cv2.drawContours(out, [r["contour"]], -1, (255, 255, 255), 1)
        cv2.putText(out, str(r["id"]), r["center"], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.putText(out, str(r["id"]), r["center"], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    return out


# ---------- 5. 文字分析報告 ----------

def overall_summary(img, gray, edges, regions):
    mean_bgr = tuple(int(c) for c in cv2.mean(img)[:3])
    contrast = float(gray.std())
    shape_count = {}
    for r in regions:
        if not r["background"]:
            shape_count[r["shape"]] = shape_count.get(r["shape"], 0) + 1
    return {
        "size": (img.shape[1], img.shape[0]),
        "brightness": float(gray.mean()),
        "brightness_level": brightness_level(gray.mean()),
        "contrast": contrast,
        "contrast_level": "低" if contrast < 30 else ("中" if contrast < 60 else "高"),
        "edge_density": cv2.countNonZero(edges) / float(gray.size),
        "dominant_color": color_name(mean_bgr),
        "shape_count": shape_count,
    }


def format_report(summary, regions):
    w, h = summary["size"]
    lines = [
        "===== 圖片特徵分析報告 =====",
        f"尺寸：{w} x {h}",
        f"整體亮度：{summary['brightness']:.1f}（{summary['brightness_level']}）",
        f"對比度：{summary['contrast']:.1f}（{summary['contrast_level']}）",
        f"邊緣密度：{summary['edge_density']:.1%}",
        f"整體主色：{summary['dominant_color']}",
        f"偵測到區域：{len(regions)} 個",
    ]
    if summary["shape_count"]:
        lines.append("形狀統計：" + "、".join(f"{k} {v}" for k, v in summary["shape_count"].items()))
    lines.append("")
    for r in regions:
        x, y, bw, bh = r["bbox"]
        lines.append(
            f"[{r['id']}] {r['shape']}｜位置 {r['position']} ({x},{y},{bw}x{bh})｜"
            f"佔畫面 {r['area_ratio']:.1%}｜亮度 {r['mean_gray']:.0f}（{r['brightness']}）｜"
            f"{r['texture']}｜原色 {r['color']}"
        )
    lines.append("")
    lines.append("理解：" + describe(summary, regions))
    return "\n".join(lines)


def describe(summary, regions):
    """把數字特徵整理成一句話的描述。"""
    objects = [r for r in regions if not r["background"]]
    if not objects:
        return (f"畫面整體偏{summary['brightness_level']}、對比{summary['contrast_level']}，"
                "沒有找到明顯的物體區域（可能是單純背景或亮度差異太小）。")
    main = objects[0]
    parts = [
        f"畫面整體偏{summary['brightness_level']}、對比{summary['contrast_level']}、主色為{summary['dominant_color']}色",
        f"共 {len(objects)} 個物體區域，最大的在{main['position']}，是{main['color']}色的"
        f"{'不規則形狀' if main['shape'] == '不規則' else main['shape']}，"
        f"佔畫面 {main['area_ratio']:.0%}",
    ]
    backgrounds = [r for r in regions if r["background"]]
    if backgrounds:
        parts.append(f"背景是{backgrounds[0]['color']}色、{backgrounds[0]['texture']}")
    if summary["edge_density"] > 0.1:
        parts.append("邊緣很多，畫面細節/紋理豐富")
    elif summary["edge_density"] < 0.02:
        parts.append("邊緣很少，畫面單純")
    regular = sum(v for k, v in summary["shape_count"].items()
                  if k in ("圓形", "正方形", "長方形", "三角形"))
    if regular >= len(objects) / 2:
        parts.append("大多是規則幾何形狀，比較像圖形/示意圖")
    else:
        parts.append("大多是不規則形狀，比較像自然照片")
    return "；".join(parts) + "。"


# ---------- CLI ----------

def run(image_path, mode="original", out_dir=DEFAULT_OUT, show=False):
    img = read_image(image_path)
    gray, edges, regions = analyze(img)
    colored = recolor(gray, regions, mode)
    report = format_report(overall_summary(img, gray, edges, regions), regions)

    out_dir = Path(out_dir)
    stem = Path(image_path).stem
    write_image(out_dir / f"{stem}_gray.png", gray)
    write_image(out_dir / f"{stem}_edges.png", edges)
    write_image(out_dir / f"{stem}_{mode}.png", colored)

    if show:
        cv2.imshow("gray", gray)
        cv2.imshow("edges", edges)
        cv2.imshow(f"recolor ({mode})", colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return report, colored


def main():
    p = argparse.ArgumentParser(description="灰階 → 特徵 → 依特徵上色 → 分析報告")
    p.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    p.add_argument("--mode", choices=["original", "shape", "brightness"], default="original")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="輸出資料夾")
    p.add_argument("--show", action="store_true", help="開視窗顯示結果")
    args = p.parse_args()

    report, _ = run(args.image, args.mode, args.out, args.show)
    print(report)
    print(f"\n輸出圖片已存到：{args.out}")


if __name__ == "__main__":
    main()
