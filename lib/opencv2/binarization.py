"""以 Python for 迴圈做手動灰階、固定門檻與 Otsu 二值化；Pillow 僅負責讀寫與圖表組裝。

灰階：gray = round_half_up(0.299R + 0.587G + 0.114B)，以整數 (299R+587G+114B+500)//1000 計算。
門檻慣例：pixel >= T 為 255，否則為 0（T 是前景類別的第一個灰階值）。
Otsu：T 從 0 到 255，背景為 [0, T-1]、前景為 [T, 255]，取類間變異數最大者；
同分取最小 T；某一類為空時變異數定義為 0，所以常數圖片全部為 0，T=0，輸出全白。
"""
import argparse, csv, json, sys
from fractions import Fraction
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.opencv2.histogram_equalization import histogram, histogram_chart  # noqa: E402

DEFAULT_THRESHOLD = 128


def gray_value(r, g, b):
    """0.299R+0.587G+0.114B，四捨五入（.5 進位）；整數運算避免浮點誤差。"""
    return (299 * r + 587 * g + 114 * b + 500) // 1000


def to_gray(image):
    """逐像素手動灰階，回傳 L 模式容器；不呼叫 Pillow 內建灰階轉換。"""
    if image.mode not in ("L", "RGB", "RGBA") or not image.width or not image.height:
        raise ValueError("輸入必須是非空 L、RGB 或 RGBA 圖片")
    result = Image.new("L", image.size)
    source, output = image.load(), result.load()
    for y in range(image.height):
        for x in range(image.width):
            pixel = source[x, y]
            output[x, y] = pixel if image.mode == "L" else gray_value(*pixel[:3])
    return result


def matrix_to_image(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("影像矩陣必須非空且為矩形")
    image = Image.new("L", (len(matrix[0]), len(matrix)))
    pixels = image.load()
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError("像素值必須是0至255的整數")
            pixels[x, y] = value
    return image


def threshold_image(gray, threshold):
    """pixel >= threshold 為 255，否則為 0。"""
    if gray.mode != "L":
        raise ValueError("二值化輸入必須是灰階圖片")
    if type(threshold) is not int or not 0 <= threshold <= 256:
        raise ValueError("門檻必須是0至256的整數")
    result = Image.new("L", gray.size)
    source, output = gray.load(), result.load()
    for y in range(gray.height):
        for x in range(gray.width):
            output[x, y] = 255 if source[x, y] >= threshold else 0
    return result


def between_class_variances(counts):
    """回傳長度256的 Fraction 清單：index T 為以 T 切分時的類間變異數。

    σ²_B(T) = w0·w1·(μ0-μ1)² = (S0·N - S·n0)² / (n0·n1·N²)，
    n0/S0 為 [0, T-1] 的像素數與灰階總和；任一類為空時定義為 0。
    """
    if len(counts) != 256 or any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError("需要256個非負整數bin，且總像素數須大於0")
    total, weighted = 0, 0
    for level in range(256):
        total += counts[level]
        weighted += level * counts[level]
    variances, n0, s0 = [], 0, 0
    for threshold in range(256):
        n1 = total - n0
        if n0 == 0 or n1 == 0:
            variances.append(Fraction(0))
        else:
            difference = s0 * total - weighted * n0
            variances.append(Fraction(difference * difference, n0 * n1 * total * total))
        n0 += counts[threshold]
        s0 += threshold * counts[threshold]
    return variances


def otsu_threshold(counts):
    """類間變異數最大的 T；以 Fraction 精確比較，同分取最小 T。"""
    variances = between_class_variances(counts)
    best = 0
    for threshold in range(1, 256):
        if variances[threshold] > variances[best]:
            best = threshold
    return best, variances


def marked_histogram(counts, fixed, otsu):
    chart = histogram_chart(counts, f"Gray histogram: fixed T={fixed} (red), Otsu T={otsu} (green)")
    draw = ImageDraw.Draw(chart)
    for threshold, color in ((fixed, "#d03030"), (otsu, "#209040")):
        x = 32 + min(threshold, 255) * 2
        draw.line((x, 30, x, 280), fill=color, width=1)
    return chart


def process(source, directory, threshold=DEFAULT_THRESHOLD):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    with Image.open(source) as image:
        # 調色盤/16位元等模式只解碼成RGB，灰階與門檻仍由迴圈計算。
        decoded = image.copy() if image.mode in ("L", "RGB", "RGBA") else image.convert("RGB")
    gray = to_gray(decoded)
    counts = histogram(gray)
    otsu, variances = otsu_threshold(counts)
    fixed_image, otsu_image = threshold_image(gray, threshold), threshold_image(gray, otsu)
    directory.mkdir(parents=True, exist_ok=True)
    gray.save(directory / "gray.png")
    fixed_image.save(directory / f"fixed_T{threshold}.png")
    otsu_image.save(directory / f"otsu_T{otsu}.png")
    marked_histogram(counts, threshold, otsu).save(directory / "histogram.png")
    with (directory / "histogram.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bin", "count", "between_class_variance_if_T"))
        for level in range(256):
            writer.writerow((level, counts[level], f"{float(variances[level]):.6f}"))
    width, height = gray.size
    preview = Image.new("L", (width * 3, height + 24), 255)
    draw = ImageDraw.Draw(preview)
    for column, (title, image) in enumerate((("Gray", gray), (f"Fixed T={threshold}", fixed_image),
                                             (f"Otsu T={otsu}", otsu_image))):
        preview.paste(image, (column * width, 24))
        draw.text((column * width + 8, 5), title, fill=0)
    preview.save(directory / "comparison.png")
    report = dict(source=str(source), size=[width, height], pixels=width * height,
                  gray_formula="round_half_up(0.299R+0.587G+0.114B)", rule="pixel >= T -> 255",
                  fixed_threshold=threshold, otsu_threshold=otsu,
                  otsu_between_class_variance=float(variances[otsu]),
                  fixed_white_pixels=histogram(fixed_image)[255], otsu_white_pixels=histogram(otsu_image)[255])
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="不改動來源圖片")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/image-processing/binarization")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="固定門檻，pixel >= T 為白")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 256:
        parser.error("--threshold 必須介於0至256")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("輸出資料夾必須位於本專案內")
    sources = args.inputs or [ROOT / "新手大禮包/旋積運算/Lenna_RGB.tif"]
    for source in sources:
        if source.resolve().is_relative_to((output / source.stem).resolve()):
            parser.error("來源不可放在產出資料夾內，以避免覆寫")
    for source in sources:
        report = process(source, output / source.stem, args.threshold)
        print(f"已輸出：{output / source.stem}（Otsu T={report['otsu_threshold']}）")


if __name__ == "__main__":
    main()
