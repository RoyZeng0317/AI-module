"""以 for 迴圈做灰階二值化（固定門檻與 Otsu 自動門檻）；不使用 NumPy／OpenCV。"""
import argparse, csv, json
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]


def to_gray(image):
    """手動灰階 0.299R+0.587G+0.114B（四捨五入），不呼叫內建 convert('L')。"""
    if image.mode != 'RGB' or not image.width or not image.height:
        raise ValueError('輸入必須是非空 RGB 圖片')
    result = Image.new('L', image.size)
    source, output = image.load(), result.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = source[x, y]
            output[x, y] = max(0, min(255, int(.299*r + .587*g + .114*b + .5)))
    return result


def histogram(image):
    if image.mode != 'L' or not image.width or not image.height:
        raise ValueError('直方圖輸入必須是非空灰階圖片')
    counts, pixels = [0] * 256, image.load()
    for y in range(image.height):
        for x in range(image.width):
            counts[pixels[x, y]] += 1
    return counts


def otsu_threshold(counts):
    """最大化類間變異數；灰階 <= t 為背景、> t 為前景。單一灰階回傳該灰階值。"""
    if len(counts) != 256 or any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError('需要256個非負整數bin，且總像素數須大於0')
    total = sum(counts)
    sum_all = sum(level * n for level, n in enumerate(counts))
    weight_b = sum_b = 0
    best, best_var = next(i for i, n in enumerate(counts) if n), -1.0
    for level in range(256):
        weight_b += counts[level]
        if not weight_b:
            continue
        weight_f = total - weight_b
        if not weight_f:
            break
        sum_b += level * counts[level]
        mean_b, mean_f = sum_b / weight_b, (sum_all - sum_b) / weight_f
        variance = weight_b * weight_f * (mean_b - mean_f) ** 2
        if variance > best_var:  # 嚴格大於：多個最佳門檻時取最小者
            best, best_var = level, variance
    return best


def binarize(image, threshold):
    """灰階 > threshold 為 255，否則 0。"""
    if image.mode != 'L' or not image.width or not image.height:
        raise ValueError('輸入必須是非空灰階圖片')
    if type(threshold) is not int or not 0 <= threshold <= 255:
        raise ValueError('門檻必須是 0 至 255 的整數')
    result = Image.new('L', image.size)
    source, output = image.load(), result.load()
    for y in range(image.height):
        for x in range(image.width):
            output[x, y] = 255 if source[x, y] > threshold else 0
    return result


def count_white(image):
    pixels, total = image.load(), 0
    for y in range(image.height):
        for x in range(image.width):
            total += pixels[x, y] == 255
    return total


def process(source, directory, threshold=127):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    with Image.open(source) as image:
        gray = to_gray(image.convert('RGB'))
    counts = histogram(gray)
    otsu = otsu_threshold(counts)
    directory.mkdir(parents=True, exist_ok=True)
    gray.save(directory / f'{source.stem}_gray.png')
    report = dict(source=str(source), size=list(gray.size), pixels=gray.width*gray.height, results={})
    for name, value in (('fixed', threshold), ('otsu', otsu)):
        result = binarize(gray, value)
        result.save(directory / f'{source.stem}_{name}_{value}.png')
        report['results'][name] = dict(threshold=value, white_pixels=count_white(result))
    with (directory / f'{source.stem}_histogram.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('gray_level', 'count'))
        writer.writerows(enumerate(counts))
    (directory / f'{source.stem}_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path,
                        default=PROJECT_DIR/'新手大禮包/直方圖均衡化/Lenna_RGB.tif')
    parser.add_argument('--threshold', type=int, default=127, help='固定門檻（0-255）')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR/'output/image-processing/binarization')
    args = parser.parse_args()
    print(json.dumps(process(args.source, args.output, args.threshold), ensure_ascii=False, indent=2))
