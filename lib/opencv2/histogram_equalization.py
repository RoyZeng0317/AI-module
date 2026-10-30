"""以 for 迴圈計算 Y/Cb/Cr、Y 直方圖與均衡化；不使用 NumPy。"""
import argparse, csv, json
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[2]
COEFFICIENTS = ((16, 65.738, 129.057, 25.064),
                (128, -37.945, -74.494, 112.439),
                (128, 112.439, -94.154, -18.285))


def byte(value):
    return max(0, min(255, int(value + .5)))


def rgb_to_ycbcr(image):
    """練習文件指定的有限範圍公式；各通道以灰階容器呈現。"""
    if image.mode != 'RGB' or not image.width or not image.height:
        raise ValueError('輸入必須是非空 RGB 圖片')
    channels = [Image.new('L', image.size) for _ in range(3)]
    source, outputs = image.load(), [channel.load() for channel in channels]
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = source[x, y]
            for pixels, (offset, cr, cg, cb) in zip(outputs, COEFFICIENTS):
                pixels[x, y] = byte(offset + (cr*r + cg*g + cb*b) / 256)
    return tuple(channels)


def histogram(image):
    if image.mode != 'L' or not image.width or not image.height:
        raise ValueError('直方圖輸入必須是非空灰階圖片')
    counts, pixels = [0] * 256, image.load()
    for y in range(image.height):
        for x in range(image.width):
            counts[pixels[x, y]] += 1
    return counts


def equalization_table(counts):
    if len(counts) != 256 or any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError('需要256個非負整數bin，且總像素數須大於0')
    cdf, total = [], 0
    for count in counts:
        total += count
        cdf.append(total)
    first = next(value for value in cdf if value)
    denominator = total - first
    lut = []
    for value in cdf:
        # 整數四捨五入；單一灰階時保留原圖，避免除以0。
        mapped = ((value-first)*255*2 + denominator) // (2*denominator) if denominator else len(lut)
        lut.append(max(0, min(255, mapped)))
    return cdf, lut


def apply_table(image, lut):
    if image.mode != 'L' or len(lut) != 256 or any(type(n) is not int or not 0 <= n <= 255 for n in lut):
        raise ValueError('需要灰階圖片及256個0至255的整數映射值')
    result = Image.new('L', image.size)
    source, output = image.load(), result.load()
    for y in range(image.height):
        for x in range(image.width):
            output[x, y] = lut[source[x, y]]
    return result


def histogram_chart(counts, title):
    canvas = Image.new('RGB', (576, 320), 'white')
    draw, peak = ImageDraw.Draw(canvas), max(counts)
    draw.text((32, 8), title, fill='black')
    draw.line((32, 280, 544, 280), fill='black')
    for level, count in enumerate(counts):
        height = count * 235 / peak if peak else 0
        draw.line((32+level*2, 279, 32+level*2, 279-height), fill='#3070b0', width=2)
    for level in (0, 64, 128, 192, 255):
        draw.text((32+level*2-5, 287), str(level), fill='black')
    draw.text((32, 304), f'peak={peak}; pixels={sum(counts)}', fill='black')
    return canvas


def process(source, directory):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    with Image.open(source) as image:
        # RGB模式解碼，不呼叫內建灰階／YCbCr／均衡化轉換。
        rgb = image.convert('RGB')
    y, cb, cr = rgb_to_ycbcr(rgb)
    before = histogram(y)
    cdf, lut = equalization_table(before)
    equalized = apply_table(y, lut)
    after = histogram(equalized)
    directory.mkdir(parents=True, exist_ok=True)
    images = {'Y': y, 'Cb': cb, 'Cr': cr, 'Y_equalized': equalized,
              'Y_histogram': histogram_chart(before, 'Y histogram before'),
              'Y_equalized_histogram': histogram_chart(after, 'Y histogram after')}
    for name, image in images.items():
        image.save(directory / f'{source.stem}_{name}.png')
    with (directory / f'{source.stem}_histogram.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('gray_level', 'before_count', 'cdf_count', 'lut', 'after_count'))
        for level in range(256):
            writer.writerow((level, before[level], cdf[level], lut[level], after[level]))
    preview = Image.new('RGB', (rgb.width*3, rgb.height+24), 'white')
    draw = ImageDraw.Draw(preview)
    for column, (title, image) in enumerate((('RGB source', rgb), ('Y (limited range)', y), ('Y equalized', equalized))):
        preview.paste(image, (column*rgb.width, 24))
        draw.text((column*rgb.width+8, 5), title, fill='black')
    preview.save(directory / f'{source.stem}_comparison.png')
    report = dict(source=str(source), size=list(rgb.size), pixels=rgb.width*rgb.height,
                  y_range=[next(i for i,n in enumerate(before) if n), max(i for i,n in enumerate(before) if n)],
                  equalized_range=[next(i for i,n in enumerate(after) if n), max(i for i,n in enumerate(after) if n)],
                  mapping='CDF-min; round-half-up; constant images unchanged', coefficients=COEFFICIENTS)
    (directory / f'{source.stem}_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path,
                        default=PROJECT_DIR/'新手大禮包/直方圖均衡化/Lenna_RGB.tif')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR/'output/image-processing/histogram')
    args = parser.parse_args()
    print(json.dumps(process(args.source, args.output), ensure_ascii=False, indent=2))
