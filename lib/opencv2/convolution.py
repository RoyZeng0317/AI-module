"""以 Python for 迴圈計算真正卷積；Pillow 僅負責影像讀寫。

卷積先將核旋轉 180 度，因此 Sobel 符號與直接套核的互相關相反。
灰階轉換是逐像素色彩運算，並非空間卷積。原始結果不裁切、不取整。
"""
import argparse, json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MATRIX_A = [
    [250, 231, 255, 253, 251, 253, 254],
    [251, 251, 255, 253, 252, 255, 255],
    [251, 254, 250, 231, 255, 253, 255],
    [253, 252, 255, 255, 254, 250, 231],
    [255, 253, 252, 255, 255, 254, 250],
    [250, 251, 255, 253, 255, 253, 252],
    [255, 253, 251, 253, 254, 255, 255],
    [252, 255, 255, 254, 250, 253, 251],
]
FILTERS = {
    "filter1": [[1, 0, 1], [0, 4, 0], [1, 0, 1]],
    "filter2": [[1 / 25 for _ in range(5)] for _ in range(5)],
    "filter3": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
}


def convolve(matrix, kernel, padding="edge"):
    """純迴圈單通道卷積；edge/zero 保持尺寸，valid 不補邊。"""
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("影像矩陣必須非空且為矩形")
    if (not kernel or not kernel[0] or len(kernel) % 2 == 0
            or len(kernel[0]) % 2 == 0
            or any(len(row) != len(kernel[0]) for row in kernel)):
        raise ValueError("卷積核必須非空、矩形，且長寬為奇數")
    if padding not in ("edge", "zero", "valid"):
        raise ValueError("padding 必須為 edge、zero 或 valid")
    height, width = len(matrix), len(matrix[0])
    kh, kw = len(kernel), len(kernel[0])
    ry, rx = kh // 2, kw // 2
    if padding == "valid" and (height < kh or width < kw):
        raise ValueError("valid 模式的影像不可小於卷積核")
    ys = range(ry, height - ry) if padding == "valid" else range(height)
    xs = range(rx, width - rx) if padding == "valid" else range(width)
    result = []
    for y in ys:
        row = []
        for x in xs:
            value = 0
            for ky in range(kh):
                iy = y + ky - ry
                for kx in range(kw):
                    weight = kernel[kh - 1 - ky][kw - 1 - kx]
                    if weight == 0:
                        continue
                    ix = x + kx - rx
                    if padding == "zero" and not (0 <= iy < height and 0 <= ix < width):
                        continue
                    sy, sx = max(0, min(height - 1, iy)), max(0, min(width - 1, ix))
                    value += matrix[sy][sx] * weight
            row.append(value)
        result.append(row)
    return result


def read_channels(path):
    """逐點讀 RGB；灰階輸入複製至三色，透明輸入忽略 alpha。"""
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            raise ValueError(f"不支援 {image.mode} 模式；請提供 L、RGB 或 RGBA")
        channels = [[], [], []]
        pixels = image.load()
        for y in range(image.height):
            rows = [[], [], []]
            for x in range(image.width):
                pixel = pixels[x, y]
                for c in range(3):
                    rows[c].append(pixel if image.mode == "L" else pixel[c])
            for c in range(3):
                channels[c].append(rows[c])
    return channels


def grayscale(channels):
    result = []
    for y in range(len(channels[0])):
        row = []
        for x in range(len(channels[0][0])):
            row.append(.299 * channels[0][y][x] + .587 * channels[1][y][x]
                       + .114 * channels[2][y][x])
        result.append(row)
    return result


def save_display(channels, path, minmax=False):
    """顯示檔才四捨五入；minmax 使用所有通道共同範圍保留相對色彩。"""
    low, high = float("inf"), float("-inf")
    for channel in channels:
        for row in channel:
            for value in row:
                low, high = min(low, value), max(high, value)
    image = Image.new("L" if len(channels) == 1 else "RGB",
                      (len(channels[0][0]), len(channels[0])))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            values = []
            for channel in channels:
                value = channel[y][x]
                if minmax:
                    value = 255 * (value - low) / (high - low) if high > low else 0
                values.append(max(0, min(255, round(value))))
            pixels[x, y] = values[0] if len(values) == 1 else tuple(values)
    image.save(path)


def process_image(source, output, padding="edge"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rgb = read_channels(source)
    gray = grayscale(rgb)
    save_display([gray], output / "gray_original.png")
    for mode, channels in (("rgb", rgb), ("gray", [gray])):
        for name, kernel in FILTERS.items():
            result = []
            for channel in channels:
                result.append(convolve(channel, kernel, padding))
            stem = output / f"{mode}_{name}"
            with stem.with_suffix(".json").open("w", encoding="utf-8") as file:
                json.dump({"padding": padding, "kernel": kernel, "channels": result}, file)
            save_display(result, output / f"{mode}_{name}_clipped.png")
            save_display(result, output / f"{mode}_{name}_minmax.png", minmax=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="不改動來源圖片")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/image-processing/convolution")
    parser.add_argument("--padding", choices=("edge", "zero", "valid"), default="edge")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("輸出資料夾必須位於本專案內")
    sources = args.inputs or [ROOT / "新手大禮包/旋積運算/Lenna_RGB.tif"]
    for source in sources:
        destination = output / source.stem
        if source.resolve().is_relative_to(destination.resolve()):
            parser.error("來源不可放在產出資料夾內，以避免覆寫")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "matrix_valid.json").open("w", encoding="utf-8") as file:
        json.dump({name: convolve(MATRIX_A, kernel, "valid")
                   for name, kernel in FILTERS.items()}, file, indent=2)
    for source in sources:
        process_image(source, output / source.stem, args.padding)
        print(f"已輸出：{output / source.stem}")


if __name__ == "__main__":
    main()
