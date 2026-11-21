"""手算像素、Otsu 與邊界；不匯入 NumPy 或啟動模型。"""
import ast
from pathlib import Path
import pytest
from PIL import Image
from lib.opencv2 import binarization as b


def gray(values):
    image = Image.new('L', (len(values), 1))
    for x, v in enumerate(values):
        image.putpixel((x, 0), v)
    return image


def row(image):
    return [image.getpixel((x, 0)) for x in range(image.width)]


def test_manual_gray_known_pixels():
    image = Image.new('RGB', (4, 1))
    for x, c in enumerate(((0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255))):
        image.putpixel((x, 0), c)
    assert row(b.to_gray(image)) == [0, 255, 76, 29]


def test_fixed_threshold_is_strictly_greater():
    assert row(b.binarize(gray([0, 127, 128, 255]), 127)) == [0, 0, 255, 255]
    assert row(b.binarize(gray([0, 127, 128, 255]), 255)) == [0, 0, 0, 0]
    assert row(b.binarize(gray([0, 127, 128, 255]), 0)) == [0, 255, 255, 255]


def test_otsu_separates_two_clusters():
    image = gray([10] * 5 + [200] * 5)
    t = b.otsu_threshold(b.histogram(image))
    assert 10 <= t < 200
    assert row(b.binarize(image, t)) == [0] * 5 + [255] * 5


def test_otsu_constant_image_returns_that_level():
    counts = b.histogram(gray([90, 90, 90]))
    assert b.otsu_threshold(counts) == 90
    assert row(b.binarize(gray([90, 90, 90]), 90)) == [0, 0, 0]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        b.binarize(gray([1]), 256)
    with pytest.raises(ValueError):
        b.binarize(Image.new('RGB', (1, 1)), 10)
    with pytest.raises(ValueError):
        b.otsu_threshold([0] * 256)
    with pytest.raises(ValueError):
        b.otsu_threshold([1] * 255)


def test_process_writes_outputs(tmp_path):
    src = tmp_path / 'a.png'
    img = Image.new('RGB', (4, 1))
    for x, v in enumerate((10, 20, 200, 220)):
        img.putpixel((x, 0), (v, v, v))
    img.save(src)
    report = b.process(src, tmp_path / 'out', 100)
    assert report['results']['fixed']['white_pixels'] == 2
    assert report['results']['otsu']['white_pixels'] == 2
    assert (tmp_path / 'out' / 'a_report.json').exists()


def test_no_numpy_or_cv2_imports():
    tree = ast.parse(Path(b.__file__).read_text(encoding='utf-8'))
    names = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
             for a in (n.names if isinstance(n, ast.Import) else [ast.alias(name=n.module or '')])}
    assert not names & {'numpy', 'cv2', 'scipy', 'skimage'}
