"""手算像素、CDF與常數邊界；不匯入NumPy或啟動模型。"""
import ast, csv
from pathlib import Path
import pytest
from PIL import Image
from lib.opencv2 import histogram_equalization as processing


def gray(values):
    image = Image.new('L', (len(values), 1))
    for x, value in enumerate(values):
        image.putpixel((x, 0), value)
    return image


def test_ycbcr_worksheet_known_pixels():
    image = Image.new('RGB', (5, 1))
    for x, color in enumerate(((0,0,0), (255,255,255), (255,0,0), (0,255,0), (0,0,255))):
        image.putpixel((x,0), color)
    y, cb, cr = processing.rgb_to_ycbcr(image)
    assert [y.getpixel((x,0)) for x in range(5)] == [16,235,81,145,41]
    assert [cb.getpixel((x,0)) for x in range(5)] == [128,128,90,54,240]
    assert [cr.getpixel((x,0)) for x in range(5)] == [128,128,240,34,110]


def test_hand_calculated_histogram_cdf_and_mapping():
    image = gray([10,10,20,30])
    counts = processing.histogram(image)
    cdf, lut = processing.equalization_table(counts)
    assert sum(counts) == 4 and [counts[i] for i in (10,20,30)] == [2,1,1]
    assert [cdf[i] for i in (10,20,30)] == [2,3,4]
    assert [lut[i] for i in (10,20,30)] == [0,128,255]
    result = processing.apply_table(image, lut)
    assert [result.getpixel((x,0)) for x in range(4)] == [0,0,128,255]
    assert [image.getpixel((x,0)) for x in range(4)] == [10,10,20,30]
    assert all(a <= b for a,b in zip(lut,lut[1:]))


@pytest.mark.parametrize('value', (0,16,128,255))
def test_constant_image_is_unchanged(value):
    image = gray([value]*4)
    _, lut = processing.equalization_table(processing.histogram(image))
    result = processing.apply_table(image,lut)
    assert [result.getpixel((x,0)) for x in range(4)] == [value]*4


def test_invalid_inputs():
    with pytest.raises(ValueError): processing.rgb_to_ycbcr(gray([10]))
    with pytest.raises(ValueError): processing.histogram(Image.new('RGB',(1,1)))
    with pytest.raises(ValueError): processing.histogram(Image.new('L',(0,0)))
    for counts in ([0]*256, [1]*255, [-1]+[1]*255):
        with pytest.raises(ValueError): processing.equalization_table(counts)


def test_real_io_chart_and_csv_keep_source(tmp_path):
    path = tmp_path/'rgb.png'
    image=Image.new('RGB',(2,1));image.putpixel((0,0),(0,0,0));image.putpixel((1,0),(255,255,255));image.save(path)
    original=path.read_bytes()
    report=processing.process(path,tmp_path/'result')
    assert path.read_bytes() == original and report['size'] == [2,1]
    assert report['y_range'] == [16,235] and report['equalized_range'] == [0,255]
    with (tmp_path/'result/rgb_histogram.csv').open(encoding='utf-8') as stream:
        assert len(list(csv.DictReader(stream))) == 256
    with Image.open(tmp_path/'result/rgb_comparison.png') as preview:
        assert preview.size == (6,25)


def test_no_numpy_or_builtin_equalization():
    tree=ast.parse(Path(processing.__file__).read_text(encoding='utf-8'))
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'cv2' not in ast.unparse(n) for n in imports)
    names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(names) & {'equalize','equalizeHist','histogram','filter'}
