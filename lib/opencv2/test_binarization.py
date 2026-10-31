"""手算門檻邊界、Otsu類間變異數與常數圖；不匯入NumPy或啟動模型。"""
import ast, csv
from fractions import Fraction
from pathlib import Path
import pytest
from PIL import Image
from lib.opencv2 import binarization as processing


def pixels(image):
    return [[image.getpixel((x, y)) for x in range(image.width)] for y in range(image.height)]


def counts_of(matrix):
    return processing.histogram(processing.matrix_to_image(matrix))


def test_manual_gray_round_half_up():
    assert processing.gray_value(20, 80, 200) == 76  # 75.76
    assert processing.gray_value(255, 255, 255) == 255
    assert processing.gray_value(1, 0, 1) == 0  # 0.413
    assert processing.gray_value(0, 0, 5) == 1  # 0.57
    image = Image.new("RGB", (1, 1), (100, 150, 200))
    assert pixels(processing.to_gray(image)) == [[141]]  # 29.9+88.05+22.8=140.75


def test_fixed_threshold_boundary_is_inclusive():
    gray = processing.matrix_to_image([[127, 128, 129], [0, 255, 128]])
    assert pixels(processing.threshold_image(gray, 128)) == [[0, 255, 255], [0, 255, 255]]
    assert pixels(processing.threshold_image(gray, 0)) == [[255] * 3] * 2
    assert pixels(processing.threshold_image(gray, 256)) == [[0] * 3] * 2
    with pytest.raises(ValueError):
        processing.threshold_image(gray, 257)


def test_otsu_bimodal_hand_computed():
    matrix = [[10, 10, 20], [200, 210, 210]]
    threshold, variances = processing.otsu_threshold(counts_of(matrix))
    # 背景{10,10,20}、前景{200,210,210}；T=21..200 同分，取最小 T=21。
    assert 20 < threshold <= 200 and threshold == 21
    # N=6, S=660, n0=3, S0=40 → (40·6-660·3)²/(3·3·36) = 1740²/324
    assert variances[21] == Fraction(1740 ** 2, 324) == variances[200]
    assert variances[20] < variances[21] and variances[201] < variances[21]
    gray = processing.matrix_to_image(matrix)
    assert pixels(processing.threshold_image(gray, threshold)) == [[0, 0, 0], [255, 255, 255]]


def test_otsu_matches_direct_variance_definition():
    values = [3, 3, 4, 7, 50, 52, 52, 90, 91, 91, 200, 201]
    threshold, variances = processing.otsu_threshold(counts_of([values]))
    for t in range(256):
        low = [v for v in values if v < t]
        high = [v for v in values if v >= t]
        if not low or not high:
            assert variances[t] == 0
            continue
        w0, w1 = Fraction(len(low), len(values)), Fraction(len(high), len(values))
        diff = Fraction(sum(low), len(low)) - Fraction(sum(high), len(high))
        assert variances[t] == w0 * w1 * diff * diff
    assert threshold == min(t for t in range(256) if variances[t] == max(variances))


@pytest.mark.parametrize("value", (0, 77, 255))
def test_constant_image_defined_result(value):
    threshold, variances = processing.otsu_threshold(counts_of([[value] * 3] * 2))
    assert threshold == 0 and all(v == 0 for v in variances)
    gray = processing.matrix_to_image([[value] * 3] * 2)
    assert pixels(processing.threshold_image(gray, threshold)) == [[255] * 3] * 2


def test_invalid_inputs():
    for matrix in ([], [[]], [[1], [2, 3]], [[256]], [[-1]], [[1.5]]):
        with pytest.raises(ValueError):
            processing.matrix_to_image(matrix)
    for counts in ([0] * 256, [1] * 255, [-1] + [1] * 255):
        with pytest.raises(ValueError):
            processing.otsu_threshold(counts)
    with pytest.raises(ValueError):
        processing.to_gray(Image.new("RGB", (0, 0)))


def test_end_to_end_outputs_keep_source(tmp_path):
    path = tmp_path / "tiny.png"
    image = Image.new("RGB", (4, 2))
    colors = [(10, 10, 10), (30, 30, 30), (220, 220, 220), (240, 240, 240)]
    for y in range(2):
        for x, color in enumerate(colors):
            image.putpixel((x, y), color)
    image.save(path)
    original = path.read_bytes()
    report = processing.process(path, tmp_path / "out" / path.stem, threshold=100)
    folder = tmp_path / "out" / "tiny"
    assert path.read_bytes() == original
    assert report["otsu_threshold"] == 31 and report["fixed_threshold"] == 100
    for name in ("gray.png", "fixed_T100.png", "otsu_T31.png", "histogram.png",
                 "histogram.csv", "comparison.png", "report.json"):
        assert (folder / name).is_file()
    with Image.open(folder / "otsu_T31.png") as result:
        assert pixels(result) == [[0, 0, 255, 255]] * 2
    with Image.open(folder / "comparison.png") as preview:
        assert preview.size == (12, 26)
    with (folder / "histogram.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 256 and rows[10]["count"] == "2" and rows[220]["count"] == "2"
    assert float(rows[31]["between_class_variance_if_T"]) == pytest.approx(report["otsu_between_class_variance"])


def test_no_array_library_or_builtin_threshold():
    tree = ast.parse(Path(processing.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(item.name.split(".")[0] not in ("numpy", "cv2", "scipy") for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in ("numpy", "cv2", "scipy")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("point", "filter", "threshold", "eval", "quantize")
            if node.func.attr == "convert":
                assert [ast.unparse(a) for a in node.args] == ["'RGB'"]
