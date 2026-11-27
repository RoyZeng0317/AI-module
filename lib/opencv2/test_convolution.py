import ast
from pathlib import Path
import pytest
from PIL import Image
from lib.opencv2.convolution import FILTERS, MATRIX_A, convolve, grayscale, read_channels, save_display


def test_true_convolution_flips_asymmetric_kernel():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert convolve(matrix, [[1, 2, 3], [0, 0, 0], [0, 0, 0]], "valid") == [[46]]
    assert convolve(matrix, FILTERS["filter3"], "valid") == [[-8]]


def test_padding_and_raw_range():
    assert convolve([[10]], [[1] * 3 for _ in range(3)], "edge") == [[90]]
    assert convolve([[10]], [[1] * 3 for _ in range(3)], "zero") == [[10]]
    assert convolve([[250] * 3 for _ in range(3)], FILTERS["filter1"], "valid") == [[2000]]
    assert convolve([[3, 8]], [[1]]) == [[3, 8]]
    result = convolve([[10] * 5 for _ in range(5)], FILTERS["filter2"], "valid")
    assert result[0][0] == pytest.approx(10)


@pytest.mark.parametrize("kernel", [[], [[]], [[1, 2]], [[1], [2]], [[1, 2, 3], [1], [1, 2, 3]]])
def test_invalid_kernel(kernel):
    with pytest.raises(ValueError):
        convolve([[1]], kernel)


def test_invalid_inputs():
    for matrix in ([], [[]], [[1], [2, 3]]):
        with pytest.raises(ValueError):
            convolve(matrix, [[1]])
    with pytest.raises(ValueError):
        convolve([[1]], FILTERS["filter1"], "valid")
    with pytest.raises(ValueError):
        convolve([[1]], [[1]], "reflect")


def test_worksheet_dimensions():
    for name, expected in (("filter1", (6, 5)), ("filter2", (4, 3)), ("filter3", (6, 5))):
        result = convolve(MATRIX_A, FILTERS[name], "valid")
        assert (len(result), len(result[0])) == expected


def test_rgb_separate_channels_and_gray(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (1, 1), (20, 80, 200)).save(source)
    channels = read_channels(source)
    result = [convolve(channel, [[2]]) for channel in channels]
    assert result == [[[40]], [[160]], [[400]]]
    assert grayscale(channels)[0][0] == pytest.approx(.299 * 20 + .587 * 80 + .114 * 200)
    display = tmp_path / "display.png"
    save_display(result, display)
    with Image.open(display) as image:
        assert image.getpixel((0, 0)) == (40, 160, 255)
    assert result[2][0][0] == 400


def test_core_has_no_array_or_filter_api():
    source = Path(__file__).with_name("convolution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(item.name.split(".")[0] not in ("numpy", "cv2", "scipy", "matplotlib") for item in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("filter", "convert", "convolve", "filter2D")
