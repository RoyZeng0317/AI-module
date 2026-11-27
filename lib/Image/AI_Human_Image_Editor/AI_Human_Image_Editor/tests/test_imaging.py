import numpy as np
from PIL import Image
import pytest

from ai_human_image_editor.imaging import (
    enhance_image, inpaint_image, image_info, ImageEditorError
)


def test_enhance_keeps_original_dimensions(tmp_path):
    src = tmp_path / "input.png"
    dst = tmp_path / "output.png"
    Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8)).save(src)

    enhance_image(src, dst, strength=0, sharpen=0)
    with Image.open(dst) as out:
        assert out.size == (120, 80)

def test_inpaint_rejects_wrong_mask_size(tmp_path):
    src = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    dst = tmp_path / "output.png"
    Image.fromarray(np.zeros((50, 60, 3), dtype=np.uint8)).save(src)
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(mask)  

    with pytest.raises(ImageEditorError):
        inpaint_image(src, mask, dst)


def test_inpaint_rejects_empty_mask(tmp_path):
    src = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    dst = tmp_path / "output.png"
    Image.fromarray(np.full((50, 60, 3), 120, dtype=np.uint8)).save(src)
    Image.fromarray(np.zeros((50, 60), dtype=np.uint8)).save(mask)

    with pytest.raises(ImageEditorError):
        inpaint_image(src, mask, dst)


def test_image_info(tmp_path):
    src = tmp_path / "input.png"
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(src)
    info = image_info(src)
    assert info["width"] == 48
    assert info["height"] == 32
