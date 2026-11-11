"""leakage_check.py 測試：用隨機雜訊圖片（每張 dHash 不同）製造各種洩漏情境，
確認檢查器抓得到，乾淨分割則回報 0。只驗證偵測邏輯，不代表能抓到所有洩漏形式。
"""

import shutil

import numpy as np
from PIL import Image

from leakage_check import build_groups, check_split_leakage, dhash, hamming


def _noise_image(path, seed, size=64):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def _clean_split(root):
    seed = 0
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        for cls in ("cat", "dog"):
            for i in range(n):
                seed += 1
                _noise_image(root / split / cls / f"{cls}_{split}_{i}.png", seed)


def test_clean_split_reports_no_leakage(tmp_path):
    _clean_split(tmp_path)
    report = check_split_leakage(tmp_path)
    assert report["leak_count"] == 0
    assert report["counts"] == {"train": 12, "val": 4, "test": 4}


def test_exact_duplicate_across_splits_detected(tmp_path):
    _clean_split(tmp_path)
    shutil.copy(tmp_path / "train/cat/cat_train_0.png", tmp_path / "val/cat/renamed.png")
    report = check_split_leakage(tmp_path)
    assert len(report["exact_duplicates"]) == 1
    assert report["near_duplicates"] == []  # 已列為完全相同，不重複計算
    assert report["leak_count"] == 1


def test_resized_recompressed_copy_is_near_duplicate(tmp_path):
    _clean_split(tmp_path)
    with Image.open(tmp_path / "train/dog/dog_train_1.png") as img:
        img.resize((128, 128)).save(tmp_path / "test/dog/upscaled.jpg", quality=80)
    report = check_split_leakage(tmp_path)
    assert report["exact_duplicates"] == []
    assert len(report["near_duplicates"]) == 1
    assert report["near_duplicates"][0]["distance"] <= 5


def test_same_image_in_two_classes_is_label_conflict(tmp_path):
    _clean_split(tmp_path)
    shutil.copy(tmp_path / "train/cat/cat_train_2.png", tmp_path / "train/dog/oops.png")
    report = check_split_leakage(tmp_path)
    assert len(report["label_conflicts"]) == 1
    assert report["leak_count"] == 0  # 同一個 split 內，不算跨 split 洩漏


def test_group_regex_leak_detected(tmp_path):
    _noise_image(tmp_path / "train/cat/img001_aug0.png", 1)
    _noise_image(tmp_path / "val/cat/img001_aug1.png", 2)  # 內容不同但同一來源
    _noise_image(tmp_path / "val/cat/img002_aug0.png", 3)
    report = check_split_leakage(tmp_path, group_regex=r"^(.+?)_aug\d+")
    assert [g["group"] for g in report["group_leaks"]] == ["img001"]


def test_dhash_distinguishes_different_images(tmp_path):
    a = dhash(_noise_image(tmp_path / "a.png", 1))
    b = dhash(_noise_image(tmp_path / "b.png", 2))
    assert hamming(a, a) == 0
    assert hamming(a, b) > 5


def test_build_groups_merges_duplicates_only(tmp_path):
    p = [_noise_image(tmp_path / f"{i}.png", i) for i in range(4)]
    shutil.copy(p[0], tmp_path / "dup.png")
    groups = build_groups(p + [tmp_path / "dup.png"])
    assert groups[0] == groups[4]
    assert len(set(groups)) == 4
