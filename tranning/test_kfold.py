"""kfold.py 測試：確認同群組不跨 fold、各類別分層平均、群組不足時會警告。"""

import shutil
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from kfold import fold_split, make_folds, stratified_group_kfold


def _dataset(root, per_class=10, classes=("cat", "dog", "bird")):
    seed = 0
    for cls in classes:
        for i in range(per_class):
            seed += 1
            rng = np.random.default_rng(seed)
            (root / cls).mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(root / cls / f"{i}.png")


def test_folds_are_stratified_and_cover_every_item(tmp_path):
    _dataset(tmp_path)
    folds = make_folds(tmp_path, k=5)
    assert len(folds["items"]) == 30
    for f in range(5):
        _, val = fold_split(folds, f)
        assert Counter(it["label"] for it in val) == {"cat": 2, "dog": 2, "bird": 2}
    assert folds["warnings"] == []


def test_near_duplicates_never_cross_folds(tmp_path):
    _dataset(tmp_path)
    for i in range(3):  # 每個原圖多一份放大重存版本，應與原圖同 fold
        with Image.open(tmp_path / "cat" / f"{i}.png") as img:
            img.resize((96, 96)).save(tmp_path / "cat" / f"{i}_big.jpg", quality=85)
    folds = make_folds(tmp_path, k=5)
    by_name = {it["path"].split("/")[-1]: it for it in folds["items"] if it["label"] == "cat"}
    for i in range(3):
        assert by_name[f"{i}.png"]["fold"] == by_name[f"{i}_big.jpg"]["fold"]
    for f in range(5):
        train, val = fold_split(folds, f)
        assert not ({it["group"] for it in train} & {it["group"] for it in val})


def test_group_regex_keeps_augmented_copies_together():
    labels = ["a"] * 6 + ["b"] * 6
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    folds, warnings = stratified_group_kfold(labels, groups, k=3)
    for g in set(groups):
        assert len({f for f, gg in zip(folds, groups) if gg == g}) == 1
    assert warnings == []


def test_rare_class_warns_instead_of_splitting_groups():
    labels = ["common"] * 10 + ["rare"] * 2
    groups = list(range(10)) + [10, 10]  # rare 只有一個群組
    _, warnings = stratified_group_kfold(labels, groups, k=5)
    assert any("rare" in w for w in warnings)


def test_too_few_groups_raises():
    with pytest.raises(ValueError):
        stratified_group_kfold(["a", "a"], [0, 0], k=2)
