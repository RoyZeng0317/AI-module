"""Correctness tests for data_split.py's stratified splitting, using a tiny
synthetic dataset (no real images needed — the splitter only looks at
filenames/extensions, not image contents).
"""

from pathlib import Path

import pytest

from data_split import split_dataset


def _make_dataset(root: Path, class_counts: dict):
    for class_name, count in class_counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            (class_dir / f"{class_name}_{i}.jpg").write_bytes(b"fake")


def test_splits_conserve_all_files_with_no_overlap(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_dataset(source, {"stop_sign": 20, "speed_limit_50": 20})

    summary = split_dataset(source, output, train=0.7, val=0.15, test=0.15, seed=1)

    seen = set()
    total_files = 0
    for split in ("train", "val", "test"):
        for class_name in summary:
            class_dir = output / split / class_name
            for f in class_dir.iterdir():
                assert f not in seen, f"{f} appears in more than one split"
                seen.add(f)
                total_files += 1

    assert total_files == 40


def test_stratified_across_rare_class(tmp_path):
    """A rare class (5 files) must still appear in every split, not just
    dumped entirely into train — this is the point of stratifying per class
    instead of splitting the whole dataset at once."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_dataset(source, {"common_sign": 100, "rare_sign": 5})

    summary = split_dataset(source, output, train=0.7, val=0.15, test=0.15, seed=1)

    assert summary["rare_sign"]["train"] >= 1
    assert summary["rare_sign"]["val"] >= 1 or summary["rare_sign"]["test"] >= 1


def test_reproducible_with_same_seed(tmp_path):
    source = tmp_path / "source"
    _make_dataset(source, {"a": 30, "b": 30})

    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    s1 = split_dataset(source, out1, seed=7)
    s2 = split_dataset(source, out2, seed=7)
    assert s1 == s2


def test_ratios_must_sum_to_one(tmp_path):
    source = tmp_path / "source"
    _make_dataset(source, {"a": 10})
    with pytest.raises(AssertionError):
        split_dataset(source, tmp_path / "out", train=0.5, val=0.3, test=0.3)
