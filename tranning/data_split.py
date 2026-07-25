"""Stratified train/validation/test dataset splitter.

Expects a directory of images arranged one subfolder per class (the
torchvision ImageFolder layout) — this is the layout the road-sign
recognition dataset (CLAUDE.md to-do #4) will use:

    dataset/
        stop_sign/
            a.jpg
            b.jpg
        speed_limit_50/
            c.jpg
            ...

Splits each class independently (stratified) so every split ends up with a
representative mix of classes even when some classes are rare — important
for a road-sign dataset where common signs vastly outnumber rare ones.
Output layout:

    output/
        train/<class>/...
        val/<class>/...
        test/<class>/...

Usage:
    python data_split.py --source dataset --output output
    python data_split.py --source dataset --output output --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_classes(source: Path) -> list:
    return sorted(p for p in source.iterdir() if p.is_dir())


def split_files(files: list, train: float, val: float, test: float, seed: int) -> dict:
    assert abs(train + val + test - 1.0) < 1e-6, "train+val+test must sum to 1.0"
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train)
    n_val = round(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def split_dataset(source: Path, output: Path, train: float = 0.7, val: float = 0.15,
                   test: float = 0.15, seed: int = 42, move: bool = False) -> dict:
    """Split every class subfolder under `source` into train/val/test under
    `output`, using the same ratios per class (stratified). Returns a
    {class_name: {split_name: file_count}} summary.
    """
    classes = list_classes(source)
    if not classes:
        raise ValueError(f"No class subfolders found under {source}")

    summary = {}
    for class_dir in classes:
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue
        parts = split_files(files, train, val, test, seed)
        summary[class_dir.name] = {}
        for split_name, split_files_ in parts.items():
            dest_dir = output / split_name / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files_:
                dest = dest_dir / f.name
                (shutil.move if move else shutil.copy2)(str(f), str(dest))
            summary[class_dir.name][split_name] = len(split_files_)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test dataset splitter")
    parser.add_argument("--source", required=True, help="dataset root, one subfolder per class")
    parser.add_argument("--output", required=True, help="where to write train/val/test folders")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="move files instead of copying")
    args = parser.parse_args()

    summary = split_dataset(Path(args.source), Path(args.output), args.train, args.val,
                             args.test, args.seed, args.move)
    total = {"train": 0, "val": 0, "test": 0}
    for cls, counts in summary.items():
        print(f"{cls}: {counts}")
        for k, v in counts.items():
            total[k] += v
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
