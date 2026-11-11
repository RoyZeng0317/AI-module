"""影像分類用的分層＋分組 k-fold 分割（不需要 torch）。

一般 k-fold 只把檔案隨機切成 k 份；但若同一張照片的近似版本（重新壓縮、
連拍、資料增強）一份在 train fold、一份在 val fold，驗證分數就會被洩漏灌水。
這裡先用 leakage_check.build_groups() 把「應該視為同一張圖」的檔案歸成群組，
再以群組為單位分配 fold：

  - 分組（group）：同一群組一定落在同一個 fold，所以 train/val 之間不會洩漏
  - 分層（stratified）：每個類別的檔案盡量平均分到各 fold，罕見類別每個 fold
    都分得到

分配採貪婪法：群組由大到小（同大小時以 seed 洗牌），逐一放進「該類別目前
最少」的 fold，平手再看 fold 總數。群組數少於 k 的類別無法每個 fold 都有，
會在 warnings 裡回報，不會硬拆群組。

用法（只看分配結果，不訓練；訓練請用 image_classifier_bnn.py --kfold）：
    python kfold.py --source path/to/dataset --k 5
    python kfold.py --source path/to/dataset --k 5 --group-regex "^(.+?)_aug\\d+" --json folds.json
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from leakage_check import DEFAULT_THRESHOLD, build_groups, list_images


def stratified_group_kfold(labels: list[str], groups: list[int], k: int, seed: int = 42) -> tuple[list[int], list[str]]:
    """回傳 (每個樣本的 fold 編號, 警告訊息)。同群組必同 fold。"""
    if k < 2:
        raise ValueError("k 必須 >= 2")
    if len(labels) != len(groups):
        raise ValueError("labels 與 groups 長度不同")

    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    if len(members) < k:
        raise ValueError(f"只有 {len(members)} 個獨立群組，不足以切成 k={k} 份")

    group_ids = list(members)
    random.Random(seed).shuffle(group_ids)
    group_ids.sort(key=lambda g: len(members[g]), reverse=True)  # sort 是穩定的，同大小維持洗牌順序

    fold_class = [Counter() for _ in range(k)]
    fold_total = [0] * k
    group_fold: dict[int, int] = {}
    for g in group_ids:
        cls_counts = Counter(labels[i] for i in members[g])
        main_cls = cls_counts.most_common(1)[0][0]
        best = min(range(k), key=lambda f: (fold_class[f][main_cls], fold_total[f], f))
        group_fold[g] = best
        fold_class[best].update(cls_counts)
        fold_total[best] += len(members[g])

    warnings = []
    for cls in sorted(set(labels)):
        empty = [f for f in range(k) if fold_class[f][cls] == 0]
        if empty:
            warnings.append(f"類別 {cls} 的獨立群組不足 {k} 個，fold {empty} 的驗證集沒有這個類別")
    mixed = [g for g in members if len({labels[i] for i in members[g]}) > 1]
    if mixed:
        warnings.append(f"{len(mixed)} 個群組同時含有不同類別（可能是標註衝突或閾值太寬），請用 leakage_check.py 檢查")

    return [group_fold[g] for g in groups], warnings


def make_folds(source: Path, k: int = 5, seed: int = 42, threshold: int = DEFAULT_THRESHOLD,
               group_regex: str | None = None) -> dict:
    """掃描 `source/<class>/*`，回傳 {"classes", "items": [{path, label, group, fold}], "warnings"}。"""
    items = list_images(Path(source))
    if not items:
        raise ValueError(f"{source} 底下找不到類別資料夾或圖片")
    paths = [p for p, _ in items]
    labels = [c for _, c in items]
    groups = build_groups(paths, threshold=threshold, group_regex=group_regex)
    folds, warnings = stratified_group_kfold(labels, groups, k, seed)
    return {
        "classes": sorted(set(labels)),
        "k": k,
        "items": [{"path": str(p), "label": c, "group": g, "fold": f}
                  for p, c, g, f in zip(paths, labels, groups, folds)],
        "warnings": warnings,
    }


def fold_split(folds: dict, fold: int) -> tuple[list[dict], list[dict]]:
    """第 `fold` 份當驗證集、其餘當訓練集。"""
    train = [it for it in folds["items"] if it["fold"] != fold]
    val = [it for it in folds["items"] if it["fold"] == fold]
    return train, val


def summarize(folds: dict) -> list[dict]:
    out = []
    for f in range(folds["k"]):
        _, val = fold_split(folds, f)
        out.append({"fold": f, "n": len(val), "per_class": dict(sorted(Counter(it["label"] for it in val).items()))})
    return out


def main():
    parser = argparse.ArgumentParser(description="分層＋分組 k-fold 分配（避免近似重複跨 fold 洩漏）")
    parser.add_argument("--source", type=Path, required=True, help="資料集根目錄，每個類別一個子資料夾")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="dHash 近似門檻；-1 關閉")
    parser.add_argument("--group-regex", default=None)
    parser.add_argument("--json", type=Path, help="另存分配結果 JSON")
    args = parser.parse_args()

    folds = make_folds(args.source, args.k, args.seed, args.threshold, args.group_regex)
    n_groups = len({it["group"] for it in folds["items"]})
    print(f"{len(folds['items'])} 張圖、{n_groups} 個群組、{len(folds['classes'])} 類")
    for row in summarize(folds):
        print(f"fold {row['fold']}: {row['n']} 張  {row['per_class']}")
    for w in folds["warnings"]:
        print(f"⚠️ {w}")
    if args.json:
        args.json.write_text(json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
