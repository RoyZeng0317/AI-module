"""資料分析與拆分（to-do #53）：讓 sinco 能對一般資料檔做統計分析、train/val/test 拆分與 k-fold。

全部是確定性演算法（只用 Python 標準庫），不交給聊天模型生成數字——17M 的 sinco 會
編造統計值，所以跟 /init、/see 一樣，數字由程式算、模型不參與（Rule 06 也不呼叫外部 API）。

支援的輸入：
  表格  .csv .tsv .json（物件陣列或 {"key": [...]}）.jsonl .xlsx（需 openpyxl）.txt（每行一筆）
  影像  每類一個子資料夾（未分割），或 train/val/test 底下再分類別（已分割）

分析內容（analyze）：
  筆數、欄位型別推斷、缺值、唯一值、數值統計（平均／標準差／四分位數／IQR 離群值）、
  類別前幾名、文字長度、重複列、目標欄位的類別分佈與不平衡、數值欄位相關係數，
  以及自動提醒（缺值多、常數欄、疑似 ID 欄、類別太少無法分層、標籤衝突…）。
拆分（split）：依比例切 train/val/test；可分層（--target）與分組（--group，同組不跨 split）。
  沒給 --group 時，特徵完全相同的列（含重複列）自動視為同組，切完再檢查跨 split 重複列。影像資料夾沿用 data_split.py／leakage_check.py。
k-fold（kfold）：分層＋分組 k-fold 分配（沿用 kfold.stratified_group_kfold），輸出每列的 fold。

用法：
    python data_analysis.py analyze ../data/mental.csv
    python data_analysis.py analyze ../data/pairs.json --target reply
    python data_analysis.py split ../data/mental.csv --target score --ratios 0.7 0.15 0.15
    python data_analysis.py kfold ../data/mental.csv --target score --k 5
    python data_analysis.py analyze ../data/dataset            # 影像資料夾
"""
import argparse
import csv
import io
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "data_analysis"
TABLE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".txt"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
TARGET_HINTS = ("label", "labels", "target", "class", "category", "y", "score", "emotion", "status",
                "標籤", "類別", "分類", "答案")
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}


# ---------------------------------------------------------------- 讀檔

def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp950", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_table(path) -> tuple[list[dict], list[str]]:
    """讀成 (rows, columns)。巢狀值（list/dict）保留原樣，分析時視為 nested 型別。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in TABLE_EXTS:
        raise ValueError(f"不支援的資料格式 {ext}（支援：{'、'.join(sorted(TABLE_EXTS))}，或影像資料夾）")
    if ext == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError("讀 .xlsx 需要 openpyxl（pip install openpyxl），或先另存成 .csv") from exc
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        values = list(sheet.iter_rows(values_only=True))
        if not values:
            return [], []
        columns = [str(c) if c is not None else f"col{i}" for i, c in enumerate(values[0])]
        rows = [{c: ("" if v is None else v) for c, v in zip(columns, row)} for row in values[1:]
                if any(v not in (None, "") for v in row)]
        return rows, columns
    text = _decode(path.read_bytes())
    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else (csv.Sniffer().sniff(text[:4096], ",;\t|").delimiter
                                                if text.strip() else ",")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = [dict(r) for r in reader if any((v or "").strip() for v in r.values())]
        return rows, list(reader.fieldnames or [])
    if ext == ".txt":
        rows = [{"text": line} for line in text.splitlines() if line.strip()]
        return rows, ["text"]
    if ext == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise ValueError("JSON 最外層要是陣列，或只含一個陣列欄位的物件")
            data = lists[0]
    if not isinstance(data, list):
        raise ValueError("JSON 內容不是資料列陣列")
    rows = [r if isinstance(r, dict) else {"value": r} for r in data]
    columns = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    return rows, columns


# ---------------------------------------------------------------- 型別與統計

def _is_missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in MISSING_TOKENS) or \
        (isinstance(v, float) and math.isnan(v))


def _to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?%?", s):
            return float(s.rstrip("%"))
    return None


def _key(v) -> str:
    """可雜湊、可比較的字串表示（巢狀值轉 JSON）。"""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v).strip() if isinstance(v, str) else str(v)


def infer_type(values: list) -> str:
    present = [v for v in values if not _is_missing(v)]
    if not present:
        return "empty"
    if any(isinstance(v, (list, dict)) for v in present):
        return "nested"
    if all(_to_number(v) is not None for v in present):
        return "numeric"
    keys = [_key(v) for v in present]
    unique = len(set(keys))
    avg_len = sum(len(k) for k in keys) / len(keys)
    if unique <= max(20, len(present) * 0.05) and avg_len <= 30:
        return "categorical"
    return "text"


def _quantile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def numeric_stats(nums: list[float]) -> dict:
    s = sorted(nums)
    q1, q3 = _quantile(s, 0.25), _quantile(s, 0.75)
    iqr = q3 - q1
    outliers = sum(1 for x in s if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr)
    return {"count": len(s), "mean": statistics.fmean(s), "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
            "min": s[0], "q1": q1, "median": _quantile(s, 0.5), "q3": q3, "max": s[-1],
            "iqr_outliers": outliers}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def guess_target(columns: list[str], col_types: dict) -> str | None:
    for hint in TARGET_HINTS:
        for c in columns:
            if c.lower() == hint and col_types.get(c) in ("categorical", "numeric"):
                return c
    return None


def analyze_table(rows: list[dict], columns: list[str], target: str | None = None) -> dict:
    if target is not None and target not in columns:
        raise ValueError(f"找不到目標欄位 {target}（可用欄位：{'、'.join(columns)}）")
    n = len(rows)
    col_types, col_info = {}, {}
    for c in columns:
        values = [r.get(c) for r in rows]
        t = infer_type(values)
        col_types[c] = t
        present = [v for v in values if not _is_missing(v)]
        info = {"type": t, "missing": n - len(present), "missing_rate": (n - len(present)) / n if n else 0.0,
                "unique": len({_key(v) for v in present})}
        if t == "numeric":
            info["stats"] = numeric_stats([_to_number(v) for v in present])
        elif t in ("categorical", "text", "nested"):
            keys = [_key(v) for v in present]
            if t == "categorical":
                info["top"] = Counter(keys).most_common(10)
            if keys:
                lengths = [len(k) for k in keys]
                info["length"] = {"min": min(lengths), "mean": statistics.fmean(lengths), "max": max(lengths)}
        col_info[c] = info

    auto_target = target is None
    target = target or guess_target(columns, col_types)
    keys = [tuple(_key(r.get(c)) for c in columns) for r in rows]
    dup_rows = n - len(set(keys))

    target_info = None
    if target:
        labels = [_key(r.get(target)) for r in rows if not _is_missing(r.get(target))]
        dist = Counter(labels)
        target_info = {"column": target, "auto_detected": auto_target, "type": col_types[target]}
        if col_types[target] == "categorical" or len(dist) <= 20:
            counts = dist.most_common()
            target_info.update(classes=len(dist), distribution=counts,
                               imbalance_ratio=(counts[0][1] / counts[-1][1]) if counts else None)
        feature_cols = [c for c in columns if c != target]
        by_features = defaultdict(set)
        for r in rows:
            by_features[tuple(_key(r.get(c)) for c in feature_cols)].add(_key(r.get(target)))
        target_info["label_conflicts"] = sum(1 for v in by_features.values() if len(v) > 1)

    numeric_cols = [c for c in columns if col_types[c] == "numeric"]
    correlations = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            pairs = [(_to_number(r.get(a)), _to_number(r.get(b))) for r in rows]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            r_val = pearson([p[0] for p in pairs], [p[1] for p in pairs])
            if r_val is not None:
                correlations.append({"a": a, "b": b, "r": r_val, "n": len(pairs)})
    correlations.sort(key=lambda d: -abs(d["r"]))

    warnings = []
    for c, info in col_info.items():
        if info["missing_rate"] > 0.2:
            warnings.append(f"欄位「{c}」缺值 {info['missing_rate']:.0%}，建議補值或排除")
        if info["unique"] == 1 and n > 1:
            warnings.append(f"欄位「{c}」只有一種值，對模型沒有資訊")
        if info["unique"] == n and n > 20 and info["type"] in ("numeric", "categorical", "text") and c != target \
                and re.search(r"(?i)(^|_)(id|uuid|key|no|編號)$", c):
            warnings.append(f"欄位「{c}」每列都不同，疑似 ID，不要當特徵（容易背答案）")
    if dup_rows:
        warnings.append(f"有 {dup_rows} 列完全重複，拆分前建議去重，否則可能 train/test 洩漏")
    if target_info:
        if target_info.get("label_conflicts"):
            warnings.append(f"有 {target_info['label_conflicts']} 組特徵完全相同但標籤不同（標註衝突）")
        ratio = target_info.get("imbalance_ratio")
        if ratio and ratio >= 3:
            warnings.append(f"目標「{target}」類別不平衡（最多/最少 = {ratio:.1f} 倍），建議分層拆分並看各類別指標")
        small = [k for k, v in target_info.get("distribution", []) if v < 5]
        if small:
            warnings.append(f"類別 {small[:5]} 少於 5 筆，分層拆分後 val/test 可能沒有這類")
    elif auto_target:
        warnings.append("沒有偵測到目標欄位；要看類別分佈或分層拆分請用 --target 指定")
    for c in correlations:
        if abs(c["r"]) >= 0.95 and target not in (c["a"], c["b"]):
            warnings.append(f"「{c['a']}」與「{c['b']}」相關係數 {c['r']:.2f}，幾乎重複")

    return {"kind": "table", "rows": n, "columns": columns, "column_info": col_info,
            "duplicate_rows": dup_rows, "target": target_info, "correlations": correlations[:10],
            "warnings": warnings}


# ---------------------------------------------------------------- 影像資料夾

def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def analyze_image_folder(folder) -> dict:
    folder = Path(folder)
    split_dirs = [folder / s for s in SPLIT_NAMES if (folder / s).is_dir()]
    layout = "split" if split_dirs else "class_folders"
    counts = {}
    sizes = []
    for base in split_dirs or [folder]:
        name = base.name if split_dirs else "all"
        counts[name] = {c.name: len(_images(c)) for c in sorted(p for p in base.iterdir() if p.is_dir())}
        for c in sorted(p for p in base.iterdir() if p.is_dir()):
            sizes += _images(c)[:5]  # 每類取樣 5 張看尺寸
    if not any(counts.values()):
        raise ValueError(f"{folder} 底下找不到「類別子資料夾/圖片」結構")
    dims = []
    try:
        from PIL import Image
        for p in sizes:
            with Image.open(p) as im:
                dims.append(im.size)
    except ImportError:
        pass
    totals = Counter()
    for per in counts.values():
        totals.update(per)
    warnings = []
    if totals:
        most, least = max(totals.values()), min(totals.values())
        if least == 0:
            warnings.append(f"類別 {[k for k, v in totals.items() if v == 0]} 沒有圖片")
        elif most / least >= 3:
            warnings.append(f"類別不平衡（最多/最少 = {most / least:.1f} 倍）")
    if len(set(dims)) > 1:
        warnings.append("圖片尺寸不一致，訓練時會被 resize（取樣結果見 image_sizes）")
    leak = None
    if layout == "split":
        from leakage_check import check_split_leakage
        report = check_split_leakage(folder)
        leak = {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}
        if report["leak_count"]:
            warnings.append(f"偵測到 {report['leak_count']} 處跨 split 重複／近似重複圖片（資料洩漏）")
    return {"kind": "images", "layout": layout, "counts": counts, "class_totals": dict(totals),
            "total_images": sum(totals.values()), "image_sizes": Counter(dims).most_common(5),
            "leakage": leak, "warnings": warnings}


def analyze_path(path, target: str | None = None) -> dict:
    path = Path(path)
    if path.is_dir():
        result = analyze_image_folder(path)
    elif path.is_file():
        rows, columns = load_table(path)
        if not rows:
            raise ValueError(f"{path.name} 沒有任何資料列")
        result = analyze_table(rows, columns, target)
    else:
        raise ValueError(f"找不到檔案或資料夾：{path}")
    result["source"] = str(path)
    return result


# ---------------------------------------------------------------- 拆分

def _allocate(n: int, ratios: tuple[float, ...]) -> list[int]:
    """最大餘數法把 n 個單位依比例分配，且 n 夠多時每份至少 1 個。"""
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - counts[i]))[: n - sum(counts)]:
        counts[i] += 1
    if n >= len(ratios):
        for i in range(len(counts)):
            if counts[i] == 0 and ratios[i] > 0:
                j = counts.index(max(counts))
                counts[j] -= 1
                counts[i] += 1
    return counts


def _feature_key(row: dict, target: str | None) -> str:
    return json.dumps({k: _key(v) for k, v in row.items() if k != target}, ensure_ascii=False, sort_keys=True)


def split_rows(rows: list[dict], ratios=(0.7, 0.15, 0.15), seed: int = 42,
               target: str | None = None, group: str | None = None) -> dict[str, list[dict]]:
    """依比例拆分；target＝分層（各類別比例一致），group＝同一組不跨 split。"""
    if len(ratios) not in (2, 3) or abs(sum(ratios) - 1) > 1e-6 or min(ratios) <= 0:
        raise ValueError("ratios 要 2 或 3 個正數且總和為 1，例如 0.7 0.15 0.15")
    names = SPLIT_NAMES[:len(ratios)] if len(ratios) == 3 else ("train", "test")
    for col in (target, group):
        if col and rows and col not in rows[0] and not any(col in r for r in rows):
            raise ValueError(f"找不到欄位 {col}")
    units = defaultdict(list)  # 拆分單位：group 值；沒給 group 時特徵相同的列（含重複列）自動同組，避免洩漏
    for i, r in enumerate(rows):
        units[_key(r.get(group)) if group else _feature_key(r, target)].append(i)
    strata = defaultdict(list)
    for uid, idxs in units.items():
        label = Counter(_key(rows[i].get(target)) for i in idxs).most_common(1)[0][0] if target else "_all"
        strata[label].append(uid)
    rng = random.Random(seed)
    assign = {}
    for label in sorted(strata):
        uids = sorted(strata[label], key=str)
        rng.shuffle(uids)
        start = 0
        for name, count in zip(names, _allocate(len(uids), ratios)):
            for uid in uids[start:start + count]:
                assign[uid] = name
            start += count
    out = {name: [] for name in names}
    for uid, idxs in units.items():
        out[assign[uid]].extend(rows[i] for i in idxs)
    return out


def check_table_leakage(splits: dict[str, list[dict]], target: str | None = None,
                        group: str | None = None) -> dict:
    """跨 split 的完全重複列（排除目標欄後的特徵相同也算）與分組跨 split。"""
    owners, feat_owner, group_owner = {}, {}, {}
    exact = features = groups = 0
    for split, rows in splits.items():
        for r in rows:
            full = json.dumps({k: _key(v) for k, v in r.items()}, ensure_ascii=False, sort_keys=True)
            feat = json.dumps({k: _key(v) for k, v in r.items() if k != target}, ensure_ascii=False, sort_keys=True)
            if owners.setdefault(full, split) != split:
                exact += 1
            elif target and feat_owner.setdefault(feat, split) != split:
                features += 1
            if group:
                if group_owner.setdefault(_key(r.get(group)), split) != split:
                    groups += 1
    return {"exact_duplicates_across_splits": exact, "same_features_across_splits": features,
            "group_leaks": groups, "leak_count": exact + features + groups}


def _write_rows(rows: list[dict], path: Path, columns: list[str]):
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({c: (_key(r.get(c)) if isinstance(r.get(c), (list, dict)) else r.get(c, "")) for c in columns})
    elif path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path) -> str:
    """專案內的路徑顯示成相對專案根目錄（跨 Windows／Linux 都看得懂）；專案外維持原樣。"""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _out_dir(source: Path, out_dir, action: str) -> Path:
    out = Path(out_dir) if out_dir else DEFAULT_OUT / f"{source.stem}_{action}_{datetime.now():%Y%m%d_%H%M%S}"
    out = out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError(f"輸出必須在專案資料夾內（規則 01）：{out}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"輸出資料夾已存在且非空，不覆寫：{out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_path(path, ratios=(0.7, 0.15, 0.15), seed=42, target=None, group=None, out_dir=None) -> dict:
    path = Path(path)
    if path.is_dir():
        from data_split import split_dataset
        from leakage_check import check_split_leakage
        if len(ratios) != 3:
            raise ValueError("影像資料夾拆分需要 3 個比例（train val test）")
        out = _out_dir(path, out_dir, "split")
        summary = split_dataset(path, out, *ratios, seed=seed)
        leak = check_split_leakage(out)
        counts = {s: sum(c.get(s, 0) for c in summary.values()) for s in SPLIT_NAMES}
        return {"kind": "images", "source": _rel(path), "out_dir": _rel(out), "counts": counts,
                "per_class": summary, "leakage": {"leak_count": leak["leak_count"],
                                                  "exact": len(leak["exact_duplicates"]),
                                                  "near": len(leak["near_duplicates"])}}
    rows, columns = load_table(path)
    if not rows:
        raise ValueError(f"{path.name} 沒有任何資料列")
    if target is None:
        col_types = {c: infer_type([r.get(c) for r in rows]) for c in columns}
        target = guess_target(columns, col_types)
    splits = split_rows(rows, ratios, seed, target, group)
    out = _out_dir(path, out_dir, "split")
    ext = path.suffix.lower() if path.suffix.lower() in (".csv", ".json", ".jsonl") else ".csv"
    if path.suffix.lower() == ".tsv":
        ext = ".csv"
    for name, part in splits.items():
        _write_rows(part, out / f"{name}{ext}", columns)
    dist = None
    if target:
        dist = {name: dict(Counter(_key(r.get(target)) for r in part).most_common()) for name, part in splits.items()}
    result = {"kind": "table", "source": _rel(path), "out_dir": _rel(out), "seed": seed, "ratios": list(ratios),
              "target": target, "group": group, "counts": {k: len(v) for k, v in splits.items()},
              "target_distribution": dist, "leakage": check_table_leakage(splits, target, group)}
    (out / "split_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def kfold_path(path, k=5, seed=42, target=None, group=None, out_dir=None) -> dict:
    path = Path(path)
    from kfold import stratified_group_kfold
    if path.is_dir():
        from kfold import make_folds, summarize
        folds = make_folds(path, k, seed)
        out = _out_dir(path, out_dir, "kfold")
        (out / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"kind": "images", "source": _rel(path), "out_dir": _rel(out), "k": k,
                "folds": summarize(folds), "warnings": folds["warnings"]}
    rows, columns = load_table(path)
    if target is None:
        col_types = {c: infer_type([r.get(c) for r in rows]) for c in columns}
        target = guess_target(columns, col_types)
    labels = [_key(r.get(target)) if target else "_all" for r in rows]
    group_ids = {}  # 沒給 group 時，特徵相同的列同組（重複列不會跨 fold）
    groups = [group_ids.setdefault(_key(r.get(group)) if group else _feature_key(r, target), len(group_ids)) for r in rows]
    assignment, warnings = stratified_group_kfold(labels, groups, k, seed)
    out = _out_dir(path, out_dir, "kfold")
    with_fold = [dict(r, fold=f) for r, f in zip(rows, assignment)]
    _write_rows(with_fold, out / ("folds.csv" if path.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx") else "folds.json"),
                columns + ["fold"])
    per_fold = []
    for f in range(k):
        idx = [i for i, a in enumerate(assignment) if a == f]
        per_fold.append({"fold": f, "n": len(idx), "per_class": dict(Counter(labels[i] for i in idx).most_common())
                         if target else None})
    result = {"kind": "table", "source": _rel(path), "out_dir": _rel(out), "k": k, "target": target,
              "group": group, "folds": per_fold, "warnings": warnings}
    (out / "kfold_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- 報告

def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def format_analysis(result: dict) -> str:
    lines = [f"# 資料分析：{Path(result['source']).name}", ""]
    if result["kind"] == "images":
        lines += [f"- 結構：{'已分割 train/val/test' if result['layout'] == 'split' else '每類一個資料夾（未分割）'}",
                  f"- 圖片總數：{result['total_images']}", "", "| 類別 | " + " | ".join(result["counts"]) + " |",
                  "|---|" + "---|" * len(result["counts"])]
        for cls in sorted(result["class_totals"]):
            lines.append(f"| {cls} | " + " | ".join(str(result["counts"][s].get(cls, 0)) for s in result["counts"]) + " |")
        if result["image_sizes"]:
            lines += ["", "尺寸取樣：" + "、".join(f"{w}×{h}（{n}張）" for (w, h), n in result["image_sizes"])]
        if result["leakage"]:
            lines += ["", f"跨 split 洩漏檢查：{result['leakage']['leak_count']} 處"]
    else:
        lines += [f"- 筆數：{result['rows']}　欄位：{len(result['columns'])}　完全重複列：{result['duplicate_rows']}", "",
                  "| 欄位 | 型別 | 缺值 | 唯一值 | 摘要 |", "|---|---|---|---|---|"]
        for c, info in result["column_info"].items():
            if "stats" in info:
                s = info["stats"]
                summary = (f"平均 {_fmt(s['mean'])}、標準差 {_fmt(s['std'])}、範圍 {_fmt(s['min'])}～{_fmt(s['max'])}、"
                           f"中位數 {_fmt(s['median'])}、IQR 離群 {s['iqr_outliers']}")
            elif "top" in info:
                summary = "、".join(f"{k}({v})" for k, v in info["top"][:5])
            elif "length" in info:
                L = info["length"]
                summary = f"長度 {L['min']}～{L['max']}，平均 {L['mean']:.1f} 字"
            else:
                summary = ""
            lines.append(f"| {c} | {info['type']} | {info['missing']}（{info['missing_rate']:.0%}） | {info['unique']} | "
                         f"{summary.replace('|', '/')} |")
        t = result["target"]
        if t:
            lines += ["", f"## 目標欄位：{t['column']}{'（自動偵測）' if t['auto_detected'] else ''}"]
            if t.get("distribution"):
                total = sum(v for _, v in t["distribution"])
                lines += ["| 類別 | 筆數 | 比例 |", "|---|---|---|"]
                lines += [f"| {k} | {v} | {v / total:.1%} |" for k, v in t["distribution"][:20]]
                lines.append(f"\n不平衡比（最多/最少）：{t['imbalance_ratio']:.2f}")
            lines.append(f"標籤衝突（特徵相同、標籤不同）：{t['label_conflicts']} 組")
        if result["correlations"]:
            lines += ["", "## 數值欄位相關係數（Pearson，前 10）", "| 欄位 A | 欄位 B | r | n |", "|---|---|---|---|"]
            lines += [f"| {c['a']} | {c['b']} | {c['r']:.3f} | {c['n']} |" for c in result["correlations"]]
    lines += ["", "## 提醒"] + ([f"- {w}" for w in result["warnings"]] or ["- 沒有發現明顯問題"])
    lines += ["", "以上數字由程式直接計算（data_analysis.py），不是模型生成。"]
    return "\n".join(lines)


def format_split(result: dict) -> str:
    lines = [f"# 資料拆分：{Path(result['source']).name}", "",
             f"- 輸出：{result['out_dir']}", f"- 筆數：{result['counts']}"]
    if result["kind"] == "table":
        lines.append(f"- 比例：{result['ratios']}　seed：{result['seed']}　分層：{result['target'] or '無'}　分組：{result['group'] or '無'}")
        if result["target_distribution"]:
            lines += ["", "| split | 類別分佈 |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in result["target_distribution"].items()]
    leak = result["leakage"]
    lines += ["", f"跨 split 洩漏檢查：{leak['leak_count']} 處" + ("（✅ 無洩漏）" if not leak["leak_count"] else "（⚠️ 請檢查重複資料）")]
    return "\n".join(lines)


def format_kfold(result: dict) -> str:
    lines = [f"# K-fold 分配：{Path(result['source']).name}", "", f"- 輸出：{result['out_dir']}", f"- k = {result['k']}"]
    for f in result["folds"]:
        lines.append(f"- fold {f['fold']}：{f['n']} 筆 {f.get('per_class') or ''}")
    lines += [f"- ⚠️ {w}" for w in result.get("warnings", [])]
    return "\n".join(lines)


def write_report(text: str, source, action: str) -> Path:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()  # 加上層資料夾名，避免不同資料夾的 train.json 互相覆蓋
    path = DEFAULT_OUT / f"{source.parent.name}_{source.stem}_{action}_report.md"
    path.write_text(text + "\n", encoding="utf-8")
    return path


COMMAND_USAGE = """用法：
  /data analyze <檔案或資料夾> [--target 欄位]
  /data split <檔案或資料夾> [--target 欄位] [--group 欄位] [--ratios 0.7 0.15 0.15] [--seed 42]
  /data kfold <檔案或資料夾> [--k 5] [--target 欄位] [--group 欄位] [--seed 42]
支援 .csv .tsv .json .jsonl .xlsx .txt，或影像資料夾（每類一個子資料夾）。
輸出寫在 output/data_analysis/，不會改動原始資料。"""
COMMAND_ACTIONS = ["analyze", "split", "kfold"]


class _ArgError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _ArgError(message)


def _tokens(arg: str) -> list[str]:
    """類 shell 切詞，但保留 Windows 反斜線；"..." 或 '...' 包住的路徑可含空白。"""
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
            for t in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', arg)]


def resolve_path(text: str) -> Path:
    """相對路徑先看目前目錄，找不到再從專案根目錄找（GUI/CLI/網頁啟動位置不同）。"""
    path = Path(text).expanduser()
    if not path.exists() and not path.is_absolute() and (ROOT / path).exists():
        path = ROOT / path
    return path


def run_command(arg: str, save_report: bool = True) -> str:
    """`/data` 指令的共用入口（GUI／CLI／網頁／自然語言路由都呼叫這裡），回傳要顯示的文字。"""
    tokens = _tokens(arg or "")
    if not tokens or tokens[0] not in COMMAND_ACTIONS:
        return COMMAND_USAGE
    action = tokens[0]
    parser = _Parser(prog=f"/data {action}", add_help=False)
    parser.add_argument("path")
    parser.add_argument("--target")
    parser.add_argument("--group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.7, 0.15, 0.15])
    parser.add_argument("--k", type=int, default=5)
    try:
        args = parser.parse_args(tokens[1:])
        path = resolve_path(args.path)
        if action == "analyze":
            if args.group:
                return "analyze 不用 --group；分組只影響 split／kfold。"
            text = format_analysis(analyze_path(path, args.target))
        elif action == "split":
            text = format_split(split_path(path, tuple(args.ratios), args.seed, args.target, args.group))
        else:
            text = format_kfold(kfold_path(path, args.k, args.seed, args.target, args.group))
    except _ArgError as exc:
        return f"指令格式錯誤：{exc}\n\n{COMMAND_USAGE}"
    except (ValueError, OSError) as exc:
        return f"資料處理失敗：{exc}"
    if save_report:
        text += f"\n\n報告已存：{write_report(text, path, action)}"
    return text


_DATA_PATH_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'|([^\s"\'，。、；：？！]+)')
_ACTION_WORDS = [("kfold", r"(?i)k-?fold|交叉驗證"), ("split", r"(?i)拆分|切分|分割|切成|split"),
                 ("analyze", r"(?i)分析|統計|看一下|檢查|概況|分佈|分布|analy[sz]e|profile")]


def route_data_request(message: str) -> tuple[str, str] | None:
    """自然語言觸發：訊息有「分析／拆分／k-fold」字眼，且提到一個存在的資料檔或影像資料夾，
    才交給 run_command；否則回 None，讓聊天模型照常回答。回傳 (trace, reply)。"""
    action = next((a for a, pattern in _ACTION_WORDS if re.search(pattern, message)), None)
    if action is None:
        return None
    candidate = None
    for match in _DATA_PATH_RE.finditer(message):
        token = next(g for g in match.groups() if g)
        path = resolve_path(token)
        if path.is_file() and path.suffix.lower() in TABLE_EXTS:
            candidate = token
            break
        if path.is_dir() and any(c.is_dir() for c in path.iterdir()) and ("/" in token or "\\" in token):
            candidate = token
            break
    if candidate is None:
        return None
    extra = []
    target = re.search(r"(?:--target|目標欄位|標籤欄位|依|按照?)\s*[「\"']?([A-Za-z_][\w]*|[\u4e00-\u9fff]{1,6})[」\"']?\s*(?:欄位)?(?:分層)?", message)
    if target and target.group(1) not in ("分層", "比例"):
        extra += ["--target", target.group(1)]
    k = re.search(r"(\d+)\s*-?\s*(?:fold|折)", message, re.I)
    if action == "kfold" and k:
        extra += ["--k", k.group(1)]
    command = " ".join([action, f'"{candidate}"', *extra])
    trace = (f"訊息含「{action}」意圖與資料路徑 {candidate} → 交給 data_analysis.py 確定性計算"
             "（數字由程式算，不是模型生成）")
    return trace, run_command(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("analyze", "split", "kfold"):
        p = sub.add_parser(name)
        p.add_argument("path", type=Path)
        p.add_argument("--target", help="目標／標籤欄位（不給會自動猜 label/class/score…）")
        if name != "analyze":
            p.add_argument("--group", help="分組欄位：同一組不跨 split／fold")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--out-dir", type=Path)
        if name == "split":
            p.add_argument("--ratios", type=float, nargs="+", default=[0.7, 0.15, 0.15])
        if name == "kfold":
            p.add_argument("--k", type=int, default=5)
        p.add_argument("--save-report", action="store_true", help=f"報告另存到 {DEFAULT_OUT}")
    args = parser.parse_args()
    try:
        if args.action == "analyze":
            text = format_analysis(analyze_path(args.path, args.target))
        elif args.action == "split":
            text = format_split(split_path(args.path, tuple(args.ratios), args.seed, args.target, args.group, args.out_dir))
        else:
            text = format_kfold(kfold_path(args.path, args.k, args.seed, args.target, args.group, args.out_dir))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(text)
    if args.save_report:
        print(f"\n報告：{write_report(text, args.path, args.action)}")


if __name__ == "__main__":
    main()
