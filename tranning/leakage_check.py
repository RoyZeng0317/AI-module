"""影像資料集的 data leakage（資料洩漏）檢查。

訓練集的圖片若以「相同或幾乎相同」的形式出現在 val/test，驗證準確率就會被灌水：
模型只是背過那張圖，不是學會辨識。常見來源：

  - 完全相同的檔案被複製到兩個 split（檔名不同也算）
  - 同一張照片的縮放／重新壓縮／輕微調色版本、連拍或影片相鄰幀
  - 資料增強後的檔案（例如 img_001_aug3.jpg）跟原圖分到不同 split
  - 同一張圖被放進兩個不同類別（標註衝突，也會讓準確率失真）

這支只看檔案內容與檔名，不需要 torch，可以在訓練前單獨執行。近似重複使用
dHash（差異雜湊，64 bit）＋漢明距離；dHash 只抓「幾乎同一張圖」，不會把
「同類別但不同照片」誤判成洩漏，但也抓不到大幅裁切或翻轉後的版本——
這是刻意的取捨，結果請當作必要檢查而不是保證。

注意：純色或極度單調的圖片（例如測試用的合成色塊）dHash 會幾乎全部相同，
這時近似重複的報告本來就沒有意義。

用法：
    python leakage_check.py --data-dir path/to/split_output
    python leakage_check.py --data-dir path/to/split_output --threshold 6
    python leakage_check.py --data-dir path/to/split_output --group-regex "^(.+?)_aug\\d+"
    有洩漏時結束碼為 1，可放進自動化流程。
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DEFAULT_THRESHOLD = 5  # 64 bit 中最多 5 bit 不同視為近似重複


def list_images(root: Path) -> list[tuple[Path, str]]:
    """`root/<class>/*.jpg` → [(path, class_name)]，依路徑排序確保結果可重現。"""
    root = Path(root)
    items = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, class_dir.name))
    return items


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int:
    """差異雜湊：縮成 (hash_size+1)×hash_size 灰階，比較左右相鄰像素亮度。"""
    with Image.open(path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        px = small.tobytes()  # L 模式每像素 1 byte；getdata() 在 Pillow 14 移除
    value = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            value = (value << 1) | (left > right)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def regex_group_key(path: Path, group_regex: str | None) -> str | None:
    """用 group_regex 的第 1 個擷取群組當群組鍵（比對檔名主幹，不含副檔名）。"""
    if not group_regex:
        return None
    match = re.search(group_regex, Path(path).stem)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(paths: list[Path], threshold: int = DEFAULT_THRESHOLD,
                 group_regex: str | None = None, hashes: list[int] | None = None) -> list[int]:
    """把「應該視為同一張圖」的檔案歸成同一組，回傳每個檔案的群組編號。

    合併條件（任一成立）：位元組完全相同、dHash 漢明距離 <= threshold、
    group_regex 擷取到相同的鍵。k-fold 與分割都應以群組為單位，
    同一組不可跨 fold／split，否則就是洩漏。threshold < 0 代表不做近似比對。
    """
    n = len(paths)
    uf = _UnionFind(n)

    by_sha: dict[str, int] = {}
    by_key: dict[str, int] = {}
    for i, p in enumerate(paths):
        sha = file_sha256(p)
        if sha in by_sha:
            uf.union(by_sha[sha], i)
        else:
            by_sha[sha] = i
        key = regex_group_key(p, group_regex)
        if key is not None:
            if key in by_key:
                uf.union(by_key[key], i)
            else:
                by_key[key] = i

    if threshold >= 0:
        hashes = hashes if hashes is not None else [dhash(p) for p in paths]
        for i in range(n):
            hi = hashes[i]
            for j in range(i + 1, n):
                if hamming(hi, hashes[j]) <= threshold:
                    uf.union(i, j)

    roots: dict[int, int] = {}
    return [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]


def check_split_leakage(data_dir: Path, threshold: int = DEFAULT_THRESHOLD,
                        group_regex: str | None = None) -> dict:
    """檢查 data_split.py 輸出（train/val/test 各一層類別資料夾）的跨 split 洩漏。

    回傳 dict：
      exact_duplicates  跨 split 位元組完全相同的檔案組
      near_duplicates   跨 split dHash 近似（排除已列在 exact 的配對）
      group_leaks       group_regex 同一鍵卻分散在不同 split
      label_conflicts   位元組相同卻屬於不同類別（不限 split）
      counts            各 split 檔案數
      leak_count        上面三種跨 split 問題的總數
    """
    data_dir = Path(data_dir)
    records = []  # (split, class, path)
    for split in SPLITS:
        split_dir = data_dir / split
        if split_dir.is_dir():
            records += [(split, cls, p) for p, cls in list_images(split_dir)]
    if not records:
        raise ValueError(f"{data_dir} 底下找不到 train/val/test 圖片")

    def rel(p: Path) -> str:
        return Path(p).relative_to(data_dir).as_posix()

    shas = [file_sha256(p) for _, _, p in records]
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, sha in enumerate(shas):
        by_sha[sha].append(i)

    exact, conflicts = [], []
    exact_pairs = set()
    for idxs in by_sha.values():
        if len(idxs) < 2:
            continue
        if len({records[i][0] for i in idxs}) > 1:
            exact.append([rel(records[i][2]) for i in idxs])
            for a in idxs:
                for b in idxs:
                    exact_pairs.add((a, b))
        if len({records[i][1] for i in idxs}) > 1:
            conflicts.append([f"{rel(records[i][2])} ({records[i][1]})" for i in idxs])

    near = []
    if threshold >= 0:
        hashes = [dhash(p) for _, _, p in records]
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if records[i][0] == records[j][0] or (i, j) in exact_pairs:
                    continue
                dist = hamming(hashes[i], hashes[j])
                if dist <= threshold:
                    near.append({"a": rel(records[i][2]), "b": rel(records[j][2]), "distance": dist})

    group_leaks = []
    if group_regex:
        by_key: dict[str, list[int]] = defaultdict(list)
        for i, (_, _, p) in enumerate(records):
            key = regex_group_key(p, group_regex)
            if key is not None:
                by_key[key].append(i)
        for key, idxs in sorted(by_key.items()):
            if len({records[i][0] for i in idxs}) > 1:
                group_leaks.append({"group": key, "files": [rel(records[i][2]) for i in idxs]})

    counts = {s: sum(r[0] == s for r in records) for s in SPLITS}
    return {
        "exact_duplicates": exact,
        "near_duplicates": near,
        "group_leaks": group_leaks,
        "label_conflicts": conflicts,
        "counts": counts,
        "leak_count": len(exact) + len(near) + len(group_leaks),
    }


def format_report(report: dict, limit: int = 20) -> str:
    lines = [f"檔案數：{report['counts']}"]
    sections = [
        ("跨 split 完全相同", report["exact_duplicates"]),
        ("跨 split 近似重複 (dHash)", report["near_duplicates"]),
        ("跨 split 同群組 (group-regex)", report["group_leaks"]),
        ("同內容不同類別 (標註衝突)", report["label_conflicts"]),
    ]
    for title, items in sections:
        lines.append(f"{title}：{len(items)}")
        for item in items[:limit]:
            lines.append(f"  - {json.dumps(item, ensure_ascii=False)}")
        if len(items) > limit:
            lines.append(f"  ...（其餘 {len(items) - limit} 筆省略）")
    verdict = "未發現跨 split 洩漏" if report["leak_count"] == 0 else f"⚠️ 發現 {report['leak_count']} 處跨 split 洩漏"
    lines.append(verdict)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="影像資料集 train/val/test 資料洩漏檢查")
    parser.add_argument("--data-dir", type=Path, required=True, help="含 train/ val/ test/ 的資料夾（data_split.py 輸出）")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="dHash 漢明距離門檻，<= 此值視為近似重複；-1 關閉近似比對")
    parser.add_argument("--group-regex", default=None,
                        help="從檔名擷取群組鍵的正規表示式（第 1 個擷取群組），例如 '^(.+?)_aug\\d+'")
    parser.add_argument("--json", type=Path, help="另存完整報告 JSON")
    args = parser.parse_args()

    report = check_split_leakage(args.data_dir, args.threshold, args.group_regex)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(1 if report["leak_count"] else 0)


if __name__ == "__main__":
    main()
