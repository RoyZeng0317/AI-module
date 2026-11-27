"""公開 PCB 缺陷資料集（Pascal VOC XML 標註，如 PKU-Market-PCB：missing_hole/mouse_bite/open_circuit/
short/spur/spurious_copper）→ 兩種輸出格式都支援，資料集由使用者自行下載（不替你連外抓）：

  1. 分類（原有）：把每個標註框裁成一張小圖，存到 <out>/<類別>/，接 data_split.py → image_classifier_bnn.py。
         python pcb_dataset_prep.py --images <圖片資料夾> --annotations <XML資料夾> --out data/pcb_crops
         python data_split.py --source data/pcb_crops --output data/pcb_split
         python image_classifier_bnn.py --data-dir data/pcb_split --out-dir tranning/pcb_classifier_runs

  2. YOLO 偵測（新增）：VOC 框直接轉成 YOLO 格式（不裁圖，保留完整畫面＋座標），接 pcb_defect_train.py。
         python pcb_dataset_prep.py --images <圖片資料夾> --annotations <XML資料夾> --yolo-out data/pcb_yolo
         python pcb_defect_train.py --data data/pcb_yolo/data.yaml

兩種模式互不影響，也不動舊的合成色塊資料（data/dataset/、image_classifier_runs/）。
"""
import argparse, random, cv2, xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_annotations(images: Path, annotations: Path):
    """讀所有找得到對應圖片的 VOC XML，回傳 [(stem, 圖片(np.ndarray), w, h, [(name, x1, y1, x2, y2), ...]), ...]。"""
    files = {p.stem: p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    parsed = []
    for xml in sorted(annotations.rglob("*.xml")):
        img_path = files.get(xml.stem)
        img = cv2.imread(str(img_path)) if img_path else None
        if img is None:
            continue
        h, w = img.shape[:2]
        objs = []
        for obj in ET.parse(xml).getroot().iter("object"):
            name, box = obj.findtext("name", "").strip(), obj.find("bndbox")
            if not name or box is None:
                continue
            x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            objs.append((name, x1, y1, x2, y2))
        parsed.append((xml.stem, img, w, h, objs))
    return parsed


def prep(images: Path, annotations: Path, out: Path, pad: int = 8, min_size: int = 8) -> dict:
    """回傳 {類別: 裁出張數}。找不到對應圖片／框太小的標註直接略過。"""
    counts = {}
    for stem, img, w, h, objs in _parse_annotations(images, annotations):
        for i, (name, x1f, y1f, x2f, y2f) in enumerate(objs):
            x1, y1, x2, y2 = (max(int(x1f) - pad, 0), max(int(y1f) - pad, 0),
                               min(int(x2f) + pad, w), min(int(y2f) + pad, h))
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            (out / name).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / name / f"{stem}_{i}.jpg"), img[y1:y2, x1:x2])
            counts[name] = counts.get(name, 0) + 1
    return counts


def prep_yolo(images: Path, annotations: Path, out: Path, classes: list = None,
              val_ratio: float = 0.2, seed: int = 0) -> dict:
    """VOC XML 標註 → YOLO 格式（<out>/images,labels/{train,val} + data.yaml），不裁圖、保留完整畫面。
    classes 未指定時，用實際出現過的類別名稱排序當 label id；回傳 {類別: 標註框數}。"""
    parsed = [p for p in _parse_annotations(images, annotations) if p[4]]
    names = classes or sorted({name for *_, objs in parsed for name, *_ in objs})
    index = {n: i for i, n in enumerate(names)}

    rng = random.Random(seed)
    val_idx = set(rng.sample(range(len(parsed)), int(len(parsed) * val_ratio))) if parsed else set()

    counts = {}
    for i, (stem, img, w, h, objs) in enumerate(parsed):
        split = "val" if i in val_idx else "train"
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        lines = []
        for name, x1, y1, x2, y2 in objs:
            if name not in index:
                continue
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{index[name]} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
            counts[name] = counts.get(name, 0) + 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    return counts


def main():
    ap = argparse.ArgumentParser(description="PCB VOC 標註 → 分類資料夾 / YOLO 偵測資料集")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pcb_crops"), help="分類模式輸出資料夾")
    ap.add_argument("--pad", type=int, default=8, help="分類模式：框外多留的像素")
    ap.add_argument("--yolo-out", type=Path, help="改成 YOLO 偵測格式輸出，給了這個就不跑分類模式")
    ap.add_argument("--classes", nargs="+", help="YOLO 模式的類別順序（未指定 = 自動偵測並排序）")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="YOLO 模式的驗證集比例")
    ap.add_argument("--seed", type=int, default=0, help="YOLO 模式切 train/val 的隨機種子")
    a = ap.parse_args()
    if a.yolo_out:
        counts = prep_yolo(a.images, a.annotations, a.yolo_out, a.classes, a.val_ratio, a.seed)
        print("\n".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "沒有轉出任何標註，請檢查資料夾與檔名是否對應")
        return
    counts = prep(a.images, a.annotations, a.out, a.pad)
    print("\n".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "沒有裁出任何圖片，請檢查資料夾與檔名是否對應")


if __name__ == "__main__":
    main()
