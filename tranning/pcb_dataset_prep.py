"""公開 PCB 缺陷資料集（Pascal VOC XML 標註，如 PKU-Market-PCB：missing_hole/mouse_bite/open_circuit/
short/spur/spurious_copper）→ ImageFolder 分類資料。把每個標註框裁成一張小圖，存到 <out>/<類別>/。
資料集由使用者自行下載（不替你連外抓）；之後接 data_split.py → image_classifier_bnn.py，全程不動舊的
合成色塊資料（data/dataset/、image_classifier_runs/）。

    python pcb_dataset_prep.py --images <圖片資料夾> --annotations <XML資料夾> --out data/pcb_crops
    python data_split.py --source data/pcb_crops --output data/pcb_split
    python image_classifier_bnn.py --data-dir data/pcb_split --out-dir tranning/pcb_classifier_runs
"""
import argparse, cv2, xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def prep(images: Path, annotations: Path, out: Path, pad: int = 8, min_size: int = 8) -> dict:
    """回傳 {類別: 裁出張數}。找不到對應圖片／框太小的標註直接略過。"""
    counts, files = {}, {p.stem: p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    for xml in sorted(annotations.rglob("*.xml")):
        img_path = files.get(xml.stem)
        img = cv2.imread(str(img_path)) if img_path else None
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, obj in enumerate(ET.parse(xml).getroot().iter("object")):
            name, box = obj.findtext("name", "").strip(), obj.find("bndbox")
            if not name or box is None:
                continue
            x1, y1, x2, y2 = (int(float(box.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, y1, x2, y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue
            (out / name).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / name / f"{xml.stem}_{i}.jpg"), img[y1:y2, x1:x2])
            counts[name] = counts.get(name, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="PCB VOC 標註 → 分類資料夾")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pcb_crops"))
    ap.add_argument("--pad", type=int, default=8, help="框外多留的像素")
    a = ap.parse_args()
    counts = prep(a.images, a.annotations, a.out, a.pad)
    print("\n".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "沒有裁出任何圖片，請檢查資料夾與檔名是否對應")


if __name__ == "__main__":
    main()
