"""物品人工標註資料核心；不會自行開相機或開始訓練。"""
import copy, hashlib, json, math, shutil, uuid
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "object_detection"


def project_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("輸出與模型路徑必須位於 AI-module 專案內。")
    return path


def rectangle(points, width, height):
    """原圖座標，支援反向拖曳，裁切越界並拒絕無效矩形。"""
    if width <= 0 or height <= 0 or len(points) != 4 or not all(math.isfinite(v) for v in points):
        raise ValueError("無效的圖片尺寸或座標。")
    x1, x2 = sorted((max(0, min(width, points[0])), max(0, min(width, points[2]))))
    y1, y2 = sorted((max(0, min(height, points[1])), max(0, min(height, points[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("標註框不能是零面積。")
    return [x1, y1, x2, y2]


def yolo_line(box, width, height, count):
    cls, *points = box
    if not isinstance(cls, int) or not 0 <= cls < count:
        raise ValueError("標註類別 ID 不存在。")
    x1, y1, x2, y2 = rectangle(points, width, height)
    return f"{cls} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"


def read_labels(text, width, height, count):
    boxes = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("YOLO 標註每行需要五個欄位。")
        cls, values = int(fields[0]), list(map(float, fields[1:]))
        cx, cy, w, h = values
        if not 0 <= cls < count or not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or min(w, h) <= 0:
            raise ValueError("YOLO 類別或正規化座標無效。")
        if cx-w/2 < -1e-7 or cy-h/2 < -1e-7 or cx+w/2 > 1+1e-7 or cy+h/2 > 1+1e-7:
            raise ValueError("YOLO 標註超出圖片。")
        boxes.append([cls, *rectangle([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height], width, height)])
    return boxes


def fit_geometry(width, height, canvas_width, canvas_height):
    scale = min(canvas_width / width, canvas_height / height)
    return scale, (canvas_width-width*scale)/2, (canvas_height-height*scale)/2


class LabelDataset:
    def __init__(self, folder=DEFAULT_DATASET):
        self.folder = project_path(folder)
        self.index = self.folder / "annotations.json"
        self.data = json.loads(self.index.read_text(encoding="utf-8")) if self.index.exists() else {"classes": ["橡皮擦"], "images": {}}
        if not self.classes or len(set(self.classes)) != len(self.classes) or not all(isinstance(c, str) and c.strip() for c in self.classes):
            raise ValueError("資料集類別清單無效。")

    @property
    def classes(self):
        return self.data["classes"]

    def persist(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        temp = self.index.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.index)

    def add_class(self, name):
        name = name.strip()
        if not name:
            raise ValueError("類別名稱不可為空。")
        if name not in self.classes:
            self.classes.append(name)
            self.persist()
        return self.classes.index(name)

    def backup(self):
        folder = project_path(self.folder / "trash" / uuid.uuid4().hex)
        folder.mkdir(parents=True)
        (folder / "annotations.json").write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        return folder

    def unmarked_keys(self):
        return [key for key, item in self.data["images"].items()
                if not item["reviewed"] and not item["boxes"] and not item["negative"]]

    def delete_unmarked(self):
        keys = self.unmarked_keys()
        if not keys:
            return 0
        paths = {key: self.image_path(key) for key in keys}
        backup, original = self.backup(), copy.deepcopy(self.data)
        moved = []
        try:
            for key, path in paths.items():
                if path.exists():
                    path.rename(backup / path.name)
                    moved.append(path)
                del self.data["images"][key]
            self.persist()
        except Exception:
            self.data = original
            for path in moved:
                (backup / path.name).rename(path)
            raise
        return len(keys)

    def rename_class(self, name, replacement):
        replacement = replacement.strip()
        if not replacement or (replacement != name and replacement in self.classes):
            raise ValueError("類別名稱不可空白或與其他類別重複。")
        index = self.classes.index(name)
        self.backup()
        original = copy.deepcopy(self.data)
        self.classes[index] = replacement
        try:
            self.persist()
        except Exception:
            self.data = original
            raise

    def delete_class(self, name):
        if len(self.classes) <= 1:
            raise ValueError("至少需保留一個類別；若是錯字請使用「更名類別」。")
        index = self.classes.index(name)
        self.backup()
        original = copy.deepcopy(self.data)
        self.classes.pop(index)
        affected = 0
        for item in self.data["images"].values():
            if any(box[0] == index for box in item["boxes"]):
                affected += 1
                item["reviewed"], item["negative"] = False, False
            item["boxes"] = [[box[0] - (box[0] > index), *box[1:]] for box in item["boxes"] if box[0] != index]
        try:
            self.persist()
        except Exception:
            self.data = original
            raise
        return affected

    def image_path(self, key):
        if key not in self.data["images"] or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("未知影像。")
        return project_path(self.folder / "source" / f"{key}.png")

    def import_image(self, source):
        with Image.open(source) as image:
            return self.add_image(ImageOps.exif_transpose(image).convert("RGB"))

    def add_image(self, image):
        image = image.convert("RGB")
        key = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
        if key not in self.data["images"]:
            self.data["images"][key] = {"boxes": [], "reviewed": False, "split": "train", "negative": False}
            target = self.image_path(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            self.persist()
        return key

    def load(self, key):
        with Image.open(self.image_path(key)) as image:
            return image.convert("RGB"), self.data["images"][key]

    def save(self, key, boxes, split="train", negative=False):
        if split not in ("train", "val") or (not boxes and not negative) or (boxes and negative):
            raise ValueError("請畫出物品框，或明確勾選「背景負例」；分組必須為 train/val。")
        image, _ = self.load(key)
        lines = [yolo_line(box, image.width, image.height, len(self.classes)) for box in boxes]
        checked = read_labels("\n".join(lines), image.width, image.height, len(self.classes))
        self.data["images"][key] = {"boxes": checked, "reviewed": True, "split": split, "negative": bool(negative)}
        self.persist()

    def export(self):
        """每次輸出獨立快照，避免修改資料時覆蓋正在訓練的檔案。"""
        import yaml
        reviewed = {key: item for key, item in self.data["images"].items() if item["reviewed"]}
        for split in ("train", "val"):
            if not any(item["split"] == split and item["boxes"] for item in reviewed.values()):
                raise ValueError(f"{split} 至少需要一張已標註物品的照片；請以不同場景分組。")
        target = project_path(self.folder / "exports" / uuid.uuid4().hex)
        for key, item in reviewed.items():
            image, _ = self.load(key)
            lines = [yolo_line(b, image.width, image.height, len(self.classes)) for b in item["boxes"]]
            split = item["split"]
            if split not in ("train", "val") or (not lines and not item["negative"]):
                raise ValueError("資料集有未確認或無效的標註。")
            for kind in ("images", "labels"):
                (target / kind / split).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.image_path(key), target / "images" / split / f"{key}.png")
            (target / "labels" / split / f"{key}.txt").write_text("\n".join(lines), encoding="utf-8")
        config = {"path": target.as_posix(), "train": "images/train", "val": "images/val", "names": dict(enumerate(self.classes))}
        path = target / "data.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path
