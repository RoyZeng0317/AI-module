"""YOLO object crops -> grayscale -> zero padding -> heatmap/features -> decision tree."""

import argparse, csv, joblib, cv2, numpy as np, sys, time, yaml
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------------------------------
# File Location
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "output" / "image_features" / "decision_tree.joblib"
SIZE, CELLS = 32, 8
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -------------------------------------------------------------------------
# Hyperparameter setting
SEEDS = (50, 200, 500, 800, 1000)
FOLDS = 5
MAX_DEPTH = 12


def preprocess(image):
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    side = max(height, width)
    padded = cv2.copyMakeBorder(gray, (side - height) // 2, (side - height + 1) // 2,
                                (side - width) // 2, (side - width + 1) // 2,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def extract_features(gray):
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    heatmap, step = cv2.magnitude(gx, gy), SIZE // CELLS
    gray_cells = [gray[y:y + step, x:x + step].mean() for y in range(0, SIZE, step) for x in range(0, SIZE, step)]
    edge_cells = [heatmap[y:y + step, x:x + step].mean() for y in range(0, SIZE, step) for x in range(0, SIZE, step)]
    return np.array(gray_cells + edge_cells, dtype=np.float32), heatmap


def load_names(dataset):
    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    return names if isinstance(names, list) else [names[index] for index in sorted(names)]


def object_count(dataset, split):
    return sum(sum(len(line.split()) == 5 for line in path.read_text(encoding="utf-8").splitlines())
               for path in (dataset / "labels" / split).glob("*.txt"))


def samples(dataset, split, total):
    images, labels = dataset / "images" / split, dataset / "labels" / split
    started, completed, last_report = time.monotonic(), 0, 0
    for label_path in labels.glob("*.txt"):
        image_path = next((images / f"{label_path.stem}{suffix}" for suffix in IMAGE_TYPES
                           if (images / f"{label_path.stem}{suffix}").is_file()), None)
        if image_path is None:
            continue
        with Image.open(image_path) as source:
            width, height, object_index = *source.size, 0
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                class_id, cx, cy, box_w, box_h = map(float, values)
                x1, y1 = int((cx - box_w / 2) * width), int((cy - box_h / 2) * height)
                x2, y2 = int((cx + box_w / 2) * width), int((cy + box_h / 2) * height)
                gray = preprocess(source.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2))))
                features, _ = extract_features(gray)
                object_index, completed = object_index + 1, completed + 1
                if completed == total or completed - last_report >= max(1, total // 20):
                    elapsed = time.monotonic() - started
                    eta = elapsed / completed * (total - completed)
                    bar = "█" * int(20 * completed / total) + "·" * (20 - int(20 * completed / total))
                    line = f"[{split}] {bar} {completed}/{total} ({completed / total:.0%}) 已用 {elapsed:.0f}s ETA {eta:.0f}s"
                    print(line if not sys.stderr.isatty() else f"\r{line}", file=sys.stderr, flush=True)
                    last_report = completed
                yield features, int(class_id), gray, image_path, object_index
    if sys.stderr.isatty():
        print(file=sys.stderr, flush=True)


def build_model(seed):
    return DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=seed, class_weight="balanced")


def score(model, rows, class_count):
    actual = [row[1] for row in rows]
    predicted = model.predict([row[0] for row in rows])
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=range(class_count), average="macro", zero_division=0)
    return {"accuracy": accuracy_score(actual, predicted), "precision": precision, "recall": recall, "f1": f1}


def save_reports(rows, report_dir, seeds, folds):
    report_dir.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (report_dir / "fold_history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    by_seed = [[seed, mean(row["validation_accuracy"] for row in rows if row["seed"] == seed),
                mean(row["validation_f1"] for row in rows if row["seed"] == seed)] for seed in seeds]
    summary = [f"物件特徵決策樹：{len(seeds)} seeds × {folds} folds", "以圖片為分組單位，避免同張照片的框跨入不同折。",
               f"整體驗證準確率：{mean(row['validation_accuracy'] for row in rows):.2%} ± {pstdev(row['validation_accuracy'] for row in rows):.2%}",
               "", "Seed | 平均準確率 | 平均 Macro-F1"]
    summary += [f"{seed} | {accuracy:.2%} | {f1:.3f}" for seed, accuracy, f1 in by_seed]
    (report_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    body = "".join(f"<tr><td>{r['seed']}</td><td>{r['fold']}</td><td>{r['train_images']}</td><td>{r['validation_images']}</td><td>{r['train_accuracy']:.2%}</td><td>{r['validation_accuracy']:.2%}</td><td>{r['validation_f1']:.3f}</td></tr>" for r in rows)
    (report_dir / "report.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>決策樹交叉驗證結果</title><h1>物件特徵決策樹：{len(seeds)} seeds × {folds} folds</h1>"
        f"<p>平均驗證準確率：{mean(row['validation_accuracy'] for row in rows):.2%} ± {pstdev(row['validation_accuracy'] for row in rows):.2%}</p>"
        "<table border='1' cellpadding='6'><tr><th>Seed</th><th>Fold</th><th>Train images</th><th>Validation images</th><th>Train accuracy</th><th>Validation accuracy</th><th>Macro-F1</th></tr>"
        f"{body}</table>", encoding="utf-8")


# -------------------------------------------------------------------------
# Run five-fold training
def train(dataset, model_path, folds=FOLDS, seeds=SEEDS):
    names = load_names(dataset)
    train_count, val_count = object_count(dataset, "train"), object_count(dataset, "val")
    train_rows, val_rows = list(samples(dataset, "train", train_count)), list(samples(dataset, "val", val_count))
    groups = np.array(sorted({row[3].as_posix() for row in train_rows}))
    if len(groups) < folds or not train_rows or not val_rows:
        raise ValueError(f"至少需要 {folds} 張 train 圖片及有效的 train/val 標註")
    history, started, total_runs = [], time.monotonic(), len(seeds) * folds
    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=folds, shuffle=True, random_state=seed).split(groups), 1):
            train_groups, val_groups = set(groups[train_idx]), set(groups[val_idx])
            train_fold = [row for row in train_rows if row[3].as_posix() in train_groups]
            val_fold = [row for row in train_rows if row[3].as_posix() in val_groups]
            model = build_model(seed + fold - 1)
            model.fit([row[0] for row in train_fold], [row[1] for row in train_fold])
            train_score, val_score = score(model, train_fold, len(names)), score(model, val_fold, len(names))
            history.append({"seed": seed, "fold": fold, "train_images": len(train_groups), "validation_images": len(val_groups),
                            "train_objects": len(train_fold), "validation_objects": len(val_fold),
                            "train_accuracy": train_score["accuracy"], "validation_accuracy": val_score["accuracy"],
                            "validation_precision": val_score["precision"], "validation_recall": val_score["recall"], "validation_f1": val_score["f1"]})
            elapsed = time.monotonic() - started
            done = len(history); eta = elapsed / done * (total_runs - done)
            bar = "█" * int(20 * done / total_runs) + "·" * (20 - int(20 * done / total_runs))
            print(f"[CV] {bar} {done}/{total_runs} seed={seed} fold={fold}/{folds} val_acc={val_score['accuracy']:.2%} ETA={eta:.0f}s", flush=True)
    report_dir = ROOT / "output" / "image_features" / f"decision_tree_{folds}fold_{datetime.now():%Y%m%d_%H%M%S_%f}"
    save_reports(history, report_dir, seeds, folds)
    final_model = build_model(42)
    final_model.fit([row[0] for row in train_rows], [row[1] for row in train_rows])
    val_score = score(final_model, val_rows, len(names))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "names": names}, model_path)
    print(f"獨立 val：accuracy={val_score['accuracy']:.2%}、macro-F1={val_score['f1']:.3f}（{len(val_rows)} 個框）")
    print(f"模型：{model_path}\n交叉驗證報告：{report_dir}")


# -------------------------------------------------------------------------
# Predict objects
def predict_crop(gray, stem, index, bundle, output_dir):
    features, heatmap = extract_features(gray)
    class_id = int(bundle["model"].predict([features])[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_{index}_gray.png"), gray)
    normalized = np.empty_like(heatmap)
    cv2.normalize(heatmap, normalized, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(output_dir / f"{stem}_{index}_heatmap.png"), cv2.applyColorMap(
        normalized.astype(np.uint8), cv2.COLORMAP_JET))
    return class_id


def predict(image_path, model_path, output_dir, box=None):
    bundle = joblib.load(model_path)
    if image_path.is_dir():
        return predict_folder(image_path, bundle, output_dir, box)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到影像檔案：{image_path}")
    with Image.open(image_path) as image:
        if box:
            coords = tuple(map(int, box.split(",")))
            if len(coords) != 4:
                raise ValueError("--box 格式為 x1,y1,x2,y2，例如 40,30,220,200")
            image = image.crop(coords)
        gray = preprocess(image)
    class_id = predict_crop(gray, image_path.stem, 1, bundle, output_dir)
    print(f"判定結果：{bundle['names'][class_id]}（類別 {class_id}）")


def predict_folder(image_dir, bundle, output_dir, box=None):
    if box:
        raise ValueError("資料夾推論使用 YOLO 標註框，請移除 --box。")
    dataset, split = image_dir.parent.parent, image_dir.name
    if not (dataset / "labels" / split).is_dir() or load_names(dataset) != bundle["names"]:
        raise ValueError("找不到相符的 YOLO 標註，或資料集類別順序與模型不同。")
    total, correct = object_count(dataset, split), 0
    for _, actual, gray, image_path, index in samples(dataset, split, total):
        predicted = predict_crop(gray, image_path.stem, index, bundle, output_dir)
        correct += predicted == actual
    if not total:
        raise ValueError(f"資料夾沒有物件標註：{image_dir}")
    print(f"分類準確率：{correct / total:.2%}（{correct}/{total} 個標註物件）")


def main():
    parser = argparse.ArgumentParser(description="灰階、zero padding、特徵熱力圖與決策樹五折驗證")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", type=Path, help="YOLO 匯出資料夾，需含 images/train、images/val、labels/train、labels/val、data.yaml")
    action.add_argument("--predict", type=Path, help="單張影像或 YOLO images/train、images/val 資料夾")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "image_features")
    parser.add_argument("--box", help="單張影像的物件框：x1,y1,x2,y2")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if args.train:
        if args.folds < 2 or not args.seeds:
            parser.error("--folds 必須至少為 2，--seeds 不可空白")
        train(args.train, args.model, args.folds, args.seeds)
    else:
        predict(args.predict, args.model, args.output, args.box)


if __name__ == "__main__":
    main()
