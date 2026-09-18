"""Combined image analysis — YOLO object detection (web/backend/detector.py)
+ the project's own MC Dropout classifier (image_classifier_bnn.py) on the
same image, one report.

Search-before-build note: a from-scratch ResNet18 cat/dog classifier was
drafted directly in lib/img_analyize.py, but tranning/image_classifier_bnn.py
already is a generic (any ImageFolder dataset), better-tested classifier
(MobileNetV2 transfer learning + anti-overfitting measures + MC Dropout
confidence/entropy) — training that again from scratch here would just be a
worse duplicate. This module does not retrain anything; it wires detect()
and the already-existing classify_image() together into one call so a single
image gets both COCO's 80 generic labels (detect(), "what and where") and
the project's own trained categories (classify_image(), whichever checkpoint
currently exists at --classifier-dir — none yet, no dataset, see
to_do_list.md #17/#31 for the same "no dataset yet" situation). Train a
custom classifier the same way as before: `python image_classifier_bnn.py
--data-dir <split_output>`; analyze_image() just also calls whatever
checkpoint exists there.

Usage:
    python image_analysis.py --analyze path/to/photo.jpg
    python image_analysis.py --analyze path/to/photo.jpg --classifier-dir tranning/image_classifier_runs
"""

import argparse, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WEB_BACKEND_DIR = _PROJECT_ROOT / "web" / "backend"
if str(_WEB_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_BACKEND_DIR))

import cv2  # noqa: E402
from detector import MODEL_NAME as DETECTOR_WEIGHTS, detect  # noqa: E402

from bayesian_utils import low_confidence_warning
from image_classifier_bnn import DEFAULT_MC_SAMPLES, DEFAULT_OUT_DIR, classify_image


def analyze_image(image_path: Path, classifier_dir: Path = DEFAULT_OUT_DIR, conf: float = 0.35,
                   weights: str = DETECTOR_WEIGHTS, mc_samples: int = DEFAULT_MC_SAMPLES) -> dict:
    """Run detect() + classify_image() on the same image; one combined dict."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"無法讀取圖片：{image_path}")

    detections = detect(image, conf=conf, weights=weights)
    label, confidence, entropy = classify_image(image_path, out_dir=classifier_dir, mc_samples=mc_samples)
    return {
        "detections": detections,
        "classification": {"label": label, "confidence": confidence, "entropy": entropy},
    }


def format_report(result: dict) -> str:
    detections = result["detections"]
    if detections:
        items = "、".join(f'{d["label"]}（信心度 {d["confidence"]:.0%}）' for d in detections[:10])
        lines = [f"偵測到 {len(detections)} 個物件：{items}"]
    else:
        lines = ["沒有偵測到任何已知物件。"]

    c = result["classification"]
    if c["confidence"] > 0.0:
        warning = low_confidence_warning(c["confidence"])
        lines.append(f'自訂分類：{c["label"]}（信心度 {c["confidence"]:.0%}，entropy={c["entropy"]:.3f}）{warning}')
    else:
        lines.append(c["label"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="YOLO 偵測 + 自訂分類器的整合圖片分析")
    parser.add_argument("--analyze", type=Path, required=True, help="要分析的圖片路徑")
    parser.add_argument("--classifier-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help="image_classifier_bnn.py --data-dir 訓練出的 checkpoint 資料夾")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--weights", default=DETECTOR_WEIGHTS)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    args = parser.parse_args()

    result = analyze_image(args.analyze, args.classifier_dir, args.conf, args.weights, args.mc_samples)
    print(format_report(result))


if __name__ == "__main__":
    main()
