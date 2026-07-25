"""YOLO object-detection wrapper. Loads the model once (lazily, on first
call) and exposes a single function so it can be exercised directly —
without FastAPI or a browser — in tests.
"""

from functools import lru_cache

import numpy as np

MODEL_NAME = "yolo11n.pt"  # nano: fastest CPU inference, downloaded on first use


@lru_cache(maxsize=4)
def _model(weights: str = MODEL_NAME):
    from ultralytics import YOLO
    return YOLO(weights)


def detect(image: np.ndarray, conf: float = 0.35, weights: str = MODEL_NAME) -> list[dict]:
    """Run object detection on a BGR image (as read by cv2 / cv2.imdecode).

    `weights` defaults to the COCO-pretrained nano model but can point at a
    fine-tuned local checkpoint (e.g. a circuit-component detector) instead.

    Returns a list of {label, confidence, box: [x1, y1, x2, y2]}, sorted by
    confidence descending.
    """
    results = _model(weights).predict(image, conf=conf, verbose=False)[0]
    names = results.names

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "label": names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "box": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
