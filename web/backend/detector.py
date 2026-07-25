"""YOLO object-detection wrapper. Loads the model once (lazily, on first
call) and exposes a single function so it can be exercised directly —
without FastAPI or a browser — in tests.
"""

from functools import lru_cache

import numpy as np

MODEL_NAME = "yolo11n.pt"  # nano: fastest CPU inference, downloaded on first use


def _device() -> str:
    # Ultralytics does NOT auto-move the model to the GPU just because CUDA
    # is available — .predict() stays on CPU unless told otherwise, which
    # left an RTX 4060 sitting idle while inference ran on CPU. Pick the GPU
    # when there is one; CPU-only deploy targets (e.g. Render.com) still get
    # a plain CPU run since torch.cuda.is_available() is False there.
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


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
    results = _model(weights).predict(image, conf=conf, device=_device(), verbose=False)[0]
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
