"""Core detection pipeline: grayscale -> blur -> Canny -> contours -> convex
hull -> polygon approximation -> shape/circularity classification, with an
optional Hough-circle cross-check.

Pure functions/dataclasses only (no camera or GUI code) so classification can
be unit-tested against synthetic images without a webcam or display.
"""

from dataclasses import dataclass
import cv2
import numpy as np

COLORS = {
    "Triangle": (0, 255, 255),
    "Square": (0, 255, 0),
    "Rectangle": (0, 200, 0),
    "Circle": (0, 0, 255),
    "Polygon": (255, 100, 200),
}


@dataclass
class DetectionParams:
    blur_ksize: int = 7
    blur_sigma: float = 1.5
    canny_t1: int = 30
    canny_t2: int = 150
    min_area: int = 500
    max_area: int = 50000
    approx_eps_pct: int = 4
    circularity_thr: int = 75
    show_triangle: bool = True
    show_square: bool = True
    show_circle: bool = True
    show_polygon: bool = False
    use_hough: bool = True
    hough_min_dist: int = 50
    hough_param1: int = 100
    hough_param2: int = 35
    hough_min_radius: int = 10
    hough_max_radius: int = 200


@dataclass
class Shape:
    label: str
    contour: np.ndarray
    area: float
    center: tuple
    color: tuple


def to_gray_blur(frame: np.ndarray, params: DetectionParams):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ksize = max(1, params.blur_ksize)
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), params.blur_sigma)
    return gray, blur


def to_edges(blur: np.ndarray, params: DetectionParams) -> np.ndarray:
    return cv2.Canny(blur, params.canny_t1, params.canny_t2)


def detect_hough_circles(blur: np.ndarray, params: DetectionParams) -> np.ndarray:
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=params.hough_min_dist,
        param1=params.hough_param1, param2=params.hough_param2,
        minRadius=params.hough_min_radius, maxRadius=params.hough_max_radius,
    )
    if circles is None:
        return np.empty((0, 3))
    return np.round(circles[0, :]).astype(int)


def _classify(cnt, approx, area, params: DetectionParams, hough_circles: np.ndarray):
    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return None
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    vtx = len(approx)

    by_shape = vtx >= 5 and circularity > (params.circularity_thr / 100.0)
    by_hough = False
    if params.use_hough and len(hough_circles):
        dist = np.hypot(hough_circles[:, 0] - cx, hough_circles[:, 1] - cy)
        by_hough = bool(np.any(dist < 20))

    if params.show_circle and (by_shape or by_hough):
        return Shape("Circle", approx, area, (cx, cy), COLORS["Circle"])
    if params.show_triangle and vtx == 3:
        return Shape("Triangle", approx, area, (cx, cy), COLORS["Triangle"])
    if params.show_square and vtx == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / h if h else 0
        label = "Square" if 0.85 <= ratio <= 1.15 else "Rectangle"
        return Shape(label, approx, area, (cx, cy), COLORS[label])
    if params.show_polygon and vtx >= 5:
        return Shape(f"Polygon-{vtx}", approx, area, (cx, cy), COLORS["Polygon"])
    return None


def detect_shapes(frame: np.ndarray, params: DetectionParams):
    """Run the full pipeline on one BGR frame.

    Returns (shapes, edges) where shapes is a list[Shape] and edges is the
    Canny edge map (useful for an "Edges" preview mode).
    """
    _, blur = to_gray_blur(frame, params)
    edges = to_edges(blur, params)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hough_circles = detect_hough_circles(blur, params) if params.use_hough else np.empty((0, 3))

    eps_pct = max(params.approx_eps_pct, 1) / 100.0
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area or area > params.max_area:
            continue
        hull = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(hull, eps_pct * cv2.arcLength(hull, True), True)
        shape = _classify(cnt, approx, area, params, hough_circles)
        if shape:
            shapes.append(shape)
    return shapes, edges


def draw_shapes(frame: np.ndarray, shapes) -> np.ndarray:
    out = frame.copy()
    for s in shapes:
        cv2.drawContours(out, [s.contour], -1, s.color, 2)
        cv2.circle(out, s.center, 4, s.color, -1)
        cv2.putText(out, f"{s.label} A:{int(s.area)}", (s.center[0] - 55, s.center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, s.color, 2)
    return out


def summarize(shapes) -> dict:
    counts: dict = {}
    for s in shapes:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts
