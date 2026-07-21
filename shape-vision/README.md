# Shape Vision

A real-time OpenCV shape & circle detector: webcam (or video/image) in →
grayscale → Gaussian blur → Canny edges → contours → convex hull → polygon
approximation → shape classification (Triangle / Square / Rectangle /
Circle / Polygon), with an optional Hough-circle transform cross-check for
circles. Every stage is tunable live from a tkinter control panel.

## Where this came from

This is a rebuild of the classical-CV progression taught in
[`2026AI-STUST` Week1 DAY7](https://github.com/harry123180/2026AI-STUST/tree/main/Week1/DAY7)
(pixel/image display → contour detection → convex hull + polygon-approx
shape classification → Hough circle detection → an interactive tuning GUI),
consolidated from six standalone teaching scripts into one tested,
reusable app:

| DAY7 script | Concept | Where it lives here |
|---|---|---|
| `像素展示.py` | load & show an image | `camera.StillImageSource` + `--image` |
| `視覺.py` | Canny + `findContours`, area filter | `pipeline.detect_shapes` |
| `凸包檢測(幾何形狀分類).py` | convex hull + `approxPolyDP` shape classification | `pipeline._classify` |
| `霍夫圓檢測.py` | `HoughCircles` circle detection | `pipeline.detect_hough_circles` |
| `視角切換與形狀偵測.py` | trackbar-tunable multi-view | `DetectionParams` + `gui.VIEWS` |
| `tkinter影像處理工具.py` | full tkinter GUI | `gui.ShapeVisionApp` |

The detection math (circularity `4πA/P²`, vertex-count classification,
Hough-circle center matching) is unchanged from the originals. What changed:
detection logic was pulled out of the GUI/camera loop into pure, testable
functions (`shape_vision/pipeline.py`), so classification can be verified
against synthetic shapes without a webcam or display (`tests/`), and the
same pipeline now drives both a live GUI and a headless single-image mode.

## Install

```bash
cd shape-vision
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Run

```bash
python main.py                        # live GUI, default webcam
python main.py --camera 1             # a different camera index
python main.py --video clip.mp4       # GUI over a video file instead of a webcam
python main.py --image photo.jpg      # GUI over a single still image (view still updates as you tune sliders)

# headless — no window, no camera; prints shape counts as JSON and (optionally) saves an annotated image
python main.py --image photo.jpg --out result.png --no-gui
```

In the GUI: pick a view (Raw / Gray / Blur / Canny Edges / Shape Detection),
tune blur/Canny/area/epsilon/circularity sliders, toggle which shapes to
draw, and hit **Save snapshot** to write a timestamped PNG.

## Test

```bash
pytest
```

Tests draw a triangle/square/rectangle/circle with `cv2` primitives on a
blank canvas and assert the pipeline classifies each correctly, plus that
the area filter rejects small noise and that show/hide toggles work.

## Layout

```
shape-vision/
├── main.py                    CLI entry (GUI + headless modes)
├── shape_vision/
│   ├── pipeline.py             detection logic (pure functions, no I/O)
│   ├── camera.py               webcam / video / still-image frame sources
│   └── gui.py                  tkinter live viewer + control panel
└── tests/
    └── test_pipeline.py        synthetic-shape classification tests
```
