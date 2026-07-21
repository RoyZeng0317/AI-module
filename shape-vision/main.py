"""Shape Vision entry point.

Live GUI (webcam):
    python main.py
    python main.py --camera 1

Headless, single image in / annotated image + JSON counts out (no display,
no webcam needed — useful for CI or a machine without a camera):
    python main.py --image photo.jpg --out result.png
"""

import argparse
import json
import sys

from shape_vision.camera import open_source
from shape_vision.pipeline import DetectionParams, detect_shapes, draw_shapes, summarize


def run_headless(image: str, out: str | None) -> int:
    import cv2

    source = open_source(image=image)
    frame = source.read()
    shapes, _ = detect_shapes(frame, DetectionParams())
    counts = summarize(shapes)
    print(json.dumps(counts, indent=2))
    if out:
        cv2.imwrite(out, draw_shapes(frame, shapes))
        print(f"Saved annotated image to {out}")
    return 0


def run_gui(camera, video, image) -> int:
    from shape_vision.gui import ShapeVisionApp

    source = open_source(camera=camera, video=video, image=image)
    ShapeVisionApp(source).run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time OpenCV shape & circle detector")
    parser.add_argument("--camera", type=int, default=None, help="webcam index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="path to a video file instead of a webcam")
    parser.add_argument("--image", type=str, default=None, help="path to a still image instead of a live source")
    parser.add_argument("--out", type=str, default=None, help="with --image: where to save the annotated result")
    parser.add_argument("--no-gui", action="store_true", help="process --image once and exit, no window")
    args = parser.parse_args()

    if args.no_gui:
        if not args.image:
            parser.error("--no-gui requires --image")
        return run_headless(args.image, args.out)

    return run_gui(args.camera, args.video, args.image)


if __name__ == "__main__":
    sys.exit(main())
