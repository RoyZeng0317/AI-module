"""Thin CLI shell for tranning/image_analysis.py (YOLO detection + the
project's own MC Dropout classifier, combined into one report).

All the actual model/inference logic lives in tranning/ (image_analysis.py,
image_classifier_bnn.py, web/backend/detector.py) per the project's
lib/ = front-end shell, tranning/ = training+inference logic split — this
file used to hold a from-scratch ResNet18 cat/dog training loop directly,
which duplicated the already-existing, better-tested generic classifier in
image_classifier_bnn.py (see image_analysis.py's docstring). Decided
2026-09-14 via AskUserQuestion to move the logic to tranning/ and keep this
as a caller only.

Usage:
    python lib/img_analyize.py path/to/photo.jpg
"""

import os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tranning"))

from image_analysis import analyze_image, format_report  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("用法：python lib/img_analyize.py path/to/photo.jpg")
        return
    print(format_report(analyze_image(Path(sys.argv[1]))))


if __name__ == "__main__":
    main()
