"""YOLO detections -> schemdraw schematic drawing.

Bridges circuit_diagram_train.py's detector output (see detect() in
web/backend/detector.py: a list of {label, confidence, box: [x1,y1,x2,y2]})
to a rendered schemdraw.Drawing, so a trained circuit-component checkpoint
can go straight from "camera frame" to "原理圖" once one exists.

No circuit-component checkpoint exists yet (see circuit_diagram_train.py's
own docstring), so this is exercised in test_circuit_schemdraw_render.py
against synthetic/mock detections only -- it proves the box-position ->
schematic-symbol plumbing runs end to end, it does not claim any detection
accuracy.

Scope, deliberately: this places each detected component's schematic
symbol at its photographed position (box center, image pixels mapped to
schemdraw's coordinate space, y-axis flipped since image y grows downward
and schemdraw y grows upward) and labels it with the detected class. It
does NOT draw connecting wires between components -- circuit_rule_check.py's
own docstring already explains why: which pins are electrically joined
cannot be read off a picture, only inferred from pixel proximity, and that
inference is exactly the kind of fuzzy judgement this project avoids
building on top of a small from-scratch detector. Wiring stays a documented
gap until there is a real connection-detection approach, not a silent one.
"""

from pathlib import Path

import schemdraw
import schemdraw.elements as elm

# circuit_diagram_train.py's default --classes list (index order there is
# arbitrary; here the dict key is the class *name*, matched against
# detect()'s string label, not by index).
LABEL_TO_ELEMENT = {
    "resistor": elm.Resistor,
    "capacitor": elm.Capacitor,
    "inductor": elm.Inductor,
    "diode": elm.Diode,
    "led": elm.LED,
    "battery": elm.Battery,
    "transistor": elm.BjtNpn,
    "switch": elm.Switch,
    "ic": elm.Ic,
    # "wire_junction" is deliberately absent: a junction dot only means
    # something sitting on an actual wire, and this module draws no wires
    # (see docstring) -- its detected box center lands on or inside a real
    # component's own body/leads more often than not, rendering as a
    # nonsense dot stamped on top of that component instead of a junction.
}

# Divides photographed pixel distances down to schemdraw's own unit scale
# (its default element length is ~3 units) so components spread across a
# 1000px-wide frame don't end up 1000 units apart on the drawing.
PIXELS_PER_UNIT = 120.0


def _to_drawing_xy(box: list, img_h: float) -> tuple:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return cx / PIXELS_PER_UNIT, (img_h - cy) / PIXELS_PER_UNIT


def detections_to_drawing(detections: list, img_w: int, img_h: int) -> schemdraw.Drawing:
    """Build a schemdraw.Drawing placing one symbol per detection.

    Unrecognized labels (anything not in LABEL_TO_ELEMENT -- e.g. still a
    COCO class like "person" because no circuit checkpoint is loaded yet)
    are skipped, not guessed at.
    """
    d = schemdraw.Drawing()
    for det in detections:
        element_cls = LABEL_TO_ELEMENT.get(det["label"])
        if element_cls is None:
            continue
        x, y = _to_drawing_xy(det["box"], img_h)
        d += element_cls().at((x, y)).label(f'{det["label"]} {det["confidence"]:.2f}')
    return d


def save_schematic(drawing: schemdraw.Drawing, out_path: Path) -> None:
    drawing.save(str(out_path))
