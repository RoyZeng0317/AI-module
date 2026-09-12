"""Pipeline smoke test for circuit_schemdraw_render.py.

No real circuit-component checkpoint exists yet, so detections here are
synthetic (hand-written label/box/confidence dicts, the exact shape
detect() returns) -- this only proves the box-position -> schematic-symbol
plumbing runs end to end and produces a savable drawing, not that any
detection is accurate.
"""

from circuit_schemdraw_render import detections_to_drawing, save_schematic

IMG_W, IMG_H = 640, 480

DETECTIONS = [
    {"label": "battery", "confidence": 0.91, "box": [40, 200, 100, 280]},
    {"label": "resistor", "confidence": 0.87, "box": [200, 200, 320, 260]},
    {"label": "led", "confidence": 0.76, "box": [400, 190, 460, 270]},
    {"label": "person", "confidence": 0.99, "box": [0, 0, 50, 50]},  # unrecognized COCO label
    # a junction box overlapping the resistor's own box -- must not render
    # as a dot stamped on top of the resistor (see circuit_schemdraw_render's
    # LABEL_TO_ELEMENT comment: no wires are drawn, so a junction dot here
    # would be meaningless, not just visually wrong)
    {"label": "wire_junction", "confidence": 0.5, "box": [220, 200, 240, 230]},
]


def test_detections_to_drawing_places_known_labels_only():
    drawing = detections_to_drawing(DETECTIONS, IMG_W, IMG_H)
    assert len(drawing.elements) == 3  # "person" and "wire_junction" skipped, not guessed at


def test_save_schematic_writes_a_file(tmp_path):
    drawing = detections_to_drawing(DETECTIONS, IMG_W, IMG_H)
    out_path = tmp_path / "schematic.svg"

    save_schematic(drawing, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_empty_detections_yields_empty_drawing():
    drawing = detections_to_drawing([], IMG_W, IMG_H)
    assert len(drawing.elements) == 0
