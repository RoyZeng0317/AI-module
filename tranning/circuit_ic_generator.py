"""Deterministic generator for an MCU / module / header minimal-system schematic.

Generation approach, and why it is deterministic code (not a trained model,
same "algorithm not neural net" category as circuit_rule_check.py and
calculus_generator.py -- see that docstring for the Rule 06 reasoning): the
project already tried a neural-net circuit generator once
(schmitt_trigger_train.py) and it still does not produce balanced/usable
output after 5 attempts (ErrorLog #23). This module builds the minimal
system (decoupling caps, oscillator + load caps, reset pull-up) from a
fixed, well-known circuit topology.

Supported parts (ic_library.py): ATmega328P (Arduino Uno), ATmega2560
(Arduino Mega 2560), ESP32-WROOM-32, Raspberry Pi 5 40-pin header.

Layout -- rewritten 2026-10-27 because the first version overlapped parts
on anything bigger than a DIP-28 (and was placed with a y-flipped transform,
so in real KiCad its caps landed on the wrong pins, see ErrorLog):

1. The IC sits in the middle. Every non-GPIO pin only gets a short wire
   stub ending in a power symbol (+5V / +3V3 / GND, rotated to point AWAY
   from the IC so neighbours 2.54mm apart never collide) or a net label
   (RESET / EN / XTAL1 / XTAL2 / AREF). GPIO and NC pins get a no-connect
   marker. Nothing else is placed next to the pins.
2. The support circuits are separate BLOCKS (a row of decoupling caps, the
   reset pull-up, the crystal with its load caps). Each block is built on
   its own, its bounding box -- part bodies AND their text -- is measured,
   and blocks are stacked in columns to the left of the IC with a fixed
   gap, so by construction no two blocks share space. They connect to the
   IC through matching power-symbol / label names, the way large MCU
   schematics are drawn in practice (and the way KiCad itself joins nets;
   circuit_rule_check.py merges same-name nets the same way).
3. The whole drawing is shifted onto the paper and the paper size (A4 / A3
   / A2) is chosen to fit, instead of sitting around (0, 0) off the sheet.

Every pin position is computed with circuit_rule_check.transform_point()
(library y-up -> sheet y-down; verified against KiCad's own demo files),
so the file opens in KiCad with each part on the pin it was meant for.
The library symbols embedded in the file include body graphics (resistor
box, capacitor plates, crystal, power symbols, IC box) so KiCad shows real
parts, not bare pins.

Usage:
    python circuit_ic_generator.py --ic atmega328p --out minimal_system.kicad_sch
    python circuit_ic_generator.py --ic mega2560 --out mega.kicad_sch --check --render mega.svg
    python circuit_ic_generator.py --ic esp32 --out esp32.kicad_sch --check
    python circuit_ic_generator.py --ic pi5 --out pi5_hat.kicad_sch --check
"""

import argparse
import uuid
from pathlib import Path

from circuit_rule_check import power_graphic_direction, transform_point
from ic_library import body_half_width, get_part, get_pins_with_layout, resolve_ic

GRID = 2.54
STUB = 5.08                 # wire from an IC pin tip to its power symbol / label
BLOCK_GAP = 7.62            # empty space kept between blocks (and between blocks and the IC area)
CAP_PITCH = 12.7            # horizontal spacing of caps in a decoupling row (room for "C10"/"100nF")
CHAR_W = 1.1                # approx. width of one 1.27mm-font character
TEXT_H = 1.6
LABEL_TEXT_H = 2.0          # a net label's text sits ABOVE its wire: font + KiCad's offset
POWER_LEN = 3.81            # power symbol stub + graphic
PAPERS = [("A4", 297.0, 210.0), ("A3", 420.0, 297.0), ("A2", 594.0, 420.0), ("A1", 841.0, 594.0)]
PAGE_MARGIN = 25.4
TITLE_BLOCK_H = 38.1        # keep clear of KiCad's title block at the bottom of the page
FILE_VERSION = 20230121     # KiCad 7 format: opens in KiCad 7, 8, 9 and 10
NS = uuid.UUID("6f1c8c7e-6b0e-4c1e-9a52-3c1f6b8d2a10")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _snap(v: float) -> float:
    return round(round(v / 1.27) * 1.27, 3)


def _box_union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _text_box(text: str, x: float, y: float, justify: str = "center") -> tuple:
    w = len(text) * CHAR_W
    if justify == "left":
        return (x, y - TEXT_H / 2, x + w, y + TEXT_H / 2)
    if justify == "right":
        return (x - w, y - TEXT_H / 2, x, y + TEXT_H / 2)
    return (x - w / 2, y - TEXT_H / 2, x + w / 2, y + TEXT_H / 2)


def _seg_box(p1, p2, half_width):
    """Box around an axis-aligned segment, widened only sideways (its ends
    stay exactly on the pins, where other parts are SUPPOSED to touch)."""
    (x1, y1), (x2, y2) = p1, p2
    if abs(x1 - x2) >= abs(y1 - y2):
        return (min(x1, x2), min(y1, y2) - half_width, max(x1, x2), max(y1, y2) + half_width)
    return (min(x1, x2) - half_width, min(y1, y2), max(x1, x2) + half_width, max(y1, y2))


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------------------
# library symbols (KiCad library coordinates: y up)
# ---------------------------------------------------------------------------

_FONT = "(effects (font (size 1.27 1.27)))"
_HIDE = "(effects (font (size 1.27 1.27)) hide)"

TWO_PIN_PARTS = {
    # lib_id: (half pin span, pin length, vertical?, graphics)
    # R and C are drawn STANDING UP in the library, like KiCad's own
    # Device:R / Device:C, so they are placed without rotation: KiCad adds a
    # symbol's rotation to its text fields' angle, and an unrotated part
    # keeps its "C1" / "100nF" text horizontal and left-justified.
    "Device:R": (3.81, 1.27, True, "(rectangle (start -1.016 -2.54) (end 1.016 2.54) (stroke (width 0.254) (type default)) (fill (type none)))"),
    "Device:C": (2.54, 2.032, True, "(polyline (pts (xy -1.524 -0.508) (xy 1.524 -0.508)) (stroke (width 0.3048) (type default)) (fill (type none)))"
                                    " (polyline (pts (xy -1.524 0.508) (xy 1.524 0.508)) (stroke (width 0.3048) (type default)) (fill (type none)))"),
    "Device:Crystal": (3.81, 1.778, False, "(rectangle (start -1.143 -1.905) (end 1.143 1.905) (stroke (width 0.254) (type default)) (fill (type none)))"
                                    " (polyline (pts (xy -2.032 -1.27) (xy -2.032 1.27)) (stroke (width 0.3048) (type default)) (fill (type none)))"
                                    " (polyline (pts (xy 2.032 -1.27) (xy 2.032 1.27)) (stroke (width 0.3048) (type default)) (fill (type none)))"),
}


def _props(ref_prefix: str, value: str, hide_ref: bool = False) -> str:
    return (f'(property "Reference" "{ref_prefix}" (at 0 0 0) {_HIDE if hide_ref else _FONT})'
            f' (property "Value" "{value}" (at 0 0 0) {_FONT})'
            f' (property "Footprint" "" (at 0 0 0) {_HIDE})'
            f' (property "Datasheet" "" (at 0 0 0) {_HIDE})')


def two_pin_local_pins(lib_id: str) -> list:
    """Library-local pin tips: pin 1 on top (y up) / left, pin 2 bottom / right."""
    half, _, vertical, _ = TWO_PIN_PARTS[lib_id]
    if vertical:
        return [{"number": "1", "x": 0.0, "y": half, "angle": 270}, {"number": "2", "x": 0.0, "y": -half, "angle": 90}]
    return [{"number": "1", "x": -half, "y": 0.0, "angle": 0}, {"number": "2", "x": half, "y": 0.0, "angle": 180}]


def _two_pin_lib(lib_id: str) -> str:
    _, length, _, graphics = TWO_PIN_PARTS[lib_id]
    part = lib_id.split(":", 1)[1]
    prefix = {"R": "R", "C": "C", "Crystal": "Y"}[part]
    pins = "".join(f'\n        (pin passive line (at {p["x"]} {p["y"]} {p["angle"]}) (length {length})'
                   f' (name "~" {_FONT}) (number "{p["number"]}" {_FONT}))' for p in two_pin_local_pins(lib_id))
    return (f'\n    (symbol "{lib_id}" (pin_numbers hide) (pin_names hide) (in_bom yes) (on_board yes) {_props(prefix, part)}'
            f'\n      (symbol "{part}_0_1" {graphics})'
            f'\n      (symbol "{part}_1_1"{pins}))')


def _power_lib(net: str) -> str:
    lib_id = f"power:{net}"
    if net == "GND":
        graphic = "(polyline (pts (xy 0 0) (xy 0 -1.27) (xy 1.27 -1.27) (xy 0 -2.54) (xy -1.27 -1.27) (xy 0 -1.27)) (stroke (width 0) (type default)) (fill (type none)))"
        pin_angle = 270
    else:
        graphic = ("(polyline (pts (xy 0 0) (xy 0 1.27)) (stroke (width 0) (type default)) (fill (type none)))"
                   " (polyline (pts (xy -0.762 1.27) (xy 0.762 1.27)) (stroke (width 0) (type default)) (fill (type none)))")
        pin_angle = 90
    return (f'\n    (symbol "{lib_id}" (power) (pin_names (offset 0)) (in_bom yes) (on_board yes) {_props("#PWR", net, hide_ref=True)}'
            f'\n      (symbol "{net}_0_1" {graphic})'
            f'\n      (symbol "{net}_1_1"'
            f'\n        (pin power_in line (at 0 0 {pin_angle}) (length 0) hide (name "{net}" {_FONT}) (number "1" {_FONT}))))')


def _ic_lib(lib_id: str, pins: list, half_w: float) -> str:
    part = lib_id.split(":", 1)[1]
    ys = [p["y"] for p in pins]
    top, bottom = max(ys) + GRID, min(ys) - GRID
    pin_lines = "\n".join(
        f'        (pin {p["electrical_type"]} line (at {p["x"]} {p["y"]} {0 if p["x"] < 0 else 180}) (length {GRID})'
        f' (name "{p["name"]}" {_FONT}) (number "{p["number"]}" {_FONT}))'
        for p in pins)
    return (f'\n    (symbol "{lib_id}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes) {_props("U", part)}'
            f'\n      (symbol "{part}_0_1" (rectangle (start {-half_w} {top}) (end {half_w} {bottom})'
            f' (stroke (width 0.254) (type default)) (fill (type background))))'
            f'\n      (symbol "{part}_1_1"\n{pin_lines}))')


# ---------------------------------------------------------------------------
# sheet model
# ---------------------------------------------------------------------------

class Sheet:
    """Everything placed on the sheet, plus the box each thing occupies
    (body and visible text), used to keep blocks apart and to verify the
    result has no overlaps before it is written."""

    def __init__(self, pin_tables: dict, next_ref):
        self.pin_tables = pin_tables          # lib_id -> [pin dicts with local x/y]
        self.next_ref = next_ref              # ONE counter for the whole sheet, shared by every block
        self.symbols, self.wires, self.labels = [], [], []
        self.junctions, self.no_connects = [], []
        self.boxes = []                       # ("owner|what", box)

    # -- placement primitives ---------------------------------------------
    def pin_xy(self, lib_id: str, number: str, x: float, y: float, angle: float) -> tuple:
        pin = next(p for p in self.pin_tables[lib_id] if p["number"] == number)
        return transform_point(pin["x"], pin["y"], None, angle, x, y)

    def part(self, lib_id, ref, value, x, y, angle, ref_at, value_at):
        """ref_at / value_at: (x, y, justify) of the visible text."""
        self.symbols.append({"lib_id": lib_id, "ref": ref, "value": value, "x": x, "y": y,
                             "angle": angle, "ref_at": ref_at, "value_at": value_at, "field_angle": 0})
        pins = [self.pin_xy(lib_id, p["number"], x, y, angle) for p in self.pin_tables[lib_id]]
        if len(pins) == 2:
            self.boxes.append((f"{ref}|body", _seg_box(pins[0], pins[1], 1.524)))
        if ref_at:
            self.boxes.append((f"{ref}|ref", _text_box(ref, *ref_at)))
        if value_at:
            self.boxes.append((f"{ref}|value", _text_box(value, *value_at)))
        return pins

    def power(self, net, x, y, direction):
        """Power symbol whose graphic points along `direction` (sheet coords)."""
        angle = next(a for a in (0, 90, 180, 270)
                     if tuple(round(c) for c in power_graphic_direction(f"power:{net}", a)) == direction)
        dx, dy = direction
        tip = (x + dx * POWER_LEN, y + dy * POWER_LEN)
        # Text is always centred; KiCad adds the symbol's rotation to the
        # field angle, so a symbol turned 90/270 gets field angle 90 to keep
        # its net name horizontal (checked with kicad-cli 7 SVG export).
        if dx == 0:      # vertical symbol: text centred just past the graphic
            text_at = (x, y + dy * (POWER_LEN + 1.27), "center")
        else:            # horizontal symbol: text continues outward
            text_at = (round(x + dx * (POWER_LEN + 0.635 + len(net) * CHAR_W / 2), 3), y, "center")
        ref = self.next_ref("#PWR")
        self.symbols.append({"lib_id": f"power:{net}", "ref": ref, "value": net, "x": x, "y": y,
                             "angle": angle, "ref_at": None, "value_at": text_at,
                             "field_angle": 90 if angle in (90, 270) else 0})
        self.boxes.append((f"{ref}|{net} symbol", _seg_box((x, y), tip, 1.27)))
        self.boxes.append((f"{ref}|{net} text", _text_box(net, *text_at)))

    def wire(self, p1, p2):
        self.wires.append((p1, p2))

    def label(self, text, x, y, toward):
        """Net label at (x, y); its text runs toward -1 (left) or +1 (right)."""
        angle = 0 if toward > 0 else 180
        self.labels.append({"text": text, "x": x, "y": y, "angle": angle})
        w = len(text) * CHAR_W + 0.5
        self.boxes.append((f'label {text} @({x}, {y})|text', (min(x, x + toward * w), y - LABEL_TEXT_H, max(x, x + toward * w), y)))

    def junction(self, p):
        self.junctions.append(p)

    def no_connect(self, p):
        self.no_connects.append(p)

    # -- block handling ------------------------------------------------------
    def bbox(self):
        return _box_union(b for _, b in self.boxes)

    def shift(self, dx, dy):
        mv = lambda p: (round(p[0] + dx, 3), round(p[1] + dy, 3))
        for s in self.symbols:
            s["x"], s["y"] = round(s["x"] + dx, 3), round(s["y"] + dy, 3)
            for key in ("ref_at", "value_at"):
                if s[key]:
                    s[key] = (round(s[key][0] + dx, 3), round(s[key][1] + dy, 3), s[key][2])
        self.wires = [(mv(a), mv(b)) for a, b in self.wires]
        for lb in self.labels:
            lb["x"], lb["y"] = round(lb["x"] + dx, 3), round(lb["y"] + dy, 3)
        self.junctions = [mv(p) for p in self.junctions]
        self.no_connects = [mv(p) for p in self.no_connects]
        self.boxes = [(o, (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)) for o, b in self.boxes]

    def merge(self, other: "Sheet"):
        for attr in ("symbols", "wires", "labels", "junctions", "no_connects", "boxes"):
            getattr(self, attr).extend(getattr(other, attr))

    def overlaps(self):
        """Pairs of boxes from DIFFERENT parts/texts that intersect (a part
        and its own ref/value text are allowed to touch)."""
        out = []
        shrink = 0.2
        items = [(o, (b[0] + shrink, b[1] + shrink, b[2] - shrink, b[3] - shrink)) for o, b in self.boxes]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0].split("|")[0] == items[j][0].split("|")[0]:
                    continue                  # a part and its own text
                if _overlap(items[i][1], items[j][1]):
                    out.append((items[i][0], items[j][0]))
        return out


class _RefCounter:
    """Per-call reference-designator counter (no module-global state, so
    repeated build_minimal_system() calls produce identical output)."""

    def __init__(self):
        self._counts = {}

    def __call__(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}{self._counts[prefix]}"


# ---------------------------------------------------------------------------
# blocks (each built around (0, 0) on its own Sheet, measured, then moved)
# ---------------------------------------------------------------------------

def _vertical_cap(sh: Sheet, ref, value, x, top_y):
    """Capacitor standing up; returns (top pin, bottom pin)."""
    cy = top_y + TWO_PIN_PARTS["Device:C"][0]
    pins = sh.part("Device:C", ref, value, x, cy, 0, (x + 2.54, cy - 1.27, "left"), (x + 2.54, cy + 1.27, "left"))
    return min(pins, key=lambda p: p[1]), max(pins, key=lambda p: p[1])


def _vertical_res(sh: Sheet, ref, value, x, top_y):
    cy = top_y + TWO_PIN_PARTS["Device:R"][0]
    pins = sh.part("Device:R", ref, value, x, cy, 0, (x + 2.54, cy - 1.27, "left"), (x + 2.54, cy + 1.27, "left"))
    return min(pins, key=lambda p: p[1]), max(pins, key=lambda p: p[1])


def _decoupling_block(tables, caps: list, next_ref) -> Sheet:
    """caps: [(top_net, is_label)] -- one 100nF per supply pin, GND below."""
    sh = Sheet(tables, next_ref)
    for i, (net, is_label) in enumerate(caps):
        x = i * CAP_PITCH
        top, bottom = _vertical_cap(sh, next_ref("C"), "100nF", x, 0.0)
        if is_label:
            up = (top[0], top[1] - GRID)
            sh.wire(top, up)
            sh.label(net, *up, toward=+1)
        else:
            sh.power(net, *top, (0, -1))
        sh.power("GND", *bottom, (0, 1))
    return sh


def _reset_block(tables, net_label: str, supply: str, with_cap: bool, next_ref) -> Sheet:
    sh = Sheet(tables, next_ref)
    top, bottom = _vertical_res(sh, next_ref("R"), "10k", 0.0, 0.0)
    sh.power(supply, *top, (0, -1))
    node = (bottom[0], bottom[1] + GRID)
    sh.wire(bottom, node)
    sh.label(net_label, *node, toward=-1)
    if with_cap:
        cap_top = (node[0] + CAP_PITCH, node[1])
        sh.wire(node, cap_top)
        sh.junction(node)
        _, cap_bottom = _vertical_cap(sh, next_ref("C"), "1uF", cap_top[0], cap_top[1])
        sh.power("GND", *cap_bottom, (0, 1))
    return sh


def _crystal_block(tables, labels: tuple, value: str, next_ref) -> Sheet:
    sh = Sheet(tables, next_ref)
    half = TWO_PIN_PARTS["Device:Crystal"][0]  # horizontal part
    span = 2 * half + 2 * GRID * 2          # crystal + a wire on each side
    left_node, right_node = (0.0, 0.0), (span, 0.0)
    cx = span / 2
    p1, p2 = sh.part("Device:Crystal", next_ref("Y"), value, cx, 0.0, 0,
                     (cx, -6.35, "center"), (cx, -3.81, "center"))   # both texts above: the load caps' text sits below
    sh.wire(left_node, min(p1, p2))
    sh.wire(max(p1, p2), right_node)
    for node, text, toward in ((left_node, labels[0], -1), (right_node, labels[1], +1)):
        up = (node[0], node[1] - 2 * GRID)
        sh.wire(node, up)
        sh.label(text, *up, toward=toward)
        sh.junction(node)
        _, cap_bottom = _vertical_cap(sh, next_ref("C"), "22pF", node[0], node[1])
        sh.power("GND", *cap_bottom, (0, 1))
    return sh


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------

def _label_name(pin: dict) -> str:
    for part in pin["name"].split("/"):
        if part.startswith("XTAL") or part in ("RESET", "EN", "AREF"):
            return part
    return pin["name"].split("/")[-1]


def build_minimal_system(ic: str = "atmega328p") -> str:
    """A minimal working system around `ic`; returns a full .kicad_sch text."""
    ic_lib_id = resolve_ic(ic)
    entry = get_part(ic_lib_id)
    supply = entry.get("supply", "+5V")
    pins = get_pins_with_layout(ic_lib_id)
    half_w = body_half_width(ic_lib_id)
    tables = {ic_lib_id: pins}
    for lib_id in TWO_PIN_PARTS:
        tables[lib_id] = two_pin_local_pins(lib_id)
    next_ref = _RefCounter()

    # --- the IC and its pin stubs -------------------------------------
    sheet = Sheet(tables, next_ref)
    ys = [p["y"] for p in pins]
    sheet.part(ic_lib_id, "U1", ic_lib_id.split(":", 1)[1], 0.0, 0.0, 0,
               (0.0, -(max(ys) + GRID) - 2.54, "center"), (0.0, -(min(ys) - GRID) + 2.54, "center"))
    sheet.boxes.append(("U1|body", (-half_w, -(max(ys) + GRID), half_w, -(min(ys) - GRID))))

    decouple_caps, reset_net, xtal_nets = [], None, []
    tips = {p["number"]: transform_point(p["x"], p["y"], None, 0, 0.0, 0.0) for p in pins}
    by_spot = {tips[p["number"]]: p for p in pins}
    for pin in pins:
        tip = tips[pin["number"]]
        out = -1 if tip[0] < 0 else 1
        stub = STUB
        fn = pin["function"]
        above = by_spot.get((tip[0], round(tip[1] - GRID, 3)))
        if fn not in ("power", "supply_out", "ground", "gpio", "nc") and above \
                and above["function"] in ("power", "supply_out", "ground"):
            # this pin gets a label, whose text sits above its wire -- right
            # where the neighbour's sideways power symbol is. Run the wire out
            # past that symbol and its net name instead of overlapping them.
            above_net = "GND" if above["function"] == "ground" else above.get("net", supply)
            clear = STUB + POWER_LEN + 0.635 + len(above_net) * CHAR_W + GRID
            stub = -(-clear // GRID) * GRID
        end = (round(tip[0] + out * stub, 3), tip[1])
        if fn in ("gpio", "nc"):
            sheet.no_connect(tip)
            continue
        sheet.wire(tip, end)
        if fn in ("power", "supply_out"):
            net = pin.get("net", supply)
            sheet.power(net, *end, (out, 0))
            if fn == "power":
                decouple_caps.append((net, False))
        elif fn == "ground":
            sheet.power("GND", *end, (out, 0))
        else:
            name = _label_name(pin)
            sheet.label(name, *end, toward=out)
            if fn == "reset":
                reset_net = name
            elif fn == "xtal":
                xtal_nets.append(name)
            elif fn == "aref":
                decouple_caps.append((name, True))

    # --- support blocks, stacked in columns left of the IC --------------
    blocks = []
    if decouple_caps:
        blocks.append(_decoupling_block(tables, decouple_caps, next_ref))
    if reset_net:
        blocks.append(_reset_block(tables, reset_net, supply, entry.get("reset_cap", False), next_ref))
    if len(xtal_nets) == 2:
        blocks.append(_crystal_block(tables, tuple(sorted(xtal_nets)), entry.get("crystal", "16MHz"), next_ref))

    ic_box = sheet.bbox()
    col_right = ic_box[0] - BLOCK_GAP
    col_top = ic_box[1]
    y = col_top
    col_width = 0.0
    for block in blocks:
        bx = block.bbox()
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        if y > col_top and y + h > ic_box[3]:          # column full: start a new one further left
            col_right -= col_width + BLOCK_GAP
            y, col_width = col_top, 0.0
        block.shift(_snap(col_right - bx[2]), _snap(y - bx[1]))
        sheet.merge(block)
        y = block.bbox()[3] + BLOCK_GAP
        col_width = max(col_width, w)

    clashes = sheet.overlaps()
    if clashes:
        raise AssertionError(f"layout produced overlapping items: {clashes[:5]}")

    # --- onto the paper -------------------------------------------------
    bx = sheet.bbox()
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    paper, pw, ph = next((p for p in PAPERS if w + 2 * PAGE_MARGIN <= p[1] and h + 2 * PAGE_MARGIN + TITLE_BLOCK_H <= p[2]),
                         PAPERS[-1])
    sheet.shift(_snap(PAGE_MARGIN - bx[0]), _snap(PAGE_MARGIN - bx[1]))
    return _serialize(sheet, ic_lib_id, pins, half_w, paper)


def _uid(*parts) -> str:
    return str(uuid.uuid5(NS, "/".join(str(p) for p in parts)))


def _serialize(sheet: Sheet, ic_lib_id: str, pins: list, half_w: float, paper: str) -> str:
    nets = sorted({s["value"] for s in sheet.symbols if s["lib_id"].startswith("power:")})
    used = {s["lib_id"] for s in sheet.symbols}
    lib = _ic_lib(ic_lib_id, pins, half_w) + "".join(_two_pin_lib(l) for l in TWO_PIN_PARTS if l in used) \
        + "".join(_power_lib(n) for n in nets)

    def field(name, text, at, hidden=False, angle=0):
        if at is None:
            return f'(property "{name}" "{text}" (at 0 0 0) {_HIDE})'
        x, y, justify = at
        just = "" if justify == "center" else f" (justify {justify})"
        hide = " hide" if hidden else ""
        return f'(property "{name}" "{text}" (at {x} {y} {angle}) (effects (font (size 1.27 1.27)){just}){hide})'

    out = [f'(kicad_sch (version {FILE_VERSION}) (generator circuit_ic_generator)',
           f'  (uuid {_uid(ic_lib_id, "sheet")})',
           f'  (paper "{paper}")',
           f'  (lib_symbols{lib}\n  )']
    for j in sheet.junctions:
        out.append(f'  (junction (at {j[0]} {j[1]}) (diameter 0) (color 0 0 0 0) (uuid {_uid("j", *j)}))')
    for p in sheet.no_connects:
        out.append(f'  (no_connect (at {p[0]} {p[1]}) (uuid {_uid("nc", *p)}))')
    for a, b in sheet.wires:
        out.append(f'  (wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) (stroke (width 0) (type default)) (uuid {_uid("w", *a, *b)}))')
    for lb in sheet.labels:
        just = "left" if lb["angle"] == 0 else "right"
        out.append(f'  (label "{lb["text"]}" (at {lb["x"]} {lb["y"]} {lb["angle"]}) (fields_autoplaced)'
                   f' (effects (font (size 1.27 1.27)) (justify {just} bottom)) (uuid {_uid("l", lb["text"], lb["x"], lb["y"])}))')
    for s in sheet.symbols:
        is_power = s["lib_id"].startswith("power:")
        pin_numbers = [p["number"] for p in sheet.pin_tables.get(s["lib_id"], [{"number": "1"}])]
        pin_uuids = " ".join(f'(pin "{n}" (uuid {_uid(s["ref"], "pin", n)}))' for n in pin_numbers)
        out.append(
            f'  (symbol (lib_id "{s["lib_id"]}") (at {s["x"]} {s["y"]} {s["angle"]}) (unit 1)'
            f' (in_bom {"no" if is_power else "yes"}) (on_board yes) (dnp no) (uuid {_uid(s["ref"])})\n'
            f'    {field("Reference", s["ref"], s["ref_at"], hidden=is_power, angle=s["field_angle"])}\n'
            f'    {field("Value", s["value"], s["value_at"], angle=s["field_angle"])}\n'
            f'    (property "Footprint" "" (at {s["x"]} {s["y"]} 0) {_HIDE})\n'
            f'    (property "Datasheet" "" (at {s["x"]} {s["y"]} 0) {_HIDE})\n'
            f'    {pin_uuids})')
    out.append('  (sheet_instances (path "/" (page "1")))\n)')
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate an MCU minimal-system .kicad_sch")
    parser.add_argument("--ic", default="atmega328p",
                        help="atmega328p/uno, atmega2560/mega2560, esp32, pi5 (see ic_library.IC_ALIASES)")
    parser.add_argument("--out", type=Path, required=True, help="output .kicad_sch path")
    parser.add_argument("--check", action="store_true", help="run circuit_rule_check's ERC after generating")
    parser.add_argument("--render", type=Path, help="also render to this .svg/.png via circuit_kicad_to_schemdraw")
    args = parser.parse_args()

    args.out.write_text(build_minimal_system(args.ic), encoding="utf-8")
    print(f"wrote {args.out}")

    if args.check:
        from circuit_rule_check import check_schematic
        findings = check_schematic(args.out)
        if findings:
            for f in findings:
                print(f'  [{f["severity"]}] {f["rule"]}: {f["message"]}')
        else:
            print("  no ERC findings")

    if args.render:
        from circuit_kicad_to_schemdraw import render_file
        unmapped, imprecise = render_file(args.out, args.render)
        print(f"rendered {args.render}")
        if unmapped:
            print(f"  unmapped lib_ids: {sorted(unmapped)}")
        if imprecise:
            print(f"  imprecise placement: {sorted(imprecise)}")


if __name__ == "__main__":
    main()
