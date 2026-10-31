"""KiCad schematic (.kicad_sch) -> schemdraw rendering.

Different input modality from circuit_schemdraw_render.py (which draws from
fuzzy YOLO photo detections and deliberately skips wires because pixel
proximity can't prove two leads are joined). A .kicad_sch file has no such
ambiguity: every placed symbol's (x, y, angle) and every wire's exact
endpoints are already numbers in the file (same ground truth
circuit_rule_check.py's ERC reasons over) -- so this module draws the real
wires too, not just component symbols.

Reuses parse_sexpr (kicad_dataset_convert.py) and get_placed_instances /
get_wires / get_junctions (circuit_rule_check.py) instead of re-parsing the
S-expression format a third time in this repo.

Symbol shapes are approximate: lib_id is matched against a small prefix
table (extendable via `class_map`, same "merge into defaults" convention as
kicad_dataset_convert.py's --class-map) to a generic schemdraw part (a
"Device:C_US" and a "Device:C" both just become a generic Capacitor -- this
draws the circuit's topology and layout faithfully, not each symbol's exact
IEC/ANSI graphic). An unmapped lib_id (e.g. a specific MCU or connector
part number) is rendered as a labeled placeholder box, not guessed at, and
returned in `unmapped` so the caller can see what wasn't drawn precisely.

2-terminal parts (R/C/L/D/LED/Crystal/Switch) are anchored at their real
pin coordinates -- reusing get_lib_pin_table/pins_for_instance/
transform_point from circuit_rule_check.py (the same pin-transform math
already verified there against a real file) -- and stretched with
schemdraw's `.to()` between the two, so a wire drawn to that same pin
coordinate visually lands exactly on the lead instead of near the
component's placement origin.

A part with more than 2 resolved pins (an IC, connector, etc. -- the
ATmega2560-16A this module was first tried against is a 100-pin example)
gets a generic body box instead of a specific schemdraw part, sized to
its real pins' spread, with one lead stub per real pin still ending
exactly at that pin's real world coordinate -- so wires still visually
land correctly even though the box itself is a rectangle, not the part's
real footprint. There is no pin `length`/`angle` parsed by
circuit_rule_check.py to compute the exact body-edge point, so a fixed
IC_LEAD_INSET_MM stub stands in for it (see _ic_body_point) -- reasonable
for pins on 2 opposite sides (most connectors/ICs drawn in KiCad), not
verified for a part with pins on all 4 sides. Rotation is not applied to
this box (real MCU/connector symbols in practice are placed at angle 0),
a known gap flagged here rather than silently assumed correct.

A part resolving to exactly 0 pins (a lib_id with no lib_symbols entry at
all, e.g. still unmapped) falls back to a single origin-anchored
placeholder box with best-effort rotation (schemdraw's .theta(-angle),
compensating for the y-axis flip, not verified against a rendered image).

Both the >2-pin box and the 0-pin fallback are reported in `imprecise`
rather than silently drawn as if they were as accurate as a 2-terminal
part's placement.

Usage:
    python circuit_kicad_to_schemdraw.py path/to/one.kicad_sch --out out.svg
"""

import argparse
from pathlib import Path

import schemdraw
import schemdraw.elements as elm

import math

from circuit_rule_check import (get_junctions, get_labels, get_lib_pin_table, get_placed_instances,
                                 get_wires, pins_for_instance, power_graphic_direction, transform_point)
from kicad_dataset_convert import parse_sexpr

MM_PER_UNIT = 5.0  # 1 schemdraw unit per 5mm of real schematic -> comfortably spaced canvas
IC_LEAD_INSET_MM = 2.54  # visual lead-stub length (one KiCad grid step) for a >2-pin body box

DEFAULT_CLASS_MAP = {
    "Device:LED": elm.LED,          # must come before "Device:D" below
    "Device:D": elm.Diode,
    "Device:R": elm.Resistor,
    "Device:C": elm.Capacitor,      # matches both "Device:C" and "Device:C_US"
    "Device:L": elm.Inductor,
    "Device:Crystal": elm.Crystal,
    "Switch:": elm.Switch,
    "power:GND": elm.Ground,
}


def _lookup_element(lib_id: str, class_map: dict):
    for prefix, element_cls in class_map.items():
        if lib_id.startswith(prefix):
            return element_cls
    return None


def _to_drawing_xy(x_mm: float, y_mm: float, min_x: float, max_y: float) -> tuple:
    return (x_mm - min_x) / MM_PER_UNIT, (max_y - y_mm) / MM_PER_UNIT


def _tip_world(pin: dict, inst: dict) -> tuple:
    return transform_point(pin["x"], pin["y"], inst["mirror"], inst["angle"], inst["x"], inst["y"])


def _ic_body_point(pin: dict) -> tuple:
    """Pull a pin's local (x, y) tip inward by IC_LEAD_INSET_MM along
    whichever axis it points, as a stand-in body-edge point (see module
    docstring for why this is a fixed stub, not the pin's real length)."""
    x, y = pin["x"], pin["y"]
    angle = pin.get("angle")
    if angle is not None:
        # KiCad stores each pin's direction (tip -> body, library y-up):
        # use it instead of guessing the side from the coordinates, which
        # put the top/bottom pins of a TALL 2-sided part (ATmega2560: pins
        # at y=+-62mm but x=+-15mm) on the wrong edge.
        step = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}.get(int(angle) % 360)
        if step:
            return x + IC_LEAD_INSET_MM * step[0], y + IC_LEAD_INSET_MM * step[1]
    if abs(x) >= abs(y):
        return x - IC_LEAD_INSET_MM * (1 if x > 0 else -1), y
    return x, y - IC_LEAD_INSET_MM * (1 if y > 0 else -1)


def _draw_box(d, corner1: tuple, corner2: tuple) -> None:
    (x1, y1), (x2, y2) = corner1, corner2
    for p1, p2 in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                   ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        d += elm.Line().at(p1).to(p2)


PIN_NAME_FONTSIZE = 7


def _draw_ic(d, xy, pins: list, inst: dict, label: str) -> None:
    bodies_world = []
    for pin in pins:
        tip_world = _tip_world(pin, inst)
        body_world = transform_point(*_ic_body_point(pin), inst["mirror"], inst["angle"],
                                      inst["x"], inst["y"])
        bodies_world.append(body_world)
        d += elm.Line().at(xy(*body_world)).to(xy(*tip_world))
        name = pin.get("name")
        if name and name not in ("~", "?"):   # "?" = no (name ...) in the file
            # pin name just inside the body edge, reading away from the pin
            inward = 1 if tip_world[0] < body_world[0] else -1
            inside = (body_world[0] + inward * 0.6, body_world[1])
            d += elm.Label().at(xy(*inside)).label(
                name, halign="left" if inward > 0 else "right", fontsize=PIN_NAME_FONTSIZE)

    xs, ys = [p[0] for p in bodies_world], [p[1] for p in bodies_world]
    # pins on the left/right edges: leave half a pin pitch above the first
    # and below the last pin so their names are inside the box, not on its edge
    pad = 1.27 if all(p.get("angle") in (0, 180, 0.0, 180.0) for p in pins) else 0.0
    _draw_box(d, xy(min(xs), min(ys) - pad), xy(max(xs), max(ys) + pad))
    d += elm.Label(label).at(xy((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))


def kicad_to_drawing(root, class_map: dict = None) -> tuple:
    """Returns (schemdraw.Drawing, unmapped_lib_ids: set, imprecise_references: set)."""
    class_map = {**DEFAULT_CLASS_MAP, **(class_map or {})}
    pin_table = get_lib_pin_table(root)
    instances = get_placed_instances(root)
    wires = get_wires(root)
    junctions = get_junctions(root)

    all_x = [i["x"] for i in instances] + [p[0] for w in wires for p in w]
    all_y = [i["y"] for i in instances] + [p[1] for w in wires for p in w]
    min_x, max_y = min(all_x, default=0.0), max(all_y, default=0.0)

    def xy(x, y):
        return _to_drawing_xy(x, y, min_x, max_y)

    d = schemdraw.Drawing()
    unmapped, imprecise = set(), set()

    for a, b in wires:
        d += elm.Line().at(xy(*a)).to(xy(*b))
    for point in junctions:
        d += elm.Dot().at(xy(*point))

    for lb in get_labels(root):
        # net label: text above its wire, running away from the wire end
        d += elm.Label().at(xy(*lb["point"])).label(
            lb["text"], halign="left" if lb.get("angle", 0) == 0 else "right", valign="bottom")

    for inst in instances:
        if inst["lib_id"].startswith("power:"):
            _draw_power(d, xy, inst)
            continue
        element_cls = _lookup_element(inst["lib_id"], class_map)
        label = inst["reference"] or inst["lib_id"]
        if element_cls is None:
            unmapped.add(inst["lib_id"])

        pins = pins_for_instance(pin_table, inst["lib_id"], inst["unit"])
        if element_cls and len(pins) == 2:
            p1, p2 = _tip_world(pins[0], inst), _tip_world(pins[1], inst)
            text = f'{label}\n{inst["value"]}' if inst.get("value") else label
            d += element_cls().at(xy(*p1)).to(xy(*p2)).label(text)
        elif element_cls and len(pins) == 1:
            d += element_cls().at(xy(*_tip_world(pins[0], inst))).label(label)
        elif len(pins) > 2:
            imprecise.add(label)
            _draw_ic(d, xy, pins, inst, label)
        else:
            imprecise.add(label)
            placeholder_cls = elm.Dot if inst["lib_id"].startswith("power:") else elm.RBox
            d += placeholder_cls().at(xy(inst["x"], inst["y"])).theta(-inst["angle"]).label(label)

    return d, unmapped, imprecise


def _draw_power(d, xy, inst: dict) -> None:
    """Power flag drawn at its pin, pointing the way it points in the file
    (circuit_rule_check.power_graphic_direction), labelled with its net --
    instead of a dot carrying the hidden "#PWR.." reference."""
    net = inst["lib_id"].split(":", 1)[1]
    dx, dy = power_graphic_direction(inst["lib_id"], inst["angle"])
    target = math.degrees(math.atan2(-dy, dx))           # sheet y-down -> drawing y-up
    is_ground = net.upper().startswith(("GND", "VSS", "AGND", "DGND"))
    base = -90.0 if is_ground else 90.0                  # schemdraw Ground hangs down, Vdd stands up
    d += (elm.Ground if is_ground else elm.Vdd)().at(xy(inst["x"], inst["y"])).theta(target - base)
    if not is_ground:
        # net name just past the flag, reading away from it (a schemdraw
        # element label would rotate/sit on the wire for sideways flags)
        end = (inst["x"] + dx * 4.5, inst["y"] + dy * 4.5)
        halign = "center" if abs(dx) < 0.5 else ("left" if dx > 0 else "right")
        d += elm.Label().at(xy(*end)).label(net, halign=halign, valign="bottom" if dy < -0.5 else "center")


def render_file(sch_path: Path, out_path: Path, class_map: dict = None) -> tuple:
    root = parse_sexpr(sch_path.read_text(encoding="utf-8"))
    drawing, unmapped, imprecise = kicad_to_drawing(root, class_map)
    drawing.save(str(out_path))
    return unmapped, imprecise


def main():
    parser = argparse.ArgumentParser(description="Render a .kicad_sch file as a schemdraw diagram")
    parser.add_argument("sch", type=Path, help=".kicad_sch file")
    parser.add_argument("--out", type=Path, required=True, help="output image (.svg/.png)")
    args = parser.parse_args()

    unmapped, imprecise = render_file(args.sch, args.out)
    print(f"saved {args.out}")
    if unmapped:
        print(f"unmapped lib_ids (drawn as placeholder boxes): {sorted(unmapped)}")
    if imprecise:
        print(f"imprecise placement (origin-anchored, not pin-accurate): {sorted(imprecise)}")


if __name__ == "__main__":
    main()
