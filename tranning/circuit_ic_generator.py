"""Deterministic generator for an MCU minimal-system schematic.

Generation approach, and why it is deterministic code (not a trained model,
same "algorithm not neural net" category as circuit_rule_check.py and
calculus_generator.py -- see that docstring for the Rule 06 reasoning): the
project already tried a neural-net circuit generator once
(schmitt_trigger_train.py) and it still does not produce balanced/usable
output after 5 attempts (ErrorLog #23). Rather than repeat that failure mode
for a second, more complex part, this module builds an MCU minimal-system
schematic (decoupling caps, oscillator + load caps, reset pull-up) directly
from a fixed, well-known circuit topology -- the same "decision-tree
algorithm" choice already made for calculus/word-problem/logic-quiz
generation and for the ERC checker.

The pin data comes from ic_library.py (public datasheet pinout, no real
.kicad_sch reference file was available -- see that module's docstring).
The output is still a syntactically real .kicad_sch this project's OWN
existing tools already understand: parsed by the exact same parse_sexpr
(kicad_dataset_convert.py), checked by the exact same check_schematic
(circuit_rule_check.py, including the missing_decoupling_cap /
missing_reset_pullup / missing_crystal_load_caps rules added alongside this
module), and rendered by the exact same kicad_to_drawing
(circuit_kicad_to_schemdraw.py) real .kicad_sch files use -- nothing new to
parse/check/render, only something new to write.

Placement, deliberately simple: every accessory part (pull-up resistor,
decoupling/load capacitors) is placed with `_attach_two_pin()`, which
rotates the part so ONE pin lands on an exact target coordinate (an IC pin
tip) and returns the OTHER pin's exact world coordinate so the caller can
chain a ground/power symbol onto it -- reusing the same rotate-then-translate
math `circuit_rule_check.transform_point()` already uses (verified there
against a real file), so a generated file's geometry is internally
consistent with how this project's own checker/renderer read real KiCad
geometry. The crystal is the one exception: its fixed 5.08mm pin pitch does
not match the IC's 2.54mm adjacent-pin pitch, so it is connected with two
explicit wires instead of forced pin-coincidence (the same "wires, not
pin-touching, for anything that isn't a straight chain" convention the
hand-written test fixtures in test_circuit_rule_check.py already use).

Usage:
    python circuit_ic_generator.py --ic atmega328p --out minimal_system.kicad_sch
    python circuit_ic_generator.py --ic atmega328p --out out.kicad_sch --check --render out.svg
"""

import argparse
import math
from pathlib import Path

from ic_library import get_pins_with_layout, resolve_ic

PIN_LEN = 2.54
_DIRECTION_ANGLE = {(-1, 0): 180, (1, 0): 0, (0, 1): 90, (0, -1): 270}


def _rotate(x: float, y: float, angle_deg: float) -> tuple:
    theta = math.radians(angle_deg)
    c, s = round(math.cos(theta), 6), round(math.sin(theta), 6)
    return x * c - y * s, x * s + y * c


def _symbol_instance(lib_id: str, ref: str, x: float, y: float, angle: float = 0.0) -> str:
    # same shape as the proven-working fixture in test_circuit_rule_check.py
    return f"""
    (symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) (uuid "{ref}-uuid")
        (property "Reference" "{ref}" (at {x} {y} {angle}) (effects (font (size 1.27 1.27))))
    )
    """


def _wire(p1: tuple, p2: tuple) -> str:
    return f'(wire (pts (xy {p1[0]} {p1[1]}) (xy {p2[0]} {p2[1]})) (stroke (width 0) (type default)))'


def _attach_two_pin(target: tuple, direction: tuple, lib_id: str, ref: str) -> tuple:
    """Place a 2-pin part with its near pin exactly on `target`, extending
    away along `direction` (an axis-aligned unit vector). Returns
    (instance_text, far_pin_world_xy)."""
    angle = _DIRECTION_ANGLE[direction]
    near_off, far_off = _rotate(-2.54, 0.0, angle), _rotate(2.54, 0.0, angle)
    inst_x, inst_y = round(target[0] - near_off[0], 3), round(target[1] - near_off[1], 3)
    far_world = (round(inst_x + far_off[0], 3), round(inst_y + far_off[1], 3))
    return _symbol_instance(lib_id, ref, inst_x, inst_y, angle), far_world


def _two_pin_lib(lib_id: str, electrical_type: str = "passive") -> str:
    part = lib_id.split(":", 1)[-1]
    return f"""
        (symbol "{lib_id}"
            (symbol "{part}_1_1"
                (pin {electrical_type} line (at -2.54 0 0) (length 2.54)
                    (name "1" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin {electrical_type} line (at 2.54 0 0) (length 2.54)
                    (name "2" (effects (font (size 1.27 1.27))))
                    (number "2" (effects (font (size 1.27 1.27)))))
            )
        )"""


def _one_pin_power_lib(lib_id: str, pin_name: str) -> str:
    part = lib_id.split(":", 1)[-1]
    return f"""
        (symbol "{lib_id}"
            (power)
            (symbol "{part}_0_1"
                (pin power_in line (at 0 0 90) (length 0)
                    (name "{pin_name}" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
            )
        )"""


def _ic_lib(lib_id: str, pins: list) -> str:
    part = lib_id.split(":", 1)[-1]
    pin_lines = "\n".join(
        f'                (pin {p["electrical_type"]} line (at {p["x"]} {p["y"]} '
        f'{0 if p["x"] >= 0 else 180}) (length {PIN_LEN})\n'
        f'                    (name "{p["name"]}" (effects (font (size 1.27 1.27))))\n'
        f'                    (number "{p["number"]}" (effects (font (size 1.27 1.27)))))'
        for p in pins
    )
    return f'\n        (symbol "{lib_id}"\n            (symbol "{part}_1_1"\n{pin_lines}\n            )\n        )'


class _RefCounter:
    """Per-call reference-designator counter (no module-global state, so
    repeated build_minimal_system() calls in the same process -- e.g. in
    tests -- don't leak counters between runs)."""

    def __init__(self):
        self._counts = {}

    def __call__(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}{self._counts[prefix]}"


def build_minimal_system(ic: str = "atmega328p") -> str:
    """A minimal working system around `ic`: reset pull-up, VCC/AVCC/AREF
    decoupling caps, crystal oscillator with load caps, GND pins grounded.
    Returns a full .kicad_sch document as text."""
    ic_lib_id = resolve_ic(ic)
    pins = get_pins_with_layout(ic_lib_id)
    by_function = {}
    for p in pins:
        by_function.setdefault(p["function"], []).append(p)

    next_ref = _RefCounter()
    body = [_symbol_instance(ic_lib_id, "U1", 0, 0)]

    def decouple(target: tuple, direction: tuple) -> None:
        cap, far = _attach_two_pin(target, direction, "Device:C", next_ref("C"))
        body.append(cap)
        body.append(_symbol_instance("power:GND", next_ref("#PWR"), *far, angle=90.0))

    def side(pin: dict) -> tuple:
        return (-1, 0) if pin["x"] < 0 else (1, 0)

    for pin in by_function.get("power", []):
        tip = (pin["x"], pin["y"])
        body.append(_symbol_instance("power:+5V", next_ref("#PWR"), *tip))
        decouple(tip, side(pin))

    for pin in by_function.get("ground", []):
        body.append(_symbol_instance("power:GND", next_ref("#PWR"), pin["x"], pin["y"], angle=90.0))

    for pin in by_function.get("aref", []):
        decouple((pin["x"], pin["y"]), side(pin))

    # GPIO pins are deliberately left for whoever uses this minimal system
    # to wire up -- an explicit (no_connect ...) marker is the real KiCad
    # convention for "yes, unwired on purpose", not a floating_pin mistake.
    for pin in by_function.get("gpio", []):
        body.append(f'(no_connect (at {pin["x"]} {pin["y"]}))')

    reset_pins = by_function.get("reset", [])
    if reset_pins:
        tip = (reset_pins[0]["x"], reset_pins[0]["y"])
        r1, r1_far = _attach_two_pin(tip, side(reset_pins[0]), "Device:R", next_ref("R"))
        body += [r1, _symbol_instance("power:+5V", next_ref("#PWR"), *r1_far)]

    xtal_pins = by_function.get("xtal", [])
    if len(xtal_pins) == 2:
        p_a, p_b = xtal_pins
        cx, cy = min(p_a["x"], p_b["x"]) - 10.0, round((p_a["y"] + p_b["y"]) / 2, 3)
        near_off, far_off = _rotate(-2.54, 0.0, 90.0), _rotate(2.54, 0.0, 90.0)
        y1_a = (round(cx + near_off[0], 3), round(cy + near_off[1], 3))
        y1_b = (round(cx + far_off[0], 3), round(cy + far_off[1], 3))
        body += [
            _symbol_instance("Device:Crystal", next_ref("Y"), cx, cy, 90.0),
            _wire((p_a["x"], p_a["y"]), y1_a),
            _wire((p_b["x"], p_b["y"]), y1_b),
        ]
        decouple(y1_a, (0, -1))
        decouple(y1_b, (0, -1))

    lib_symbols = (_ic_lib(ic_lib_id, pins) + _two_pin_lib("Device:R") + _two_pin_lib("Device:C")
                   + _two_pin_lib("Device:Crystal") + _one_pin_power_lib("power:GND", "GND")
                   + _one_pin_power_lib("power:+5V", "+5V"))

    return (f'(kicad_sch (version 20231120) (generator circuit_ic_generator) (paper "A4")\n'
            f'    (lib_symbols{lib_symbols}\n    )\n' + "\n".join(body) + "\n)")


def main():
    parser = argparse.ArgumentParser(description="Generate an MCU minimal-system .kicad_sch")
    parser.add_argument("--ic", default="atmega328p", help="known IC name/alias (see ic_library.IC_ALIASES)")
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
