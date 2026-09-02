"""Tests for circuit_rule_check.py.

Two layers, same pattern as test_kicad_dataset_convert.py-style modules in
this repo: (1) small hand-written synthetic .kicad_sch text covering one
rule each, so every finding type is pinned down exactly; (2) a regression
check using real numbers read directly out of a real file
("touch LED.kicad_sch", outside this repo) to make sure the pin-transform
math -- the part most likely to be silently wrong -- matches an actual
KiCad placement, not just an assumption.
"""

from pathlib import Path

import pytest

from circuit_rule_check import check_schematic, get_lib_pin_table, transform_point
from kicad_dataset_convert import parse_sexpr

# A minimal but structurally real "Test:R" two-pin part: pin 1 at local
# (-2.54, 0), pin 2 at local (2.54, 0), horizontal, unit 1. Small enough to
# hand-verify every coordinate in the tests below.
_LIB_R = """
    (lib_symbols
        (symbol "Test:R"
            (property "Reference" "R" (at 0 0 0) (effects (font (size 1.27 1.27))))
            (symbol "R_0_1"
                (rectangle (start -1 -0.5) (end 1 0.5))
            )
            (symbol "R_1_1"
                (pin passive line (at -2.54 0 0) (length 2.54)
                    (name "1" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 2.54 0 0) (length 2.54)
                    (name "2" (effects (font (size 1.27 1.27))))
                    (number "2" (effects (font (size 1.27 1.27)))))
            )
        )
        (symbol "power:GND"
            (power)
            (symbol "GND_0_1"
                (pin power_in line (at 0 0 90) (length 0)
                    (name "GND" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
            )
        )
    )
"""


def _symbol_instance(lib_id, ref, x, y, angle=0, uuid="00000000-0000-0000-0000-000000000001"):
    return f"""
    (symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) (uuid "{uuid}")
        (property "Reference" "{ref}" (at {x} {y} {angle}) (effects (font (size 1.27 1.27))))
    )
    """


def _wire(x1, y1, x2, y2):
    return f'(wire (pts (xy {x1} {y1}) (xy {x2} {y2})) (stroke (width 0) (type default)))'


def _write_sch(tmp_path: Path, body: str) -> Path:
    text = f'(kicad_sch (version 20231120) (generator test) (paper "A4") {_LIB_R} {body})'
    path = tmp_path / "test.kicad_sch"
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_real_touch_led_layout_pin_transform():
    """Regression against real numbers read out of "touch LED.kicad_sch":
    a Device:R_US resistor placed at (142.24, 53.34, 90) with library pins
    at local (0, 3.81) / (0, -3.81) must land exactly on the schematic's
    own wire endpoints (138.43, 53.34) and (146.05, 53.34)."""
    p1 = transform_point(0, 3.81, None, 90, 142.24, 53.34)
    p2 = transform_point(0, -3.81, None, 90, 142.24, 53.34)
    assert {p1, p2} == {(138.43, 53.34), (146.05, 53.34)}


def test_clean_circuit_has_no_findings(tmp_path):
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        + _symbol_instance("power:GND", "#PWR01", 97.46, 100, angle=90)
        + _wire(97.46, 100, 100, 100)
        + _wire(102.54, 100, 105, 100)
        + '(label "OUT" (at 105 100 0) (effects (font (size 1.27 1.27))))'
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    assert findings == []


def test_floating_pin_detected(tmp_path):
    # Only pin 1 (local -2.54,0 -> abs 97.46,100) gets a wire; pin 2 is left
    # dangling on purpose.
    body = _symbol_instance("Test:R", "R1", 100, 100) + _wire(90, 100, 97.46, 100)
    findings = check_schematic(_write_sch(tmp_path, body))
    floating = [f for f in findings if f["rule"] == "floating_pin"]
    assert len(floating) == 1
    assert floating[0]["point"] == (102.54, 100.0)


def test_no_connect_conflict_detected(tmp_path):
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        + _wire(90, 100, 97.46, 100)
        + _wire(102.54, 100, 110, 100)
        + "(no_connect (at 102.54 100))"
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    conflicts = [f for f in findings if f["rule"] == "no_connect_conflict"]
    assert len(conflicts) == 1
    assert not any(f["rule"] == "floating_pin" for f in findings)


def test_duplicate_reference_detected(tmp_path):
    body = (
        _symbol_instance("Test:R", "R1", 100, 100, uuid="00000000-0000-0000-0000-000000000001")
        + _symbol_instance("Test:R", "R1", 200, 100, uuid="00000000-0000-0000-0000-000000000002")
        + _wire(90, 100, 97.46, 100) + _wire(102.54, 100, 110, 100)
        + _wire(190, 100, 197.46, 100) + _wire(202.54, 100, 210, 100)
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    dups = [f for f in findings if f["rule"] == "duplicate_reference"]
    assert len(dups) == 1
    assert "R1" in dups[0]["message"]


def test_net_name_conflict_detected(tmp_path):
    # Two different global labels wired directly together -> unintended short.
    body = (
        '(global_label "GND" (at 100 100 0) (effects (font (size 1.27 1.27))))'
        + '(global_label "+5V" (at 110 100 0) (effects (font (size 1.27 1.27))))'
        + _wire(100, 100, 110, 100)
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    conflicts = [f for f in findings if f["rule"] == "net_name_conflict"]
    assert len(conflicts) == 1
    assert "GND" in conflicts[0]["message"] and "5V" in conflicts[0]["message"]


def test_missing_ground_warns_when_no_ground_net(tmp_path):
    lib_with_power_pin = _LIB_R.replace(
        '(pin passive line (at -2.54 0 0)', '(pin power_in line (at -2.54 0 0)'
    )
    text = f'(kicad_sch (version 20231120) (generator test) (paper "A4") {lib_with_power_pin} ' \
           f'{_symbol_instance("Test:R", "R1", 100, 100)} {_wire(90, 100, 97.46, 100)} ' \
           f'{_wire(102.54, 100, 110, 100)})'
    path = tmp_path / "test.kicad_sch"
    path.write_text(text, encoding="utf-8")
    findings = check_schematic(path)
    assert any(f["rule"] == "missing_ground" for f in findings)


def test_get_lib_pin_table_reads_unit_1_pins():
    root = parse_sexpr(f'(kicad_sch {_LIB_R})')
    table = get_lib_pin_table(root)
    pins = table["Test:R"][1]
    numbers = {p["number"] for p in pins}
    assert numbers == {"1", "2"}
