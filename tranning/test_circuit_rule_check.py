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


def test_transform_flips_library_y_axis_real_kicad_pot():
    """Regression against KiCad's own demo "pic_programmer.kicad_sch"
    (eeschema 10): RV1 (a potentiometer) placed at (243.84, 149.86, 90).
    Its wiper pin 2 is at library-local (0, 3.81) -- NOT symmetric about the
    origin, unlike the resistor above -- and the sheet's own wire ends at
    (240.03, 149.86). Library coords are y-up and sheet coords y-down, so
    without the y flip this pin would land at x=247.65, off the wire."""
    assert transform_point(0, 3.81, None, 90, 243.84, 149.86) == (240.03, 149.86)
    assert transform_point(-6.35, 0, None, 90, 243.84, 149.86) == (243.84, 156.21)
    assert transform_point(6.35, 0, None, 90, 243.84, 149.86) == (243.84, 143.51)
    # angle 0: a pin above the origin in the library is ABOVE on the sheet (smaller y)
    assert transform_point(0, 2.54, None, 0, 100, 100) == (100, 97.46)


def test_clean_circuit_has_no_findings(tmp_path):
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        # angle 270: the ground graphic points LEFT, away from R1's body
        # (at 90 it would point right, into the resistor -- a real overlap
        # the overlapping_symbols rule reports, see the tests at the end)
        + _symbol_instance("power:GND", "#PWR01", 97.46, 100, angle=270)
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


# ---------------------------------------------------------------------------
# MCU-aware checks: missing_decoupling_cap / missing_reset_pullup /
# missing_crystal_load_caps. A tiny 3-pin "Test:MCU" part (VCC/RESET/XTAL1,
# named exactly like a real MCU library so the *_RE name-matching in
# circuit_rule_check.py is what's actually under test) plus generic
# "Device:R"/"Device:C" two-pin parts and a "power:+5V" flag alongside the
# existing "power:GND" fixture.
# Library coords are y-up (see transform_point): VCC at local y=-5.08 lands
# at sheet y=+5.08 and XTAL1 at local y=+5.08 lands at sheet y=-5.08 -- the
# sheet positions the tests below wire to.
# ---------------------------------------------------------------------------

_LIB_MCU_EXTRAS = """
    (lib_symbols
        (symbol "Test:MCU"
            (symbol "MCU_1_1"
                (pin power_in line (at -2.54 -5.08 180) (length 2.54)
                    (name "VCC" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin input line (at -2.54 0 180) (length 2.54)
                    (name "RESET" (effects (font (size 1.27 1.27))))
                    (number "2" (effects (font (size 1.27 1.27)))))
                (pin passive line (at -2.54 5.08 180) (length 2.54)
                    (name "XTAL1" (effects (font (size 1.27 1.27))))
                    (number "3" (effects (font (size 1.27 1.27)))))
            )
        )
        (symbol "Device:R"
            (symbol "R_1_1"
                (pin passive line (at -2.54 0 0) (length 2.54)
                    (name "1" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 2.54 0 0) (length 2.54)
                    (name "2" (effects (font (size 1.27 1.27))))
                    (number "2" (effects (font (size 1.27 1.27)))))
            )
        )
        (symbol "Device:C"
            (symbol "C_1_1"
                (pin passive line (at -2.54 0 0) (length 2.54)
                    (name "1" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 2.54 0 0) (length 2.54)
                    (name "2" (effects (font (size 1.27 1.27))))
                    (number "2" (effects (font (size 1.27 1.27)))))
            )
        )
        (symbol "power:+5V"
            (power)
            (symbol "+5V_0_1"
                (pin power_in line (at 0 0 90) (length 0)
                    (name "+5V" (effects (font (size 1.27 1.27))))
                    (number "1" (effects (font (size 1.27 1.27)))))
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


def _write_mcu_sch(tmp_path: Path, body: str) -> Path:
    text = (f'(kicad_sch (version 20231120) (generator test) (paper "A4") '
            f'{_LIB_MCU_EXTRAS} {body})')
    path = tmp_path / "mcu.kicad_sch"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_decoupling_cap_flagged_when_vcc_pin_undecoupled(tmp_path):
    # MCU at (0,0): VCC pin -> abs (-2.54, 5.08). Wired only to a bare label,
    # no capacitor anywhere -- must be flagged.
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _wire(-10, 5.08, -2.54, 5.08)
        + '(label "VCC_NET" (at -10 5.08 0) (effects (font (size 1.27 1.27))))'
    )
    findings = check_schematic(_write_mcu_sch(tmp_path, body))
    assert any(f["rule"] == "missing_decoupling_cap" for f in findings)


def test_missing_decoupling_cap_clear_when_cap_bridges_to_ground(tmp_path):
    # Same VCC pin, but now a Device:C bridges its net straight to power:GND.
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _symbol_instance("Device:C", "C1", -7.54, 5.08)      # pins at -10.08,5.08 / -5.0,5.08
        + _wire(-5.0, 5.08, -2.54, 5.08)
        + _symbol_instance("power:GND", "#PWR01", -10.08, 5.08, angle=90)
    )
    findings = check_schematic(_write_mcu_sch(tmp_path, body))
    assert not any(f["rule"] == "missing_decoupling_cap" for f in findings)


def test_missing_reset_pullup_flagged_when_reset_pin_floats_free(tmp_path):
    # RESET pin -> abs (-2.54, 0). No pull-up resistor anywhere.
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _wire(-10, 0, -2.54, 0)
        + '(label "RESET_NET" (at -10 0 0) (effects (font (size 1.27 1.27))))'
    )
    findings = check_schematic(_write_mcu_sch(tmp_path, body))
    assert any(f["rule"] == "missing_reset_pullup" for f in findings)


def test_missing_reset_pullup_clear_when_resistor_bridges_to_power_net(tmp_path):
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _symbol_instance("Device:R", "R1", -7.54, 0)          # pins at -10.08,0 / -5.0,0
        + _wire(-5.0, 0, -2.54, 0)
        + _symbol_instance("power:+5V", "#PWR02", -10.08, 0)
    )
    findings = check_schematic(_write_mcu_sch(tmp_path, body))
    assert not any(f["rule"] == "missing_reset_pullup" for f in findings)


def test_missing_crystal_load_caps_flagged_when_xtal_pin_undecoupled(tmp_path):
    # XTAL1 pin -> abs (-2.54, -5.08). No capacitor to ground anywhere.
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _wire(-10, -5.08, -2.54, -5.08)
    )
    findings = check_schematic(_write_mcu_sch(tmp_path, body))
    assert any(f["rule"] == "missing_crystal_load_caps" for f in findings)


# ---------------------------------------------------------------------------
# 2026-10-27: same-name nets + overlapping_symbols
# ---------------------------------------------------------------------------

def test_same_name_power_symbols_are_one_net(tmp_path):
    # The cap is NOT wired to the MCU: it sits elsewhere between its own +5V
    # and GND symbols. KiCad joins every "+5V" on a sheet, so the VCC pin
    # (which only touches a +5V symbol) IS decoupled.
    body = (
        _symbol_instance("Test:MCU", "U1", 0, 0)
        + _wire(-10, 5.08, -2.54, 5.08)
        + _symbol_instance("power:+5V", "#PWR01", -10, 5.08, angle=90)
        + _symbol_instance("Test:MCU", "U2", 0, 0, uuid="00000000-0000-0000-0000-000000000009").replace(
            '"Test:MCU"', '"Device:C"').replace("(at 0 0 0) (unit", "(at 50 50 90) (unit")
        + _symbol_instance("power:+5V", "#PWR02", 50, 47.46)
        + _symbol_instance("power:GND", "#PWR03", 50, 52.54)
    )
    rules = [f["rule"] for f in check_schematic(_write_mcu_sch(tmp_path, body))]
    assert "missing_decoupling_cap" not in rules


def test_overlapping_parts_are_reported(tmp_path):
    # two resistors drawn on top of each other (not connected) -- the
    # "parts placed on the same spot" problem
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        + _symbol_instance("Test:R", "R2", 100.5, 100.3, uuid="00000000-0000-0000-0000-000000000002")
        + _wire(90, 100, 97.46, 100) + _wire(102.54, 100, 110, 100)
        + _wire(90, 120, 97.96, 100.3) + _wire(103.04, 100.3, 110, 120)
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    assert any(f["rule"] == "overlapping_symbols" and "R1" in f["message"] and "R2" in f["message"]
               for f in findings)


def test_power_symbol_pointing_into_a_part_is_reported(tmp_path):
    # GND at R1's left pin rotated 90: its graphic points RIGHT, into R1
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        + _symbol_instance("power:GND", "#PWR01", 97.46, 100, angle=90)
        + _wire(102.54, 100, 105, 100)
        + '(label "OUT" (at 105 100 0) (effects (font (size 1.27 1.27))))'
    )
    rules = [f["rule"] for f in check_schematic(_write_sch(tmp_path, body))]
    assert "overlapping_symbols" in rules


def test_label_text_on_top_of_a_part_is_reported(tmp_path):
    body = (
        _symbol_instance("Test:R", "R1", 100, 100)
        + _symbol_instance("power:GND", "#PWR01", 97.46, 100, angle=270)
        + _wire(102.54, 100, 105, 100)
        # label anchored left of R1 with its text running right, across R1's body
        + _wire(105, 100, 105, 110)
        + '(label "A_LONG_NET_NAME" (at 98 100 0) (effects (font (size 1.27 1.27))))'
    )
    findings = check_schematic(_write_sch(tmp_path, body))
    assert any(f["rule"] == "overlapping_symbols" and "A_LONG_NET_NAME" in f["message"] for f in findings)
