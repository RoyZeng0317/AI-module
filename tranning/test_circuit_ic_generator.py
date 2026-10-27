"""Tests for circuit_ic_generator.py.

The real point of this generator is that its output is a genuinely valid
.kicad_sch this project's OWN existing tools already understand -- so these
tests run the generated text through the real parser, the real ERC checker,
and the real renderer, instead of re-implementing separate assertions about
the generator's internal Python structures.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from circuit_ic_generator import build_minimal_system
from circuit_kicad_to_schemdraw import kicad_to_drawing
from circuit_rule_check import check_schematic
from kicad_dataset_convert import parse_sexpr


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "generated.kicad_sch"
    path.write_text(text, encoding="utf-8")
    return path


def test_output_parses_as_valid_sexpr():
    text = build_minimal_system("atmega328p")
    root = parse_sexpr(text)
    assert root[0] == "kicad_sch"


def test_generated_circuit_has_no_erc_findings(tmp_path):
    findings = check_schematic(_write(tmp_path, build_minimal_system("atmega328p")))
    # a correctly-built minimal system should trip none of the checks,
    # including the new MCU-aware ones this same task added
    assert findings == [], findings


def test_repeated_calls_do_not_leak_reference_counters():
    # _RefCounter is per-call state now (not module-global) -- two calls in
    # the same process must produce identical reference numbering, not
    # C7/C8/... continuing to climb on the second call.
    first = build_minimal_system("atmega328p")
    second = build_minimal_system("atmega328p")
    assert first == second


def test_every_ic_pin_is_placed_with_a_real_tip_coordinate():
    root = parse_sexpr(build_minimal_system("atmega328p"))
    from circuit_rule_check import get_lib_pin_table
    table = get_lib_pin_table(root)
    ic_pins = table["MCU_Microchip_ATmega:ATmega328P-PU"][1]
    assert len(ic_pins) == 28
    numbers = sorted(int(p["number"]) for p in ic_pins)
    assert numbers == list(range(1, 29))


def test_renders_without_error_and_ic_is_not_a_zero_pin_placeholder(tmp_path):
    root = parse_sexpr(build_minimal_system("atmega328p"))
    drawing, unmapped, imprecise = kicad_to_drawing(root)
    # the IC itself is an unmapped >2-pin part (drawn as a labeled body box,
    # see circuit_kicad_to_schemdraw.py's own docstring) -- that's expected,
    # not a bug, so it must show up in `imprecise`, not silently look exact
    assert "U1" in imprecise
    assert "MCU_Microchip_ATmega:ATmega328P-PU" in unmapped
    out = tmp_path / "out.svg"
    drawing.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def _strip_balanced_instances(text: str, marker: str) -> str:
    """Remove every top-level `(symbol (lib_id "<marker>") ...)` instance
    block, tracking paren depth char-by-char instead of using regex --
    these blocks contain nested parens a regex can't reliably balance."""
    out, i = [], 0
    while True:
        idx = text.find(marker, i)
        if idx == -1:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:idx])
        depth, j = 0, idx
        while True:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        i = j


def test_removing_a_decoupling_cap_is_caught_by_the_checker(tmp_path):
    # sanity check that the "clean" test above isn't vacuously true: drop
    # every Device:C instance and the new missing_decoupling_cap /
    # missing_crystal_load_caps checks must fire.
    text = build_minimal_system("atmega328p")
    stripped = _strip_balanced_instances(text, '(symbol (lib_id "Device:C")')
    findings = check_schematic(_write(tmp_path, stripped))
    rules = {f["rule"] for f in findings}
    assert "missing_decoupling_cap" in rules
    assert "missing_crystal_load_caps" in rules


# ---------------------------------------------------------------------------
# 2026-10-27: larger parts (ESP32 / Arduino Mega 2560 / Raspberry Pi 5
# header) and the no-overlap layout. The user reported parts drawn on top of
# each other in the schematic for complex boards; these pin that down.
# ---------------------------------------------------------------------------

ALL_PARTS = ["atmega328p", "mega2560", "esp32", "pi5"]


@pytest.mark.parametrize("ic", ALL_PARTS)
def test_every_supported_part_generates_with_no_erc_or_overlap_findings(tmp_path, ic):
    findings = check_schematic(_write(tmp_path, build_minimal_system(ic)))
    # includes overlapping_symbols: no part / power symbol / label sits on another
    assert findings == [], findings


@pytest.mark.parametrize("ic", ALL_PARTS)
def test_output_is_deterministic_with_unique_references(ic):
    text = build_minimal_system(ic)
    assert text == build_minimal_system(ic)
    placed = text[text.index("(symbol (lib_id"):]
    refs = re.findall(r'\(property "Reference" "([^"]+)"', placed)
    assert len(refs) == len(set(refs)), "duplicate reference designators"


@pytest.mark.parametrize("ic, pin_count", [("mega2560", 100), ("esp32", 38), ("pi5", 40)])
def test_large_parts_have_every_pin(ic, pin_count):
    from circuit_rule_check import get_lib_pin_table
    from ic_library import resolve_ic
    table = get_lib_pin_table(parse_sexpr(build_minimal_system(ic)))
    numbers = sorted(int(p["number"]) for p in table[resolve_ic(ic)][1])
    assert numbers == list(range(1, pin_count + 1))


@pytest.mark.parametrize("ic", ALL_PARTS)
def test_drawing_is_inside_the_paper(ic):
    text = build_minimal_system(ic)
    paper = re.search(r'\(paper "(\w+)"\)', text).group(1)
    width, height = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594)}[paper]
    body = text[text.index("(junction") if "(junction" in text else text.index("(wire"):]
    points = [(float(x), float(y)) for x, y in re.findall(r"\(at ([-\d.]+) ([-\d.]+)", body)]
    points = [p for p in points if p != (0.0, 0.0)]          # hidden library-default fields
    assert min(p[0] for p in points) > 10 and min(p[1] for p in points) > 10
    assert max(p[0] for p in points) < width - 10 and max(p[1] for p in points) < height - 10


def test_mega2560_gets_one_decoupling_cap_per_supply_pin(tmp_path):
    text = build_minimal_system("mega2560")
    # 4x VCC + AVCC + AREF -> 6 x 100nF, plus 2 x 22pF crystal load caps
    assert text.count('"Value" "100nF"') == 6
    assert text.count('"Value" "22pF"') == 2
    stripped = _strip_balanced_instances(text, '(symbol (lib_id "Device:C")')
    rules = {f["rule"] for f in check_schematic(_write(tmp_path, stripped))}
    assert {"missing_decoupling_cap", "missing_crystal_load_caps"} <= rules


def test_esp32_enable_pin_needs_its_pullup(tmp_path):
    # ESP32 calls its reset pin EN -- the checker must treat it as a reset pin
    text = build_minimal_system("esp32")
    assert "16MHz" not in text and '"Value" "1uF"' in text   # module has its own crystal; EN gets an RC delay
    stripped = _strip_balanced_instances(text, '(symbol (lib_id "Device:R")')
    rules = {f["rule"] for f in check_schematic(_write(tmp_path, stripped))}
    assert "missing_reset_pullup" in rules


def _kicad_netlist(tmp_path: Path, ic: str) -> dict:
    sch = tmp_path / f"{ic}.kicad_sch"
    sch.write_text(build_minimal_system(ic), encoding="utf-8")
    net = tmp_path / f"{ic}.net"
    subprocess.run(["kicad-cli", "sch", "export", "netlist", "--format", "kicadxml", "-o", str(net), str(sch)],
                   check=True, capture_output=True)
    nets = {}
    for name, body in re.findall(r'<net [^>]*name="([^"]*)"[^>]*>(.*?)</net>', net.read_text(), re.S):
        nets[name] = {f"{r}.{p}" for r, p in re.findall(r'ref="([^"]+)" pin="([^"]+)"', body)}
    return nets


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="KiCad not installed")
def test_real_kicad_connects_every_part_to_the_intended_pin(tmp_path):
    """Opens the generated file in REAL KiCad (kicad-cli netlist export).
    The pre-2026-10-27 generator passed this repo's own checker but in KiCad
    its +5V landed on the GND pin and the reset pull-up on PB0, because pin
    positions were computed without KiCad's library y-flip."""
    nets = _kicad_netlist(tmp_path, "atmega328p")
    assert {"U1.7", "U1.20", "C1.1", "C2.1", "R1.1"} <= nets["+5V"]
    assert {"U1.8", "U1.22"} <= nets["GND"] and "U1.7" not in nets["GND"]
    assert {"U1.1", "R1.2"} <= nets["/RESET"]
    assert {"U1.9", "Y1.1"} <= nets["/XTAL1"] and {"U1.10", "Y1.2"} <= nets["/XTAL2"]

    mega = _kicad_netlist(tmp_path, "mega2560")
    assert {"U1.10", "U1.31", "U1.61", "U1.80", "U1.100"} <= mega["+5V"]
    assert {"U1.11", "U1.32", "U1.62", "U1.81", "U1.99"} <= mega["GND"]
    assert {"U1.30", "R1.2"} <= mega["/RESET"]

    esp = _kicad_netlist(tmp_path, "esp32")
    assert {"U1.2", "C1.1", "R1.1"} <= esp["+3V3"] and {"U1.3", "R1.2", "C2.1"} <= esp["/EN"]
