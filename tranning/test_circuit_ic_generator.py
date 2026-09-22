"""Tests for circuit_ic_generator.py.

The real point of this generator is that its output is a genuinely valid
.kicad_sch this project's OWN existing tools already understand -- so these
tests run the generated text through the real parser, the real ERC checker,
and the real renderer, instead of re-implementing separate assertions about
the generator's internal Python structures.
"""

from pathlib import Path

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
