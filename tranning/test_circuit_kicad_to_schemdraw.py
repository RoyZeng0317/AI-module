"""Tests for circuit_kicad_to_schemdraw.py.

Same synthetic-fixture pattern as test_circuit_rule_check.py: small
hand-written .kicad_sch text, not a real file, so every assertion is
pinned to a known layout.
"""

from pathlib import Path

from circuit_kicad_to_schemdraw import kicad_to_drawing, render_file
from kicad_dataset_convert import parse_sexpr

_LIB_R = """
    (lib_symbols
        (symbol "Device:R"
            (symbol "R_1_1"
                (pin passive line (at -2.54 0 0) (length 2.54)
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 2.54 0 0) (length 2.54)
                    (number "2" (effects (font (size 1.27 1.27)))))
            )
        )
        (symbol "MCU:Fake4Pin"
            (symbol "Fake4Pin_1_1"
                (pin passive line (at -7.62 5.08 0) (length 2.54)
                    (number "1" (effects (font (size 1.27 1.27)))))
                (pin passive line (at -7.62 -5.08 0) (length 2.54)
                    (number "2" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 7.62 5.08 0) (length 2.54)
                    (number "3" (effects (font (size 1.27 1.27)))))
                (pin passive line (at 7.62 -5.08 0) (length 2.54)
                    (number "4" (effects (font (size 1.27 1.27)))))
            )
        )
    )
"""


def _symbol_instance(lib_id, ref, x, y, angle=0):
    return f"""
    (symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) (uuid "u1")
        (property "Reference" "{ref}" (at {x} {y} {angle}) (effects (font (size 1.27 1.27))))
    )
    """


def _wire(x1, y1, x2, y2):
    return f'(wire (pts (xy {x1} {y1}) (xy {x2} {y2})) (stroke (width 0) (type default)))'


def _parse(body: str):
    return parse_sexpr(f'(kicad_sch (version 20231120) (generator test) (paper "A4") {_LIB_R} {body})')


def test_mapped_and_unmapped_lib_ids():
    root = _parse(
        _symbol_instance("Device:R", "R1", 10, 10)
        + _symbol_instance("Some:UnknownPart", "U1", 30, 10)
    )
    drawing, unmapped, imprecise = kicad_to_drawing(root)

    assert len(drawing.elements) == 2  # both drawn: one pin-accurate, one placeholder
    assert unmapped == {"Some:UnknownPart"}
    assert imprecise == {"U1"}  # no lib_symbols entry for it -> can't resolve pins


def test_two_pin_part_is_anchored_at_its_real_pins_not_its_origin():
    # Device:R's pins are at local (-2.54, 0) and (2.54, 0); placed at (10, 10)
    # with no rotation, so world pins are (7.46, 10) and (12.54, 10) -- neither
    # equals the instance origin (10, 10) itself, unlike the old origin-anchored
    # placement this replaces.
    root = _parse(_symbol_instance("Device:R", "R1", 10, 10))
    drawing, unmapped, imprecise = kicad_to_drawing(root)

    assert unmapped == set()
    assert imprecise == set()  # 2 real pins resolved -> pin-accurate, not a fallback
    resistor = drawing.elements[0]
    # segments[0] is the element's own leaddefinition in absolute drawing
    # coordinates once placed; just confirm it is not a single point collapsed
    # onto the origin (i.e. .to() actually stretched it between two pins).
    assert resistor.absanchors["start"] != resistor.absanchors["end"]


def test_multi_pin_part_gets_a_body_box_with_lead_stubs_to_real_pins():
    root = _parse(_symbol_instance("MCU:Fake4Pin", "U1", 50, 50))
    drawing, unmapped, imprecise = kicad_to_drawing(root)

    assert unmapped == {"MCU:Fake4Pin"}  # not in DEFAULT_CLASS_MAP -> no specific 2-terminal shape
    assert imprecise == {"U1"}           # box/stub approximation, not a real 2-terminal placement
    # 4 lead stubs + 4 box edges + 1 center label = 9 elements
    assert len(drawing.elements) == 9


def test_real_wires_and_junctions_are_drawn():
    root = _parse(
        _symbol_instance("Device:R", "R1", 10, 10)
        + _wire(10, 10, 20, 10)
        + '(junction (at 20 10))'
    )
    drawing, _, _ = kicad_to_drawing(root)

    assert len(drawing.elements) == 3  # resistor + wire line + junction dot


def test_render_file_writes_a_file(tmp_path: Path):
    sch_path = tmp_path / "test.kicad_sch"
    sch_path.write_text(
        f'(kicad_sch (version 20231120) (generator test) (paper "A4") {_LIB_R} '
        f'{_symbol_instance("Device:R", "R1", 10, 10)})',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.svg"

    unmapped, imprecise = render_file(sch_path, out_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    assert unmapped == set()
    assert imprecise == set()
