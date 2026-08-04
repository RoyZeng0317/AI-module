"""Pipeline test for kicad_dataset_convert.py.

No real KiCad install is available on this dev machine, so `kicad-cli sch
export image` is swapped for a fake render function that just writes a
blank PNG of a known pixel size -- this still exercises the real S-
expression parsing, class mapping, and pixel-space bounding-box math
end-to-end; it does not (and cannot, without KiCad) verify that the real
kicad-cli export lines up with these coordinates. See the Limitations
section in kicad_dataset_convert.py's docstring.
"""

from pathlib import Path

import pytest
from PIL import Image

from kicad_dataset_convert import (
    DEFAULT_CLASS_MAP,
    build_dataset,
    classify_lib_id,
    convert_one,
    get_junctions,
    get_page_size_mm,
    get_symbol_instances,
    load_class_map,
    parse_sexpr,
)

FAKE_IMG_SIZE = (800, 600)  # arbitrary -- scale is derived from this, not from --resolution
CLASSES = ["resistor", "capacitor", "inductor", "diode", "ic",
           "transistor", "battery", "switch", "led", "wire_junction"]

SCH_TEMPLATE = """(kicad_sch (version 20231120) (generator eeschema)
  (paper "A4")
  (lib_symbols
    (symbol "Device:R" (in_bom yes) (on_board yes)
      (symbol "Device:R_0_1" (rectangle (start -1.016 -2.54) (end 1.016 2.54)))
      (symbol "Device:R_1_1" (pin passive line (at 0 3.81 270) (length 1.27) (name "~") (number "1")))
    )
  )
  (symbol (lib_id "Device:R") (at 100 50 0) (unit 1) (uuid "aaaa"))
  (symbol (lib_id "Device:C") (at 150 80 90) (unit 1) (uuid "bbbb"))
  (symbol (lib_id "Foo:Bar") (at 50 50 0) (unit 1) (uuid "cccc"))
  (junction (at 120 60) (diameter 0) (color 0 0 0 0))
)
"""


def _fake_render(sch_path: Path, out_png: Path, resolution_dpi, kicad_cli, exclude_drawing_sheet):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", FAKE_IMG_SIZE, "white").save(out_png)


def _write_sch(path: Path, text: str = SCH_TEMPLATE):
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_sexpr_roundtrips_structure():
    root = parse_sexpr(SCH_TEMPLATE)
    assert root[0] == "kicad_sch"
    assert get_page_size_mm(root) == (297.0, 210.0)


def test_get_symbol_instances_excludes_lib_symbols_definitions():
    root = parse_sexpr(SCH_TEMPLATE)
    instances = get_symbol_instances(root)
    lib_ids = sorted(inst["lib_id"] for inst in instances)
    # exactly the 3 placed instances -- NOT the "Device:R_0_1"/"Device:R_1_1"
    # sub-unit definitions living inside lib_symbols
    assert lib_ids == ["Device:C", "Device:R", "Foo:Bar"]


def test_get_junctions():
    root = parse_sexpr(SCH_TEMPLATE)
    assert get_junctions(root) == [(120.0, 60.0)]


def test_classify_lib_id_prefix_word_boundary():
    class_map = load_class_map()
    assert classify_lib_id("Device:R", class_map) == "resistor"
    assert classify_lib_id("Device:R_Small", class_map) == "resistor"
    # must NOT match "Device:R" prefix just because it starts with "R"
    assert classify_lib_id("Device:Relay", class_map) != "resistor"
    assert classify_lib_id("Totally:Unknown", class_map) is None


def test_load_class_map_merges_overrides(tmp_path):
    override_path = tmp_path / "class_map.json"
    override_path.write_text(
        '{"ic": {"prefixes": ["Amplifier_Operational:LM358"], "box_mm": [12, 8]}}',
        encoding="utf-8",
    )
    class_map = load_class_map(override_path)
    assert class_map["ic"]["prefixes"] == ["Amplifier_Operational:LM358"]
    # untouched classes still present from the default map
    assert class_map["resistor"] == DEFAULT_CLASS_MAP["resistor"]


def test_convert_one_writes_yolo_labels(tmp_path):
    sch_path = _write_sch(tmp_path / "one.kicad_sch")
    image_out = tmp_path / "images" / "one.png"
    label_out = tmp_path / "labels" / "one.txt"
    class_map = load_class_map()

    n_boxes, unmapped = convert_one(sch_path, image_out, label_out, CLASSES, class_map,
                                     resolution_dpi=100, kicad_cli="kicad-cli",
                                     exclude_drawing_sheet=False, render_fn=_fake_render)

    assert image_out.exists()
    assert unmapped == {"Foo:Bar"}  # unmapped lib_id skipped, not guessed at
    lines = label_out.read_text(encoding="utf-8").strip().splitlines()
    assert n_boxes == len(lines) == 3  # resistor + capacitor + 1 junction, Foo:Bar skipped

    img_w, img_h = FAKE_IMG_SIZE
    page_w_mm, page_h_mm = 297.0, 210.0
    parsed = [line.split() for line in lines]
    class_ids = {int(p[0]) for p in parsed}
    assert class_ids == {CLASSES.index("resistor"), CLASSES.index("capacitor"),
                          CLASSES.index("wire_junction")}

    resistor_line = next(p for p in parsed if int(p[0]) == CLASSES.index("resistor"))
    cx, cy, w, h = (float(v) for v in resistor_line[1:])
    expected_cx = 100 * (img_w / page_w_mm) / img_w
    expected_cy = 50 * (img_h / page_h_mm) / img_h
    assert cx == pytest.approx(expected_cx, abs=1e-6) and cy == pytest.approx(expected_cy, abs=1e-6)
    assert 0 < w <= 1 and 0 < h <= 1

    # capacitor is rotated 90 degrees -> its default box_mm [w, h] should swap
    cap_line = next(p for p in parsed if int(p[0]) == CLASSES.index("capacitor"))
    cap_w, cap_h = float(cap_line[3]), float(cap_line[4])
    box_w_mm, box_h_mm = DEFAULT_CLASS_MAP["capacitor"]["box_mm"]
    assert cap_w == pytest.approx(box_h_mm * (img_w / page_w_mm) / img_w, abs=1e-6)
    assert cap_h == pytest.approx(box_w_mm * (img_h / page_h_mm) / img_h, abs=1e-6)


def test_build_dataset_splits_train_val(tmp_path):
    sch_dir = tmp_path / "sch"
    sch_dir.mkdir()
    _write_sch(sch_dir / "a.kicad_sch")
    _write_sch(sch_dir / "b.kicad_sch")
    out_dir = tmp_path / "dataset"
    class_map = load_class_map()

    total_boxes, unmapped = build_dataset(
        sorted(sch_dir.glob("*.kicad_sch")), out_dir, CLASSES, class_map,
        resolution_dpi=100, kicad_cli="kicad-cli", exclude_drawing_sheet=False,
        val_ratio=0.5, seed=0, render_fn=_fake_render,
    )

    assert total_boxes == 6  # 3 boxes per file x 2 files
    assert unmapped == {"Foo:Bar"}
    train_imgs = list((out_dir / "images" / "train").glob("*.png"))
    val_imgs = list((out_dir / "images" / "val").glob("*.png"))
    assert len(train_imgs) == 1 and len(val_imgs) == 1
    for img_path in train_imgs + val_imgs:
        label_path = out_dir / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        assert label_path.exists()
        assert len(label_path.read_text(encoding="utf-8").strip().splitlines()) == 3
