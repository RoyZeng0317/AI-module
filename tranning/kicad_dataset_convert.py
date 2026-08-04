r"""KiCad schematic (.kicad_sch) -> YOLO training dataset converter.

circuit_diagram_train.py needs images/{train,val} + labels/{train,val}
(YOLO-format bounding boxes). A .kicad_sch file is not an image and has no
bounding boxes -- it is a text (S-expression) description of a schematic:
every component's type (lib_id, e.g. "Device:R") and position (at x y
angle) is already recorded exactly. This script turns that into a labeled
dataset automatically instead of hand-drawing boxes:

    1. Render each .kicad_sch to a PNG: `kicad-cli sch export pdf` (vector;
       kicad-cli has no raster/DPI export for schematics) then rasterize
       that PDF ourselves with PyMuPDF at --resolution DPI.
    2. Parse the same .kicad_sch file's S-expression to find every placed
       symbol (lib_id + position + rotation) and every wire junction.
    3. Map each lib_id to one of the target classes via a prefix table
       (DEFAULT_CLASS_MAP, extendable with --class-map).
    4. Convert each component's schematic-space position into a pixel
       bounding box (using a per-class approximate size in mm -- see
       Limitations) and write it out as a YOLO label line.

Requires KiCad installed (for `kicad-cli`, not on PATH by default on
Windows -- pass its full path via --kicad-cli, e.g.
"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe") and `pip install
pymupdf` (rasterizes kicad-cli's PDF export; no separate Poppler/
Ghostscript install needed).

Usage:
    # label every .kicad_sch in a folder, 80/20 train/val split
    python kicad_dataset_convert.py --sch-dir path/to/schematics --out-dir path/to/dataset --kicad-cli "C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

    # single file first, to sanity-check alignment before batch-labeling
    python kicad_dataset_convert.py --sch path/to/one.kicad_sch --out-dir path/to/dataset --kicad-cli "C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

    # then feed the resulting dataset straight into circuit_diagram_train.py
    python circuit_diagram_train.py --dataset-dir path/to/dataset --classes \
        resistor capacitor inductor diode ic transistor battery switch led wire_junction

Extending the class map (unmapped lib_ids are skipped and reported, not
guessed at) -- pass --class-map pointing at a JSON file like:
    {
      "ic": {"prefixes": ["Amplifier_Operational:LM358", "74xx:74HC00"], "box_mm": [12, 8]}
    }
Keys merge into DEFAULT_CLASS_MAP (new class, or override an existing one's
prefixes/box_mm).

Verified for real (2026-07-26, KiCad 10.0.4, `kicad-cli` at
"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"): ran this script's
actual CLI end to end against a real schematic (KiCad's own bundled demo,
share/kicad/demos/interf_u/interf_u.kicad_sch, 52 placed symbols + 39
junctions) and then, separately, checked every one of the 52 generated
YOLO boxes against the generated PNG -- cropped each box out of the image
and confirmed it contains drawn ink (52/52), vs. control points away from
any symbol being blank. That confirms the "paper" node mm size matches
the real exported page size and schematic (x, y) in mm maps linearly (no
offset/flip) onto the rasterized page. Two real bugs were caught and
fixed by this real test, not left as guesses: (1) an earlier version
assumed a `kicad-cli sch export image` subcommand; KiCad 10 has no such
thing (only pdf/svg/dxf/... vector exports), fixed by exporting PDF and
rasterizing it with PyMuPDF instead; (2) `subprocess.run(..., text=True)`
decoded kicad-cli's UTF-8 stdout/stderr using the OS's default codepage
(cp950 on this Traditional Chinese Windows install), which crashed a
reader thread on kicad-cli's own non-ASCII progress messages -- fixed by
passing encoding="utf-8" explicitly.

Limitations (read before trusting this at scale):
    - Bounding boxes use a fixed approximate size per class (DEFAULT_CLASS_MAP
      box_mm), not the symbol's real drawn extent -- computing the exact
      extent would mean parsing every graphical primitive (rectangle/arc/
      polyline/pin) out of the schematic's embedded lib_symbols block and
      applying rotation+mirror transforms. The fixed-box approach is
      simpler and good enough to bootstrap a dataset, but you should open a
      few generated images with their labels overlaid (e.g. via the
      Ultralytics dataset viewer) before training on hundreds of files, and
      tune --class-map box_mm per class if boxes look off.
    - Real projects (e.g. the interf_u demo used above) often use
      project-local library copies like "interf_u:R" instead of the
      generic "Device:R" -- DEFAULT_CLASS_MAP only matches the generic
      "Device:*"/"Switch:*" names. Unmapped lib_ids are skipped and printed
      at the end of a run so you can add your project's actual library
      names via --class-map instead of them silently going unlabeled.
    - Only handles the current KiCad 6+ S-expression format (.kicad_sch).
      Legacy KiCad 5 and earlier (.sch, non-S-expression) is not supported.
    - "ic" has no default prefixes (IC libraries vary too much to guess) --
      supply your own via --class-map or those symbols are skipped.
    - Hierarchical multi-sheet designs: only tested against a flat,
      single-sheet schematic. `export_schematic_image` always reads page 0
      of the exported PDF; if `kicad-cli sch export pdf` on a root sheet
      that has child sheets emits one page per sheet (untested here),
      pages beyond the first would be silently ignored. Each .kicad_sch
      file in a hierarchy is itself a valid standalone document, so the
      safe path is running this script once per sheet file rather than
      pointing --sch-dir at a hierarchy's root only.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import fitz  # PyMuPDF -- pip install pymupdf; rasterizes kicad-cli's PDF export
from PIL import Image

# (class_name, prefixes matching "Library:Symbol" lib_ids, default box size in mm)
# A lib_id matches a prefix if it equals the prefix exactly or starts with
# "<prefix>_" (word-boundary match) -- plain str.startswith() would let
# "Device:Relay" wrongly match the "Device:R" resistor prefix.
DEFAULT_CLASS_MAP = {
    "resistor": {"prefixes": ["Device:R"], "box_mm": [10.0, 3.0]},
    "capacitor": {"prefixes": ["Device:C", "Device:CP"], "box_mm": [6.0, 3.0]},
    "inductor": {"prefixes": ["Device:L"], "box_mm": [8.0, 4.0]},
    "diode": {"prefixes": ["Device:D"], "box_mm": [6.0, 3.0]},
    "led": {"prefixes": ["Device:LED"], "box_mm": [6.0, 3.0]},
    "transistor": {"prefixes": ["Device:Q"], "box_mm": [8.0, 8.0]},
    "battery": {"prefixes": ["Device:Battery"], "box_mm": [8.0, 6.0]},
    "switch": {"prefixes": ["Switch:SW"], "box_mm": [8.0, 6.0]},
    "ic": {"prefixes": [], "box_mm": [12.0, 10.0]},
}
JUNCTION_BOX_MM = (1.5, 1.5)

# ISO paper sizes in mm, landscape (width, height) -- KiCad's schematic
# default orientation. A "(paper ... portrait)" node swaps these.
PAPER_SIZES_MM = {
    "A5": (210.0, 148.0), "A4": (297.0, 210.0), "A3": (420.0, 297.0),
    "A2": (594.0, 420.0), "A1": (841.0, 594.0), "A0": (1189.0, 841.0),
    "A": (279.4, 215.9), "B": (431.8, 279.4), "C": (558.8, 431.8),
    "D": (863.6, 558.8), "E": (1117.6, 863.6),
}


# ---------------------------------------------------------------------------
# Minimal S-expression parser (KiCad 6+ file formats are plain S-expressions)
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c in "()":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            tokens.append(("str", "".join(buf)))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in " \t\r\n()":
                j += 1
            tokens.append(("atom", text[i:j]))
            i = j
    return tokens


def parse_sexpr(text: str):
    """Parse a full S-expression document into nested Python lists.

    Strings become plain str, atoms that look like numbers become float,
    everything else stays str. Top-level is the single outermost list,
    e.g. ["kicad_sch", ["version", 20231120], ["paper", "A4"], ...].
    """
    tokens = _tokenize(text)
    pos = 0

    def parse_one():
        nonlocal pos
        tok = tokens[pos]
        if tok != "(":
            raise ValueError(f"expected '(' at token {pos}, got {tok!r}")
        pos += 1
        node = []
        while tokens[pos] != ")":
            if tokens[pos] == "(":
                node.append(parse_one())
            else:
                kind, val = tokens[pos]
                if kind == "str":
                    node.append(val)
                else:
                    try:
                        node.append(float(val) if ("." in val or "e" in val.lower()) else int(val))
                    except ValueError:
                        node.append(val)
                pos += 1
        pos += 1  # consume ")"
        return node

    root = parse_one()
    if pos != len(tokens):
        raise ValueError("trailing tokens after top-level expression")
    return root


def find_all(node, tag):
    """Recursively find every sub-list whose first element == tag."""
    out = []
    if isinstance(node, list):
        if node and node[0] == tag:
            out.append(node)
        for child in node:
            out.extend(find_all(child, tag))
    return out


def find_child(node, tag):
    for child in node:
        if isinstance(child, list) and child and child[0] == tag:
            return child
    return None


# ---------------------------------------------------------------------------
# Schematic -> (page size, symbol instances, junctions)
# ---------------------------------------------------------------------------

def get_page_size_mm(root):
    paper = find_child(root, "paper")
    if paper is None:
        return PAPER_SIZES_MM["A4"]
    name = paper[1]
    if name == "User" and len(paper) >= 4:
        w, h = float(paper[2]), float(paper[3])
    else:
        w, h = PAPER_SIZES_MM.get(name, PAPER_SIZES_MM["A4"])
    if "portrait" in paper[1:]:
        w, h = h, w
    return w, h


def get_symbol_instances(root):
    """Placed symbols only -- library *definitions* under `lib_symbols` are
    `(symbol "Device:R" ...)` (name as a bare string), while placed
    instances are `(symbol (lib_id "Device:R") (at x y angle) ...)` (a
    nested lib_id node). Matching on that nested node excludes definitions.
    """
    instances = []
    for node in find_all(root, "symbol"):
        lib_id_node = find_child(node, "lib_id")
        at_node = find_child(node, "at")
        if lib_id_node is None or at_node is None:
            continue
        x, y = float(at_node[1]), float(at_node[2])
        angle = float(at_node[3]) if len(at_node) > 3 else 0.0
        instances.append({"lib_id": lib_id_node[1], "x": x, "y": y, "angle": angle})
    return instances


def get_junctions(root):
    points = []
    for node in find_all(root, "junction"):
        at_node = find_child(node, "at")
        if at_node is not None:
            points.append((float(at_node[1]), float(at_node[2])))
    return points


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

def load_class_map(class_map_path=None):
    class_map = {k: {"prefixes": list(v["prefixes"]), "box_mm": list(v["box_mm"])}
                 for k, v in DEFAULT_CLASS_MAP.items()}
    if class_map_path:
        overrides = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        for name, spec in overrides.items():
            class_map[name] = {"prefixes": list(spec["prefixes"]), "box_mm": list(spec["box_mm"])}
    return class_map


def classify_lib_id(lib_id: str, class_map: dict):
    for class_name, spec in class_map.items():
        for prefix in spec["prefixes"]:
            if lib_id == prefix or lib_id.startswith(prefix + "_"):
                return class_name
    return None


# ---------------------------------------------------------------------------
# Rendering (kicad-cli) -- isolated in its own function so tests can swap it
# for a synthetic image instead of needing a real KiCad install.
# ---------------------------------------------------------------------------

def export_schematic_image(sch_path: Path, out_png: Path, resolution_dpi: int,
                            kicad_cli: str, exclude_drawing_sheet: bool):
    """kicad-cli has no schematic-to-raster export -- only vector formats
    (pdf/svg/dxf/...). Export PDF, then rasterize page 0 ourselves with
    PyMuPDF at resolution_dpi. See the module docstring's "Verified for
    real" note for how this was checked against an actual KiCad install.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_png.with_suffix(".pdf")
    cmd = [kicad_cli, "sch", "export", "pdf", "-o", str(pdf_path), str(sch_path)]
    if exclude_drawing_sheet:
        cmd.append("-e")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise RuntimeError(
            f"'{kicad_cli}' not found -- install KiCad or pass --kicad-cli <path to kicad-cli>"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"kicad-cli failed on {sch_path}:\n{e.stderr}\n"
            "run `kicad-cli sch export pdf --help` to check the flags for your KiCad version"
        )

    doc = fitz.open(pdf_path)
    try:
        doc[0].get_pixmap(dpi=resolution_dpi).save(str(out_png))
    finally:
        doc.close()
    pdf_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# One schematic -> one (image, YOLO label) pair
# ---------------------------------------------------------------------------

def convert_one(sch_path: Path, image_out: Path, label_out: Path, classes: list, class_map: dict,
                 resolution_dpi: int, kicad_cli: str, exclude_drawing_sheet: bool,
                 render_fn=export_schematic_image):
    root = parse_sexpr(sch_path.read_text(encoding="utf-8"))
    page_w_mm, page_h_mm = get_page_size_mm(root)

    render_fn(sch_path, image_out, resolution_dpi, kicad_cli, exclude_drawing_sheet)
    with Image.open(image_out) as img:
        img_w, img_h = img.size
    px_per_mm_x, px_per_mm_y = img_w / page_w_mm, img_h / page_h_mm

    lines = []
    unmapped = set()
    for inst in get_symbol_instances(root):
        class_name = classify_lib_id(inst["lib_id"], class_map)
        if class_name is None:
            unmapped.add(inst["lib_id"])
            continue
        box_w_mm, box_h_mm = class_map[class_name]["box_mm"]
        if round(inst["angle"]) % 180 == 90:
            box_w_mm, box_h_mm = box_h_mm, box_w_mm
        lines.append(_yolo_line(classes.index(class_name), inst["x"], inst["y"],
                                 box_w_mm, box_h_mm, px_per_mm_x, px_per_mm_y, img_w, img_h))

    if "wire_junction" in classes:
        jw_mm, jh_mm = JUNCTION_BOX_MM
        for x, y in get_junctions(root):
            lines.append(_yolo_line(classes.index("wire_junction"), x, y, jw_mm, jh_mm,
                                     px_per_mm_x, px_per_mm_y, img_w, img_h))

    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), unmapped


def _yolo_line(class_id, x_mm, y_mm, box_w_mm, box_h_mm, px_per_mm_x, px_per_mm_y, img_w, img_h):
    cx_px, cy_px = x_mm * px_per_mm_x, y_mm * px_per_mm_y
    w_px, h_px = box_w_mm * px_per_mm_x, box_h_mm * px_per_mm_y
    cx, cy = min(max(cx_px / img_w, 0.0), 1.0), min(max(cy_px / img_h, 0.0), 1.0)
    w, h = min(w_px / img_w, 1.0), min(h_px / img_h, 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# Dataset build: many .kicad_sch -> images/{train,val} + labels/{train,val}
# ---------------------------------------------------------------------------

def build_dataset(sch_files: list, out_dir: Path, classes: list, class_map: dict,
                   resolution_dpi: int, kicad_cli: str, exclude_drawing_sheet: bool,
                   val_ratio: float, seed: int, render_fn=export_schematic_image):
    sch_files = list(sch_files)
    rng = random.Random(seed)
    rng.shuffle(sch_files)
    n_val = int(len(sch_files) * val_ratio) if len(sch_files) > 1 else 0
    splits = {"val": sch_files[:n_val], "train": sch_files[n_val:]}
    if len(sch_files) > 1 and n_val == 0:
        print("[warning: too few files for --val-ratio to place any in val -- everything went to train]")
    elif len(sch_files) <= 1:
        print("[warning: only one schematic given -- no validation split is possible]")

    total_boxes, all_unmapped = 0, set()
    for split, files in splits.items():
        for sch_path in files:
            image_out = out_dir / "images" / split / f"{sch_path.stem}.png"
            label_out = out_dir / "labels" / split / f"{sch_path.stem}.txt"
            n_boxes, unmapped = convert_one(sch_path, image_out, label_out, classes, class_map,
                                             resolution_dpi, kicad_cli, exclude_drawing_sheet,
                                             render_fn=render_fn)
            total_boxes += n_boxes
            all_unmapped |= unmapped

    print(f"labeled {len(sch_files)} schematic(s) -> {total_boxes} boxes "
          f"({len(splits['train'])} train / {len(splits['val'])} val)")
    if all_unmapped:
        print(f"[unmapped lib_ids, skipped -- extend --class-map to include them: "
              f"{', '.join(sorted(all_unmapped))}]")
    return total_boxes, all_unmapped


def main():
    parser = argparse.ArgumentParser(description="Convert KiCad .kicad_sch schematics into a YOLO dataset")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sch-dir", type=Path, help="folder of .kicad_sch files")
    src.add_argument("--sch", type=Path, help="a single .kicad_sch file (no val split)")
    parser.add_argument("--out-dir", type=Path, required=True, help="dataset root to write images/labels into")
    parser.add_argument("--classes", nargs="+",
                        default=["resistor", "capacitor", "inductor", "diode", "ic",
                                 "transistor", "battery", "switch", "led", "wire_junction"],
                        help="class list, index = label id (must match circuit_diagram_train.py --classes)")
    parser.add_argument("--class-map", type=Path, help="JSON file merging into/overriding DEFAULT_CLASS_MAP")
    parser.add_argument("--resolution", type=int, default=300, help="export DPI")
    parser.add_argument("--kicad-cli", default="kicad-cli", help="path to the kicad-cli executable")
    parser.add_argument("--exclude-drawing-sheet", action="store_true",
                        help="drop the page border/title block from the exported image")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_map = load_class_map(args.class_map)
    sch_files = sorted(args.sch_dir.glob("*.kicad_sch")) if args.sch_dir else [args.sch]
    if not sch_files:
        parser.error(f"no .kicad_sch files found in {args.sch_dir}")

    build_dataset(sch_files, args.out_dir, args.classes, class_map, args.resolution,
                  args.kicad_cli, args.exclude_drawing_sheet, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
