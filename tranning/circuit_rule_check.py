r"""KiCad schematic (.kicad_sch) electrical rule checker -- from-scratch ERC.

Treats a .kicad_sch file the way it actually is: a text (S-expression)
description of the circuit's structure, not a picture. Reuses the same
S-expression parser as kicad_dataset_convert.py (parse_sexpr / find_all /
find_child, already verified against a real KiCad export there) instead of
duplicating it, then does what circuit_diagram_train.py's YOLO detector
cannot: it reasons about *connectivity* -- which pins are actually wired to
which nets -- and reports real wiring mistakes.

Why build this instead of a vision model for "circuit correctness": a
picture of a schematic never states which pins are electrically joined --
that has to be inferred from pixel proximity, which is exactly the kind of
fuzzy judgement a small from-scratch YOLO detector (no real dataset yet,
see circuit_diagram_train.py) is worst at. The .kicad_sch file already
records every symbol's placement, every pin's library-defined offset, and
every wire's exact endpoints as numbers -- the "answer" is already in the
file, just not summarized. This is deterministic graph/geometry code, not a
trained model (same "decision-tree algorithm, not a neural net" category as
tranning/calculus_generator.py) -- it does not violate Rule 06 (no external
AI APIs) and needs no checkpoint or training data to be useful immediately.

How pin positions are computed (the part that actually needs to be
correct): a placed symbol instance only stores (lib_id, at x y angle,
unit, mirror) -- KiCad 6+ does NOT repeat each pin's position per
instance, only a bare (pin "<number>" (uuid ...)) reference. The real pin
offsets live once in the (lib_symbols ...) block, nested under a sub-symbol
named "<Name>_<unit>_<body_style>" (unit 0 = shared across all units, e.g.
a quad-gate IC's pins are unit-specific but its outline is unit 0). Each
pin's absolute position is: mirror the library-local (x, y) about the
requested axis, rotate by the instance's angle (0/90/180/270, always exact
since KiCad snaps symbol rotation to 90-degree steps), then translate by
the instance's (x, y). This was checked against a real file, not assumed:
in "touch LED.kicad_sch", a Device:R_US resistor is placed at
(142.24, 53.34, 90) with library pins at local (0, 3.81) and (0, -3.81);
this module's transform maps them to (138.43, 53.34) and (146.05, 53.34),
and the schematic's own wire endpoints at those exact two coordinates
confirm the mapping is right, not just plausible.

Why no point-on-segment geometry is needed for connectivity: real
KiCad-drawn schematics already break every wire into segments that meet
other wires/pins/junctions at an exact shared coordinate (confirmed by
inspection of multiple real files in this pass) rather than crossing
through the middle of a longer segment. So connectivity here is plain
union-find keyed by (x, y) rounded to 3 decimals (KiCad's own on-grid
precision) -- two items are on the same net iff they share a coordinate
key, transitively through wire endpoints. A pin sitting in the exact
middle of an unbroken wire (no shared endpoint) would be missed by this
model; this is a known, documented limitation, not a silent gap.

Checks implemented (see CircuitFinding below), each intentionally scoped to
something a human eyeballing hundreds of nets would actually miss:
  - floating_pin: a non-power-flag, non-no_connect pin whose exact
    coordinate touches nothing else at all (no wire endpoint, junction,
    other pin, or label) -- the single most common real "forgot to wire
    this" mistake.
  - no_connect_conflict: a pin explicitly flagged "no connect" that is
    nonetheless wired to something -- a stale/contradictory marker.
  - duplicate_reference: two placed (non-power) symbols sharing the same
    Reference designator (e.g. two "R1"), same category real KiCad ERC
    flags, because downstream netlist/BOM tools silently pick one.
  - net_name_conflict: a single electrical net touched by two or more
    *different* label/global_label/power-symbol names (e.g. GND wired
    directly to +5V by mistake) -- KiCad will happily let you draw this
    wire; it never asks "did you mean to short these".

Usage:
    python circuit_rule_check.py path/to/one.kicad_sch
    python circuit_rule_check.py path/to/project_dir --recursive
    python circuit_rule_check.py path/to/one.kicad_sch --json

Known limitations (read before trusting this at scale -- same honesty bar
as kicad_dataset_convert.py's docstring):
  - Hierarchical sheets (multi-file designs) are checked file-by-file; a
    net that only becomes complete by crossing a (sheet ...) boundary into
    a different .kicad_sch will show as floating here even though the
    full hierarchical netlist would not flag it. Run per sheet and expect
    some sheet-boundary pins to show up as floating pins -- that is a
    real known-false-positive category, not a crash.
  - Multi-unit parts fall back to "unit 0 (shared) + unit 1" pins if the
    instance's own unit number has no pins defined for it in
    lib_symbols -- a defensive fallback for malformed/edge-case library
    entries, not a claim that every unit's pins are always resolved
    correctly for parts with many gates/units.
  - No bus (`{...}`) expansion -- bus wires and bus labels are read as
    plain wires/labels (whatever coordinate/text they literally have) so
    single-net-name bus members are not individually cross-checked.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from kicad_dataset_convert import find_all, find_child, parse_sexpr

GND_NAME_HINTS = ("GND", "GNDREF", "VSS", "AGND", "DGND")


# ---------------------------------------------------------------------------
# Geometry: library-local pin offset -> absolute schematic coordinate
# ---------------------------------------------------------------------------

def _mirror_point(x: float, y: float, mirror) -> tuple:
    if mirror == "x":
        return x, -y
    if mirror == "y":
        return -x, y
    return x, y


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple:
    theta = math.radians(angle_deg)
    c, s = round(math.cos(theta), 6), round(math.sin(theta), 6)
    return x * c - y * s, x * s + y * c


def transform_point(local_x: float, local_y: float, mirror, angle_deg: float,
                     inst_x: float, inst_y: float) -> tuple:
    """Library-local pin (x, y) -> absolute schematic (x, y), rounded to
    KiCad's own on-grid precision so equal points compare equal."""
    mx, my = _mirror_point(local_x, local_y, mirror)
    rx, ry = _rotate_point(mx, my, angle_deg)
    return round(inst_x + rx, 3), round(inst_y + ry, 3)


# ---------------------------------------------------------------------------
# lib_symbols -> {lib_id: {unit: [pin, ...]}}
# ---------------------------------------------------------------------------

def _parse_pin_node(node) -> dict:
    electrical_type = node[1] if len(node) > 1 and isinstance(node[1], str) else "unknown"
    at = find_child(node, "at")
    x, y = (float(at[1]), float(at[2])) if at else (0.0, 0.0)
    number_node = find_child(node, "number")
    number = number_node[1] if number_node else "?"
    return {"number": number, "electrical_type": electrical_type, "x": x, "y": y}


def _walk_pins(node, units: dict, current_unit: int) -> None:
    """Recurse into a lib_symbols entry, tracking which nested sub-symbol
    ("<Name>_<unit>_<body_style>") each (pin ...) node belongs to. Unit 0
    means "shared by every unit" (usually just outline graphics, but some
    parts put pins there too)."""
    for child in node:
        if not (isinstance(child, list) and child):
            continue
        if child[0] == "symbol" and isinstance(child[1], str):
            unit = current_unit
            name_parts = child[1].rsplit("_", 2)
            if len(name_parts) == 3:
                try:
                    unit = int(name_parts[1])
                except ValueError:
                    pass
            _walk_pins(child, units, unit)
        elif child[0] == "pin":
            units[current_unit].append(_parse_pin_node(child))
        else:
            _walk_pins(child, units, current_unit)


def get_lib_pin_table(root) -> dict:
    table = {}
    lib_symbols = find_child(root, "lib_symbols")
    if lib_symbols is None:
        return table
    for node in lib_symbols:
        if not (isinstance(node, list) and node and node[0] == "symbol" and isinstance(node[1], str)):
            continue
        units = defaultdict(list)
        _walk_pins(node, units, current_unit=1)
        table[node[1]] = dict(units)
    return table


def pins_for_instance(pin_table: dict, lib_id: str, unit: int) -> list:
    units = pin_table.get(lib_id, {})
    pins = list(units.get(0, [])) + list(units.get(unit, []))
    if not pins and unit != 1:
        pins = list(units.get(0, [])) + list(units.get(1, []))
    return pins


# ---------------------------------------------------------------------------
# Placed instances / wires / junctions / labels / no-connects
# ---------------------------------------------------------------------------

def get_placed_instances(root) -> list:
    """Placed symbols only (see kicad_dataset_convert.get_symbol_instances
    for the same lib_id-node-vs-bare-string distinction, verified there
    against a real file)."""
    instances = []
    for node in find_all(root, "symbol"):
        lib_id_node = find_child(node, "lib_id")
        at_node = find_child(node, "at")
        if lib_id_node is None or at_node is None:
            continue
        unit_node = find_child(node, "unit")
        mirror_node = find_child(node, "mirror")
        reference = None
        for prop in find_all(node, "property"):
            if len(prop) > 2 and prop[1] == "Reference":
                reference = prop[2]
        instances.append({
            "lib_id": lib_id_node[1],
            "x": float(at_node[1]), "y": float(at_node[2]),
            "angle": float(at_node[3]) if len(at_node) > 3 else 0.0,
            "unit": int(unit_node[1]) if unit_node else 1,
            "mirror": mirror_node[1] if mirror_node else None,
            "reference": reference,
        })
    return instances


def get_wires(root) -> list:
    wires = []
    for node in find_all(root, "wire"):
        pts = find_child(node, "pts")
        if pts is None:
            continue
        coords = [(round(float(xy[1]), 3), round(float(xy[2]), 3))
                  for xy in pts if isinstance(xy, list) and xy and xy[0] == "xy"]
        if len(coords) >= 2:
            wires.append((coords[0], coords[-1]))
    return wires


def get_junctions(root) -> list:
    points = []
    for node in find_all(root, "junction"):
        at = find_child(node, "at")
        if at:
            points.append((round(float(at[1]), 3), round(float(at[2]), 3)))
    return points


def get_labels(root) -> list:
    labels = []
    for tag in ("label", "global_label", "hierarchical_label"):
        for node in find_all(root, tag):
            at = find_child(node, "at")
            if at is None or not isinstance(node[1], str):
                continue
            labels.append({"text": node[1], "kind": tag,
                            "point": (round(float(at[1]), 3), round(float(at[2]), 3))})
    return labels


def get_no_connects(root) -> set:
    points = set()
    for node in find_all(root, "no_connect"):
        at = find_child(node, "at")
        if at:
            points.add((round(float(at[1]), 3), round(float(at[2]), 3)))
    return points


# ---------------------------------------------------------------------------
# Union-find over (x, y) coordinate keys
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class CircuitFinding(dict):
    def __init__(self, rule: str, severity: str, message: str, point=None):
        super().__init__(rule=rule, severity=severity, message=message, point=point)


def check_schematic(sch_path: Path) -> list:
    root = parse_sexpr(sch_path.read_text(encoding="utf-8"))
    pin_table = get_lib_pin_table(root)
    instances = get_placed_instances(root)
    wires = get_wires(root)
    junctions = get_junctions(root)
    labels = get_labels(root)
    no_connects = get_no_connects(root)

    findings = []
    uf = UnionFind()
    for a, b in wires:
        uf.union(a, b)

    point_pins = defaultdict(list)   # point -> [(instance, pin), ...]
    point_touch = defaultdict(int)   # point -> how many wire/junction/label items sit there
    net_labels = defaultdict(set)    # union-find root -> {label text, ...}
    references_seen = defaultdict(list)

    for a, b in wires:
        point_touch[a] += 1
        point_touch[b] += 1
    for j in junctions:
        point_touch[j] += 1
        uf.find(j)

    is_power = lambda lib_id: lib_id.startswith("power:")

    for inst in instances:
        if not is_power(inst["lib_id"]) and inst["reference"]:
            references_seen[inst["reference"]].append(inst)
        pins = pins_for_instance(pin_table, inst["lib_id"], inst["unit"])
        for pin in pins:
            point = transform_point(pin["x"], pin["y"], inst["mirror"], inst["angle"], inst["x"], inst["y"])
            point_pins[point].append((inst, pin))
            uf.find(point)
            if is_power(inst["lib_id"]):
                net_name = inst["lib_id"].split(":", 1)[1]
                net_labels[uf.find(point)].add(net_name)
                point_touch[point] += 1

    for label in labels:
        point_touch[label["point"]] += 1
        uf.find(label["point"])
        net_labels[uf.find(label["point"])].add(label["text"])

    # floating_pin / no_connect_conflict
    for point, pin_list in point_pins.items():
        others_here = point_touch[point] + max(0, len(pin_list) - 1)
        at_no_connect = point in no_connects
        for inst, pin in pin_list:
            if is_power(inst["lib_id"]):
                continue
            if pin["electrical_type"] == "no_connect":
                continue
            if at_no_connect and others_here > 0:
                findings.append(CircuitFinding(
                    "no_connect_conflict", "error",
                    f'{inst["reference"] or inst["lib_id"]} pin {pin["number"]} is marked '
                    f'no-connect at {point} but is also wired to something else',
                    point))
            elif not at_no_connect and others_here == 0:
                findings.append(CircuitFinding(
                    "floating_pin", "error",
                    f'{inst["reference"] or inst["lib_id"]} pin {pin["number"]} '
                    f'({pin["electrical_type"]}) at {point} touches no wire, junction, '
                    f'other pin, or label',
                    point))

    # duplicate_reference
    for ref, insts in references_seen.items():
        if ref and len(insts) > 1:
            findings.append(CircuitFinding(
                "duplicate_reference", "error",
                f'reference "{ref}" is used by {len(insts)} placed symbols '
                f'({", ".join(i["lib_id"] for i in insts)})',
                None))

    # net_name_conflict
    for root_point, names in net_labels.items():
        if len(names) > 1:
            findings.append(CircuitFinding(
                "net_name_conflict", "warning",
                f'net touches {len(names)} different names that are electrically '
                f'joined: {", ".join(sorted(names))} -- likely an unintended short',
                root_point))

    # missing_ground (schematic-level, only meaningful if the sheet has power pins at all)
    has_power_pin = any(
        pin["electrical_type"] == "power_in"
        for inst in instances if not is_power(inst["lib_id"])
        for pin in pins_for_instance(pin_table, inst["lib_id"], inst["unit"])
    )
    has_ground_net = any(
        any(name.upper().startswith(hint) for hint in GND_NAME_HINTS for name in names)
        for names in net_labels.values()
    )
    if has_power_pin and not has_ground_net:
        findings.append(CircuitFinding(
            "missing_ground", "warning",
            "schematic has power-input pins but no net named GND/GNDREF/VSS/AGND/DGND "
            "was found -- check this isn't a sheet-boundary false positive (see module docstring)",
            None))

    return findings


def check_path(path: Path, recursive: bool) -> dict:
    files = sorted(path.rglob("*.kicad_sch")) if path.is_dir() and recursive else \
        ([path] if path.is_file() else sorted(path.glob("*.kicad_sch")))
    return {str(f): check_schematic(f) for f in files}


def main():
    parser = argparse.ArgumentParser(description="Electrical rule check for KiCad .kicad_sch files")
    parser.add_argument("path", type=Path, help=".kicad_sch file, or a directory of them")
    parser.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    args = parser.parse_args()

    results = check_path(args.path, args.recursive)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        total = 0
        for file_path, findings in results.items():
            if not findings:
                continue
            print(f"\n{file_path}")
            for f in findings:
                total += 1
                print(f'  [{f["severity"]}] {f["rule"]}: {f["message"]}')
        n_files = len(results)
        n_clean = sum(1 for f in results.values() if not f)
        print(f"\n{n_files} file(s) checked, {n_clean} clean, {total} finding(s) total")

    sys.exit(1 if any(f["severity"] == "error" for findings in results.values() for f in findings) else 0)


if __name__ == "__main__":
    main()
