"""Generates the Schmitt-trigger training corpus for schmitt_trigger_train.py.

No real Schmitt-trigger .kicad_sch dataset exists (user confirmed 2026-09-11,
see to_do_list.md #29) -- this hand-designs the circuit topology in Python
(coordinate arithmetic, not hand-typed S-expressions) and self-validates every
generated file through tranning/circuit_rule_check.py (to-do #23's from-
scratch ERC) before it is ever written to disk or trained on, so a wiring
mistake is caught here instead of silently teaching the model a broken
circuit.

Five topologies, several component-value variants each (see build_corpus()):

  1. Non-inverting op-amp Schmitt trigger: Vin -> R1 -> node A (op-amp IN+),
     op-amp OUT -> R2 -> node A (positive feedback = hysteresis), IN- tied to
     GND (symmetric-about-0 threshold; single-supply reference-divider
     version is a documented simplification, not modelled here -- see
     schmitt_trigger_train.py docstring). V+/V- powered from VCC/GND.
  2. Single logic Schmitt-trigger inverter (e.g. 74HC14 gate already has a
     Schmitt input): Vin -> IN, OUT -> Vout, VCC/GND. Far simpler wiring than
     (1), included for topology diversity in a very small corpus.
  3. Inverting op-amp amplifier: Vin -> R1 -> node A (op-amp IN-), OUT -> R2
     -> node A (negative feedback), IN+ tied to GND. Gain = -R2/R1.
  4. Non-inverting op-amp amplifier: Vin -> IN+ directly, IN- <- R1 -> GND
     and IN- <- R2 -> OUT (negative feedback). Gain = 1 + R2/R1.
  5. Op-amp comparator: Vin -> IN- directly, a VCC/GND resistor divider
     (R1 top half, R2 bottom half) sets the reference voltage at IN+, no
     feedback (open-loop).
  6. Op-amp voltage follower / buffer: Vin -> IN+ directly, IN- wired
     straight to OUT (100% feedback, gain = 1). No resistors.

  Topologies 3-6 (2026-09-11/12, continuing the KiCad-generator scope from
  ErrorLog #23) were added after confirming the standard gain formulas and
  comparator wiring against circuitdigest.com/geeksforgeeks.org (inverting
  A=-R2/R1, non-inverting A=1+R2/R1) and renesas.com/electronics-tutorials.ws
  (op-amp-as-comparator with a reference-divider input) -- same "verify
  against a real source, don't guess" bar as the connectivity math itself
  (see circuit_rule_check.py's docstring for that precedent).

All instances share one fixed, minimal lib_symbols block (a 2-pin passive
part, a 5-pin op-amp, a 2-pin (IN/OUT) + 2-pin power inverter gate, plus
power:GND/power:VCC) -- same boilerplate-reuse idea as a real KiCad project
reusing one symbol library across many sheets.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tranning/ (for circuit_rule_check)

from circuit_rule_check import check_schematic  # noqa: E402

OUT_CORPUS = Path(__file__).resolve().parent / "schmitt_trigger_corpus.txt"
OUT_PAIRS = Path(__file__).resolve().parent.parent.parent / "data" / "schmitt_trigger_pairs.json"

# Library block re-attached to every body by wrap_kicad() -- never generated
# by the model, so it can carry real graphics at no cost to the model.
#
# 2026-10-27 redesign (to_do_list.md #52). The earlier library had no body
# graphics and an op-amp whose five 2.54mm pins all met in its centre (a
# zero-size part), so every power symbol / resistor next to it overlapped,
# wires ran diagonally through it, and -- because its pins were written in
# sheet coordinates while KiCad libraries are y-up -- real KiCad swapped
# IN+/IN- and V+/V-. Now: KiCad-sized parts (the op-amp matches the shape of
# KiCad's own LM358 symbol: inputs at x=-7.62, output at x=+7.62, supply pins
# at x=-2.54 reaching y=+-7.62), every pin position in the bodies below is
# computed from THIS block with circuit_rule_check.transform_point() (the
# formula verified against KiCad's demo files), wiring is orthogonal, and
# every body is checked for errors AND overlapping_symbols before training.
#
# (name ...)/(effects ...) clauses are still left out where KiCad does not
# need them; power pins keep their name, which is what KiCad uses to name
# the GND / VCC net.
LIB_SYMBOLS = """
(lib_symbols
    (symbol "Sinco:R" (pin_names hide) (pin_numbers hide)
        (property "Reference" "R" (at 0 0 0))
        (symbol "R_0_1"
            (rectangle (start -2.54 -1.016) (end 2.54 1.016) (stroke (width 0.254)) (fill (type none)))
        )
        (symbol "R_1_1"
            (pin passive line (at -3.81 0 0) (length 1.27) (number "1"))
            (pin passive line (at 3.81 0 180) (length 1.27) (number "2"))
        )
    )
    (symbol "Sinco:OpAmp" (pin_names hide)
        (property "Reference" "U" (at 0 0 0))
        (symbol "OpAmp_0_1"
            (polyline (pts (xy -5.08 5.08) (xy 5.08 0) (xy -5.08 -5.08) (xy -5.08 5.08)) (stroke (width 0.254)) (fill (type background)))
            (polyline (pts (xy -4.318 2.54) (xy -3.302 2.54)) (stroke (width 0.1524)) (fill (type none)))
            (polyline (pts (xy -3.81 3.048) (xy -3.81 2.032)) (stroke (width 0.1524)) (fill (type none)))
            (polyline (pts (xy -4.318 -2.54) (xy -3.302 -2.54)) (stroke (width 0.1524)) (fill (type none)))
        )
        (symbol "OpAmp_1_1"
            (pin input line (at -7.62 2.54 0) (length 2.54) (number "1"))
            (pin input line (at -7.62 -2.54 0) (length 2.54) (number "2"))
            (pin output line (at 7.62 0 180) (length 2.54) (number "3"))
            (pin power_in line (at -2.54 7.62 270) (length 3.81) (number "4"))
            (pin power_in line (at -2.54 -7.62 90) (length 3.81) (number "5"))
        )
    )
    (symbol "Sinco:SchmittInverter" (pin_names hide)
        (property "Reference" "U" (at 0 0 0))
        (symbol "SchmittInverter_0_1"
            (polyline (pts (xy -5.08 3.81) (xy 4.064 0) (xy -5.08 -3.81) (xy -5.08 3.81)) (stroke (width 0.254)) (fill (type background)))
            (circle (center 4.572 0) (radius 0.508) (stroke (width 0.254)) (fill (type none)))
            (polyline (pts (xy -3.556 -0.762) (xy -2.54 -0.762) (xy -2.032 0.762) (xy -1.016 0.762)) (stroke (width 0.1524)) (fill (type none)))
        )
        (symbol "SchmittInverter_1_1"
            (pin input line (at -7.62 0 0) (length 2.54) (number "1"))
            (pin output line (at 7.62 0 180) (length 2.54) (number "2"))
            (pin power_in line (at -2.54 7.62 270) (length 5.08) (number "14"))
            (pin power_in line (at -2.54 -7.62 90) (length 5.08) (number "7"))
        )
    )
    (symbol "power:GND" (power) (pin_names hide)
        (property "Reference" "#PWR" (at 0 0 0))
        (property "Value" "GND" (at 0 -3.81 0))
        (symbol "GND_0_1"
            (polyline (pts (xy 0 0) (xy 0 -1.27) (xy 1.27 -1.27) (xy 0 -2.54) (xy -1.27 -1.27) (xy 0 -1.27)) (stroke (width 0)) (fill (type none)))
        )
        (symbol "GND_1_1" (pin power_in line (at 0 0 270) (length 0) hide (name "GND") (number "1")))
    )
    (symbol "power:VCC" (power) (pin_names hide)
        (property "Reference" "#PWR" (at 0 0 0))
        (property "Value" "VCC" (at 0 3.81 0))
        (symbol "VCC_0_1"
            (polyline (pts (xy 0 0) (xy 0 1.27)) (stroke (width 0)) (fill (type none)))
            (polyline (pts (xy -0.762 1.27) (xy 0.762 1.27)) (stroke (width 0)) (fill (type none)))
        )
        (symbol "VCC_1_1" (pin power_in line (at 0 0 90) (length 0) hide (name "VCC") (number "1")))
    )
)
"""


def _lib_pins() -> dict:
    from circuit_rule_check import get_lib_pin_table
    from kicad_dataset_convert import parse_sexpr
    table = get_lib_pin_table(parse_sexpr(f"(kicad_sch {LIB_SYMBOLS})"))
    return {lib_id: {p["number"]: p for unit in units.values() for p in unit} for lib_id, units in table.items()}


_PINS = _lib_pins()


def _pin(lib_id, number, x, y, angle=0):
    """Sheet coordinate of a placed part's pin tip (same formula KiCad uses)."""
    from circuit_rule_check import transform_point
    p = _PINS[lib_id][str(number)]
    return transform_point(p["x"], p["y"], None, angle, x, y)


def _sym(lib_id, ref, x, y, angle=0, value=None, ref_at=None, value_at=None):
    """Placed part. Text defaults to centred above the part; a part turned
    90/270 gets field angle 90 so KiCad still shows its text horizontally
    (KiCad adds the symbol's rotation to its fields' angle)."""
    fa = 90 if angle % 180 == 90 else 0
    hide = " (effects hide)" if lib_id.startswith("power:") else ""   # "#PWR01" is bookkeeping, not shown
    rx, ry = ref_at or (x, y - 4.064)
    value_prop = ""
    if value:
        vx, vy = value_at or (x, y - 2.286)
        value_prop = f'(property "Value" "{value}" (at {vx} {vy} {fa}))'
    return (
        f'(symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) '
        f'(property "Reference" "{ref}" (at {rx} {ry} {fa}){hide}) {value_prop})'
    )


def _power(net, ref, point, text_at=None):
    """VCC always points up, GND always points down (angle 0); the net name
    sits just past the graphic (above VCC / below GND) unless text_at moves it."""
    default = (point[0], point[1] - 2.54) if net != "GND" else (point[0], point[1] + 3.81)
    return _sym(f"power:{net}", ref, point[0], point[1], value=net, value_at=text_at or default)


def _wire(p1, p2):
    return f'(wire (pts (xy {p1[0]} {p1[1]}) (xy {p2[0]} {p2[1]})))'


def _wires(*points):
    """Polyline as separate straight wires (every bend is a wire end)."""
    return "".join(_wire(a, b) for a, b in zip(points, points[1:]))


def _label(text, x, y, angle=0):
    """angle 0: text runs right of the point; 180: text runs left."""
    return f'(label "{text}" (at {x} {y} {angle}))'


def _junction(x, y):
    return f"(junction (at {x} {y}))"


def _add_uuids(body: str) -> str:
    """Every placed symbol gets a deterministic uuid here instead of in the
    body: they are pure bookkeeping (~90 tokens per circuit the model would
    otherwise have to reproduce). Bodies that already carry one (output of
    a model trained before 2026-10-27) are left alone."""
    count = iter(range(1, 10_000))
    return re.sub(r'\(unit 1\) (?!\(uuid)',
                  lambda m: f'(unit 1) (uuid "00000000-0000-0000-0000-{next(count):012d}") ', body)


def wrap_kicad(body: str) -> str:
    return f'(kicad_sch (version 20231120) (generator sinco_schmitt_gen) (paper "A4") {LIB_SYMBOLS} {_add_uuids(body)})'


# Shared op-amp placement and the points every op-amp template routes to.
_U = (120.0, 100.0)
_OPAMP = "Sinco:OpAmp"
_IN_PLUS = _pin(_OPAMP, 1, *_U)      # (112.38, 97.46)  upper input
_IN_MINUS = _pin(_OPAMP, 2, *_U)     # (112.38, 102.54) lower input
_OUT = _pin(_OPAMP, 3, *_U)          # (127.62, 100.0)
_V_PLUS = _pin(_OPAMP, 4, *_U)       # (117.46, 92.38)  top
_V_MINUS = _pin(_OPAMP, 5, *_U)      # (117.46, 107.62) bottom
_OUT_NODE = (132.08, _OUT[1])        # where the feedback leaves the output wire
_VOUT_PT = (137.16, _OUT[1])
_FB_TOP_Y = 82.55                    # feedback path above the op-amp (clear of VCC and its text)
_FB_BOTTOM_Y = 120.65                # feedback path below the op-amp (clear of GND and its text)


def _opamp_core():
    """The op-amp, its supplies and the output wire to VOUT (with the
    feedback tap at _OUT_NODE)."""
    return (
        _sym(_OPAMP, "U1", *_U, value="LM358",
             ref_at=(_U[0] + 3.81, _U[1] - 7.62), value_at=(_U[0] + 3.81, _U[1] - 5.715))
        + _power("VCC", "#PWR01", _V_PLUS)
        + _power("GND", "#PWR02", _V_MINUS)
        + _wires(_OUT, _OUT_NODE, _VOUT_PT)
    )


def _feedback(node, y, r_value, ref="R2"):
    """Horizontal resistor from `node` (left) round to _OUT_NODE (right) at
    height y: node -> up/down -> R -> right -> down/up -> output."""
    r_x = 115.57
    p1, p2 = _pin("Sinco:R", 1, r_x, y), _pin("Sinco:R", 2, r_x, y)
    return (
        _sym("Sinco:R", ref, r_x, y, value=r_value)
        + _wires(node, (node[0], y), p1)
        + _wires(p2, (_OUT_NODE[0], y), _OUT_NODE)
        + _junction(*_OUT_NODE)
    )


def build_opamp_schmitt(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Non-inverting op-amp Schmitt trigger: VIN -> R1 -> node A -> IN+,
    OUT -> R2 -> node A (positive feedback, drawn ABOVE the op-amp),
    IN- -> GND. Returns the BODY only (wrap_kicad() adds the library)."""
    r1 = (99.06, _IN_PLUS[1])
    r1_p1, r1_p2 = _pin("Sinco:R", 1, *r1), _pin("Sinco:R", 2, *r1)
    node_a = (107.95, _IN_PLUS[1])
    vin_pt = (r1_p1[0] - 5.08, r1[1])
    gnd_in = (108.585, _IN_MINUS[1])
    return (
        _sym("Sinco:R", "R1", *r1, value=r1_value)
        + _feedback(node_a, _FB_TOP_Y, r2_value)
        + _opamp_core()
        + _power("GND", "#PWR03", gnd_in)
        + _wires(vin_pt, r1_p1)
        + _wires(r1_p2, node_a, _IN_PLUS)
        + _wires(_IN_MINUS, gnd_in)
        + _junction(*node_a)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *_VOUT_PT)
    )


def build_inverting_amp(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Inverting amplifier (gain = -R2/R1): VIN -> R1 -> node A -> IN-,
    OUT -> R2 -> node A (negative feedback, drawn BELOW the op-amp),
    IN+ -> GND."""
    r1 = (99.06, _IN_MINUS[1])
    r1_p1, r1_p2 = _pin("Sinco:R", 1, *r1), _pin("Sinco:R", 2, *r1)
    node_a = (107.95, _IN_MINUS[1])
    vin_pt = (r1_p1[0] - 5.08, r1[1])
    gnd_in = (108.585, _IN_PLUS[1])
    return (
        _sym("Sinco:R", "R1", *r1, value=r1_value)
        + _feedback(node_a, _FB_BOTTOM_Y, r2_value)
        + _opamp_core()
        + _power("GND", "#PWR03", gnd_in, text_at=(gnd_in[0] - 3.81, gnd_in[1] + 1.905))
        + _wires(vin_pt, r1_p1)
        + _wires(r1_p2, node_a, _IN_MINUS)
        + _wires(_IN_PLUS, gnd_in)
        + _junction(*node_a)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *_VOUT_PT)
    )


def build_noninverting_amp(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Non-inverting amplifier (gain = 1 + R2/R1): VIN -> IN+ directly,
    node A on IN-: R1 (vertical) -> GND, R2 -> OUT (feedback BELOW)."""
    node_a = (107.95, _IN_MINUS[1])
    r1_node = (100.33, _IN_MINUS[1])
    r1 = (r1_node[0], r1_node[1] + 3.81)                  # standing resistor, top pin on r1_node
    r1_top, r1_bottom = sorted((_pin("Sinco:R", 1, *r1, 90), _pin("Sinco:R", 2, *r1, 90)), key=lambda p: p[1])
    vin_pt = (r1_node[0] - 5.08, _IN_PLUS[1])
    return (
        _sym("Sinco:R", "R1", *r1, angle=90, value=r1_value,
             ref_at=(r1[0] - 3.81, r1[1] - 1.27), value_at=(r1[0] - 6.35, r1[1] + 1.27))
        + _feedback(node_a, _FB_BOTTOM_Y, r2_value)
        + _opamp_core()
        + _power("GND", "#PWR03", r1_bottom)
        + _wires(vin_pt, _IN_PLUS)
        + _wires(r1_top, node_a, _IN_MINUS)
        + _junction(*node_a)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *_VOUT_PT)
    )


def build_comparator(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Open-loop comparator: VIN -> IN- (brought in from below), and a
    VCC -R1- node -R2- GND divider sets the reference on IN+."""
    node = (100.33, _IN_PLUS[1])
    r1 = (node[0], node[1] - 3.81)                         # above the node, to VCC
    r2 = (node[0], node[1] + 3.81)                         # below the node, to GND
    r1_top = min((_pin("Sinco:R", 1, *r1, 90), _pin("Sinco:R", 2, *r1, 90)), key=lambda p: p[1])
    r2_bottom = max((_pin("Sinco:R", 1, *r2, 90), _pin("Sinco:R", 2, *r2, 90)), key=lambda p: p[1])
    vin_corner = (108.585, _IN_MINUS[1])
    vin_pt = (vin_corner[0], 113.03)
    return (
        _sym("Sinco:R", "R1", *r1, angle=90, value=r1_value,
             ref_at=(r1[0] - 3.81, r1[1] - 1.27), value_at=(r1[0] - 6.35, r1[1] + 1.27))
        + _sym("Sinco:R", "R2", *r2, angle=90, value=r2_value,
               ref_at=(r2[0] - 3.81, r2[1] - 1.27), value_at=(r2[0] - 6.35, r2[1] + 1.27))
        + _opamp_core()
        + _power("VCC", "#PWR03", r1_top)
        + _power("GND", "#PWR04", r2_bottom)
        + _wires(node, _IN_PLUS)
        + _wires(vin_pt, vin_corner, _IN_MINUS)
        + _junction(*node)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *_VOUT_PT)
    )


def build_schmitt_inverter(gate_label="U1", in_label="VIN", out_label="VOUT") -> str:
    """Single Schmitt-input logic gate (e.g. 74HC14): VIN -> IN, OUT -> VOUT,
    VCC on top, GND below. Body only, see build_opamp_schmitt()."""
    u = (110.0, 100.0)
    gate = "Sinco:SchmittInverter"
    in_pin, out_pin = _pin(gate, 1, *u), _pin(gate, 2, *u)
    vin_pt, vout_pt = (in_pin[0] - 5.08, u[1]), (out_pin[0] + 5.08, u[1])
    return (
        _sym(gate, gate_label, *u, value="74HC14",
             ref_at=(u[0] + 3.81, u[1] - 7.62), value_at=(u[0] + 3.81, u[1] - 5.715))
        + _power("VCC", "#PWR01", _pin(gate, 14, *u))
        + _power("GND", "#PWR02", _pin(gate, 7, *u))
        + _wires(vin_pt, in_pin)
        + _wires(out_pin, vout_pt)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *vout_pt)
    )


def build_voltage_follower(in_label="VIN", out_label="VOUT") -> str:
    """Unity-gain voltage follower: VIN -> IN+, IN- wired back to OUT
    (100% feedback, routed BELOW the op-amp). Not in the training corpus
    (see build_corpus()), kept as a validated template."""
    node = (107.95, _IN_MINUS[1])
    vin_pt = (node[0] - 5.08, _IN_PLUS[1])
    return (
        _opamp_core()
        + _wires(vin_pt, _IN_PLUS)
        + _wires(_IN_MINUS, node, (node[0], _FB_BOTTOM_Y), (_OUT_NODE[0], _FB_BOTTOM_Y), _OUT_NODE)
        + _junction(*_OUT_NODE)
        + _label(in_label, *vin_pt, 180)
        + _label(out_label, *_VOUT_PT)
    )


# --- prompt phrasing pools (Chinese, several ways to ask for the same circuit) ---

_OPAMP_PROMPTS = [
    "幫我設計一個史密特觸發電路，R1 用 {r1}，R2 用 {r2}",
    "用運算放大器設計史密特觸發器，正回授電阻 R2 = {r2}，輸入電阻 R1 = {r1}",
    "我要一個 op-amp 史密特觸發電路的 KiCad 電路圖，R1={r1} R2={r2}",
]
_INVERTER_PROMPTS = [
    "用 74HC14 邏輯閘設計一個簡單的史密特觸發反相器電路",
    "幫我畫一個史密特觸發反相器（Schmitt trigger inverter）電路圖",
    "設計一個單閘史密特觸發電路，輸入 VIN 輸出 VOUT",
]
_INV_AMP_PROMPTS = [
    "幫我設計一個反相放大器電路，R1 用 {r1}，R2 用 {r2}",
    "用運算放大器做一個反相放大電路，輸入電阻 R1={r1}，回授電阻 R2={r2}",
    "我要一個 op-amp 反相放大器的 KiCad 電路圖，R1={r1} R2={r2}",
]
# "同相放大器" (in-phase amp), not the also-valid "非反相放大器" (non-inverting): attempt 1
# (2026-09-11 training run) used "非反相" here and the trained model deterministically
# (reproduced under near-greedy decoding) generated a hybrid of this topology's reply and
# build_inverting_amp()'s -- "反相放大器" vs "非反相放大器" differ by one token ("非") in an
# otherwise near-identical prompt, same single-negation-token confusion class already
# documented for the chat model in ErrorLog #22. "同相放大器" is the more common term in
# Chinese EE material anyway (see enroo.com tutorial titled 同相放大器) and shares no
# token-level minimal pair with "反相放大器", which removes the ambiguity at the data level
# instead of just tuning around it.
_NONINV_AMP_PROMPTS = [
    "幫我設計一個同相放大器電路，R1 用 {r1}，R2 用 {r2}",
    "用運算放大器做一個同相放大電路，接地電阻 R1={r1}，回授電阻 R2={r2}",
    "我要一個 op-amp 同相放大器的 KiCad 電路圖，R1={r1} R2={r2}",
]
_COMPARATOR_PROMPTS = [
    "幫我設計一個電壓比較器電路，分壓電阻 R1 用 {r1}，R2 用 {r2}",
    "用運算放大器做一個比較器電路，參考電壓分壓電阻 R1={r1}，R2={r2}",
    "我要一個 op-amp comparator 的 KiCad 電路圖，R1={r1} R2={r2}",
]
_FOLLOWER_PROMPTS = [
    "幫我設計一個電壓隨耦器電路",
    "用運算放大器做一個電壓跟隨器（unity gain buffer）電路",
    "我要一個 op-amp voltage follower 的 KiCad 電路圖",
]

_OPAMP_VARIANTS = [("10k", "100k"), ("10k", "47k"), ("4.7k", "100k"), ("22k", "220k"), ("10k", "220k")]
_INV_AMP_VARIANTS = [("10k", "100k"), ("10k", "47k"), ("4.7k", "100k"), ("22k", "220k")]
_NONINV_AMP_VARIANTS = [("10k", "100k"), ("10k", "47k"), ("4.7k", "100k"), ("22k", "220k")]
_COMPARATOR_VARIANTS = [("10k", "10k"), ("10k", "20k"), ("22k", "10k"), ("47k", "47k")]

# Literal placeholder tokens instead of a real resistor value in every reply the corpus ever
# trains on -- see build_corpus() docstring for why (2026-09-12 audit: 18/18 structure-correct,
# 1/18 value-correct). schmitt_trigger_train.fill_values() substitutes these with the value the
# user actually typed, parsed straight out of their prompt, after generation.
PLACEHOLDER_R1 = "R1_VALUE"
PLACEHOLDER_R2 = "R2_VALUE"


def build_corpus() -> list[dict]:
    """Returns [{"prompt": ..., "reply": kicad_sch_BODY_text}, ...] -- reply
    is body-only (see build_opamp_schmitt()); wrap_kicad(reply) is what
    actually gets validated/written to a real .kicad_sch file.

    Every reply's R1/R2 Value text is the literal placeholder token
    (PLACEHOLDER_R1/PLACEHOLDER_R2), not the real value requested in that
    pair's prompt: a real training run's output was audited end-to-end
    (2026-09-12, all 18 unique circuits, near-greedy decoding) and while
    circuit *structure* was correct 18/18 times (0 ERC errors anywhere),
    the *value* text matched what was actually asked for only 1/18 times --
    this 6-layer/192-dim model, trained on only 4-5 value variants per
    topology, hadn't learned to copy an arbitrary number out of the prompt,
    it had learned some other (wrong) value association instead. Same fix
    philosophy as wrap_kicad()'s lib_symbols reattachment (attempt 4,
    schmitt_trigger_train.py's history comment): don't spend a tiny model's
    limited capacity reproducing something that's already known
    deterministically. This also collapses each topology's many value
    variants down to ONE unique reply text (was 18 unique circuits, now 5),
    which is a strictly easier, less ambiguous SFT target -- the model only
    has to learn "which of 5 templates does this request want", not
    "which template AND which exact numbers".
    """
    pairs = []
    for builder, variants, templates in (
        (build_opamp_schmitt, _OPAMP_VARIANTS, _OPAMP_PROMPTS),
        (build_inverting_amp, _INV_AMP_VARIANTS, _INV_AMP_PROMPTS),
        (build_noninverting_amp, _NONINV_AMP_VARIANTS, _NONINV_AMP_PROMPTS),
        (build_comparator, _COMPARATOR_VARIANTS, _COMPARATOR_PROMPTS),
    ):
        template_text = builder(PLACEHOLDER_R1, PLACEHOLDER_R2)
        for r1, r2 in variants:
            for template in templates:
                pairs.append({"prompt": template.format(r1=r1, r2=r2), "reply": template_text})

    inverter_text = build_schmitt_inverter()
    for template in _INVERTER_PROMPTS:
        pairs.append({"prompt": template, "reply": inverter_text})

    # build_voltage_follower() is deliberately NOT included here: adding it as a 6th topology
    # (2026-09-12) was tried and retrained -- value-substitution stayed perfect (57/57) but ERC
    # structure regressed from 54/54 to 52/57 (val_loss plateaued flat at 0.0059 for 258/400
    # epochs, a genuine capacity ceiling, not an undertrained state -- confirmed before reverting,
    # not assumed). Reverted rather than spend the remaining training-attempt budget chasing it;
    # the 5-topology set below is the one that measured 100% ERC-clean AND 100% value-correct
    # across all 54 training prompts (see to_do_list.md #29 2026-09-11/12 update). Re-add only
    # after either a longer/bigger dedicated run or accepting a lower topology count elsewhere.

    return pairs


def validate_pairs(pairs: list[dict], tmp_dir: Path) -> None:
    """Wraps each unique body with wrap_kicad(), writes it to a temp
    .kicad_sch, and runs circuit_rule_check over it -- raises AssertionError
    with the exact rule/message if a generated circuit is structurally
    broken, instead of silently training on a bad example."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for pair in pairs:
        body = pair["reply"]
        if body in seen:
            continue
        seen[body] = True
        path = tmp_dir / f"check_{len(seen)}.kicad_sch"
        path.write_text(wrap_kicad(body), encoding="utf-8")
        findings = check_schematic(path)
        # errors AND warnings (overlapping_symbols included): since the
        # 2026-10-27 redesign every template is drawn with nothing on top
        # of anything else, so the model is never trained on a messy layout
        assert not findings, f"generated circuit failed circuit_rule_check: {findings}"


def main():
    pairs = build_corpus()
    validate_pairs(pairs, Path(__file__).resolve().parent / "_schmitt_validate_tmp")

    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PAIRS.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    # pretrain corpus stays FULL wrapped files (lib_symbols included) so that
    # stage still teaches the model general .kicad_sch syntax; only the SFT
    # pairs' reply is body-only (see build_corpus()'s docstring).
    unique_bodies = list({p["reply"] for p in pairs})
    OUT_CORPUS.write_text("\n\n".join(wrap_kicad(b) for b in unique_bodies), encoding="utf-8")

    print(f"{len(pairs)} pairs ({len(unique_bodies)} unique circuits) -> {OUT_PAIRS}")
    print(f"pretrain corpus ({len(unique_bodies)} unique circuits) -> {OUT_CORPUS}")


if __name__ == "__main__":
    main()
