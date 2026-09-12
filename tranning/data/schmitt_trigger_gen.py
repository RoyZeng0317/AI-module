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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tranning/ (for circuit_rule_check)

from circuit_rule_check import check_schematic  # noqa: E402

OUT_CORPUS = Path(__file__).resolve().parent / "schmitt_trigger_corpus.txt"
OUT_PAIRS = Path(__file__).resolve().parent.parent.parent / "data" / "schmitt_trigger_pairs.json"

# Deliberately stripped of every (name ...)/(effects (font ...)) clause a
# real KiCad file would carry: circuit_rule_check.py's parser only ever reads
# electrical_type (node[1]), (at x y) and (number "N" ...) off a pin, and
# only (at x y) / node[1] text off a property/label -- the font/effects
# clauses exist purely for KiCad's own rendering and would just be dead
# weight tripling the token count of every example for no training signal
# (measured: cut the longest example from ~3170 to well under block_size
# without changing a single check_schematic() result).
LIB_SYMBOLS = """
(lib_symbols
    (symbol "Sinco:R"
        (property "Reference" "R" (at 0 -3 0))
        (symbol "R_1_1"
            (pin passive line (at -2.54 0 0) (length 2.54) (number "1"))
            (pin passive line (at 2.54 0 0) (length 2.54) (number "2"))
        )
    )
    (symbol "Sinco:OpAmp"
        (property "Reference" "U" (at 0 -3 0))
        (symbol "OpAmp_1_1"
            (pin input line (at -2.54 1.27 180) (length 2.54) (number "1"))
            (pin input line (at -2.54 -1.27 180) (length 2.54) (number "2"))
            (pin output line (at 2.54 0 0) (length 2.54) (number "3"))
            (pin power_in line (at 0 2.54 90) (length 2.54) (number "4"))
            (pin power_in line (at 0 -2.54 270) (length 2.54) (number "5"))
        )
    )
    (symbol "Sinco:SchmittInverter"
        (property "Reference" "U" (at 0 -3 0))
        (symbol "SchmittInverter_1_1"
            (pin input line (at -2.54 0 180) (length 2.54) (number "1"))
            (pin output line (at 2.54 0 0) (length 2.54) (number "2"))
            (pin power_in line (at 0 2.54 90) (length 2.54) (number "14"))
            (pin power_in line (at 0 -2.54 270) (length 2.54) (number "7"))
        )
    )
    (symbol "power:GND"
        (power)
        (symbol "GND_0_1" (pin power_in line (at 0 0 90) (length 0) (number "1")))
    )
    (symbol "power:VCC"
        (power)
        (symbol "VCC_0_1" (pin power_in line (at 0 0 90) (length 0) (number "1")))
    )
)
"""


def _sym(lib_id, ref, x, y, angle=0, value=None, uuid_n=1):
    value_prop = f'(property "Value" "{value}" (at {x} {y - 2} 0))' if value else ""
    return (
        f'(symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) '
        f'(uuid "00000000-0000-0000-0000-{uuid_n:012d}")'
        f'(property "Reference" "{ref}" (at {x} {y - 4} 0)) {value_prop})'
    )


def _wire(p1, p2):
    return f'(wire (pts (xy {p1[0]} {p1[1]}) (xy {p2[0]} {p2[1]})))'


def _label(text, x, y):
    return f'(label "{text}" (at {x} {y} 0))'


def _junction(x, y):
    return f"(junction (at {x} {y}))"


def wrap_kicad(body: str) -> str:
    return f'(kicad_sch (version 20231120) (generator sinco_schmitt_gen) (paper "A4") {LIB_SYMBOLS} {body})'


def build_opamp_schmitt(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Non-inverting op-amp Schmitt trigger -- see module docstring for the
    topology. All coordinates are computed, not guessed, so the pins that
    must coincide (op-amp IN+ / R2 pin2, the ground/VCC power-symbol pins)
    land on the exact transform_point() result instead of an eyeballed value.

    Returns the BODY only (placed symbols + wires + labels), not a full
    .kicad_sch file -- LIB_SYMBOLS is identical, fixed boilerplate across
    every example (~half of a full file's length, see attempt 3/4 in
    to_do_list.md #29) and is attached separately via wrap_kicad() instead of
    being something the model has to reproduce from memory on every request.
    """
    r1_x, r1_y = 100.0, 100.0                      # R1 instance (horizontal)
    r1_p1 = (r1_x - 2.54, r1_y)                     # -> Vin side
    r1_p2 = (r1_x + 2.54, r1_y)                     # -> node A

    u_x, u_y = 110.0, 100.0                         # op-amp instance
    in_plus = (u_x - 2.54, u_y + 1.27)               # node A
    in_minus = (u_x - 2.54, u_y - 1.27)              # -> GND
    out_pin = (u_x + 2.54, u_y)                      # -> node OUT
    v_plus = (u_x, u_y + 2.54)                       # -> VCC
    v_minus = (u_x, u_y - 2.54)                      # -> GND

    r2_x, r2_y = in_plus[0] - 2.54, in_plus[1]       # R2 instance: pin2 lands exactly on node A
    r2_p1 = (r2_x - 2.54, r2_y)                      # -> node OUT
    r2_p2 = (r2_x + 2.54, r2_y)                      # == in_plus (node A, same point, no wire needed)

    vin_label_pt = (r1_p1[0] - 2.46, r1_p1[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:R", "R1", r1_x, r1_y, value=r1_value, uuid_n=1)
        + _sym("Sinco:R", "R2", r2_x, r2_y, value=r2_value, uuid_n=2)
        + _sym("Sinco:OpAmp", "U1", u_x, u_y, value="LM358", uuid_n=3)
        + _sym("power:GND", "#PWR01", in_minus[0], in_minus[1], angle=90, uuid_n=4)
        + _sym("power:GND", "#PWR02", v_minus[0], v_minus[1], angle=90, uuid_n=5)
        + _sym("power:VCC", "#PWR03", v_plus[0], v_plus[1], angle=90, uuid_n=6)
        + _wire(vin_label_pt, r1_p1)
        + _wire(r1_p2, in_plus)
        + _wire(r2_p1, out_pin)
        + _wire(out_pin, vout_label_pt)
        + _junction(*in_plus)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


def build_inverting_amp(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Inverting op-amp amplifier (gain = -R2/R1). Mirror image of
    build_opamp_schmitt(): the R1/R2 network lands on IN- (pin 2, negative
    feedback) instead of IN+, and IN+ ties to GND instead of IN-."""
    r1_x, r1_y = 100.0, 100.0
    r1_p1 = (r1_x - 2.54, r1_y)
    r1_p2 = (r1_x + 2.54, r1_y)

    u_x, u_y = 110.0, 100.0
    in_plus = (u_x - 2.54, u_y + 1.27)               # -> GND
    in_minus = (u_x - 2.54, u_y - 1.27)              # node A
    out_pin = (u_x + 2.54, u_y)
    v_plus = (u_x, u_y + 2.54)
    v_minus = (u_x, u_y - 2.54)

    r2_x, r2_y = in_minus[0] - 2.54, in_minus[1]
    r2_p1 = (r2_x - 2.54, r2_y)                      # -> OUT

    vin_label_pt = (r1_p1[0] - 2.46, r1_p1[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:R", "R1", r1_x, r1_y, value=r1_value, uuid_n=1)
        + _sym("Sinco:R", "R2", r2_x, r2_y, value=r2_value, uuid_n=2)
        + _sym("Sinco:OpAmp", "U1", u_x, u_y, value="LM358", uuid_n=3)
        + _sym("power:GND", "#PWR01", in_plus[0], in_plus[1], angle=90, uuid_n=4)
        + _sym("power:GND", "#PWR02", v_minus[0], v_minus[1], angle=90, uuid_n=5)
        + _sym("power:VCC", "#PWR03", v_plus[0], v_plus[1], angle=90, uuid_n=6)
        + _wire(vin_label_pt, r1_p1)
        + _wire(r1_p2, in_minus)
        + _wire(r2_p1, out_pin)
        + _wire(out_pin, vout_label_pt)
        + _junction(*in_minus)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


def build_noninverting_amp(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Non-inverting op-amp amplifier (gain = 1 + R2/R1). Vin drives IN+
    directly (no series resistor); R1 ties IN- to GND (vertical, angle 90 --
    _sym()'s angle parameter rotates the library-local horizontal pin
    offsets, already used for the power symbols) and R2 feeds back
    OUT -> IN-."""
    u_x, u_y = 110.0, 100.0
    in_plus = (u_x - 2.54, u_y + 1.27)                # Vin drives this directly
    in_minus = (u_x - 2.54, u_y - 1.27)               # node A: R1 (-> GND) / R2 (feedback)
    out_pin = (u_x + 2.54, u_y)
    v_plus = (u_x, u_y + 2.54)
    v_minus = (u_x, u_y - 2.54)

    r1_x, r1_y = in_minus[0], in_minus[1] - 2.54      # vertical: pin2 (top) lands on node A
    r1_gnd_pt = (r1_x, r1_y - 2.54)                   # pin1 (bottom) -> GND

    r2_x, r2_y = in_minus[0] - 2.54, in_minus[1]      # horizontal: pin2 lands on node A
    r2_p1 = (r2_x - 2.54, r2_y)                       # -> OUT

    vin_label_pt = (in_plus[0] - 2.46, in_plus[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:R", "R1", r1_x, r1_y, angle=90, value=r1_value, uuid_n=1)
        + _sym("Sinco:R", "R2", r2_x, r2_y, value=r2_value, uuid_n=2)
        + _sym("Sinco:OpAmp", "U1", u_x, u_y, value="LM358", uuid_n=3)
        + _sym("power:GND", "#PWR01", r1_gnd_pt[0], r1_gnd_pt[1], angle=90, uuid_n=4)
        + _sym("power:GND", "#PWR02", v_minus[0], v_minus[1], angle=90, uuid_n=5)
        + _sym("power:VCC", "#PWR03", v_plus[0], v_plus[1], angle=90, uuid_n=6)
        + _wire(vin_label_pt, in_plus)
        + _wire(r2_p1, out_pin)
        + _wire(out_pin, vout_label_pt)
        + _junction(*in_minus)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


def build_comparator(r1_value: str, r2_value: str, in_label="VIN", out_label="VOUT") -> str:
    """Open-loop op-amp comparator: Vin -> IN- directly, a VCC/GND resistor
    divider (R1 top half, R2 bottom half) sets the reference voltage at IN+,
    no feedback resistor."""
    u_x, u_y = 110.0, 100.0
    in_plus = (u_x - 2.54, u_y + 1.27)                # divider midpoint (Vref)
    in_minus = (u_x - 2.54, u_y - 1.27)               # Vin drives this directly
    out_pin = (u_x + 2.54, u_y)
    v_plus = (u_x, u_y + 2.54)
    v_minus = (u_x, u_y - 2.54)

    r1_x, r1_y = in_plus[0], in_plus[1] + 2.54        # vertical: pin1 (bottom) lands on node
    r1_top_pt = (r1_x, r1_y + 2.54)                   # pin2 (top) -> VCC
    r2_x, r2_y = in_plus[0], in_plus[1] - 2.54        # vertical: pin2 (top) lands on node
    r2_bottom_pt = (r2_x, r2_y - 2.54)                # pin1 (bottom) -> GND

    vin_label_pt = (in_minus[0] - 2.46, in_minus[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:R", "R1", r1_x, r1_y, angle=90, value=r1_value, uuid_n=1)
        + _sym("Sinco:R", "R2", r2_x, r2_y, angle=90, value=r2_value, uuid_n=2)
        + _sym("Sinco:OpAmp", "U1", u_x, u_y, value="LM358", uuid_n=3)
        + _sym("power:VCC", "#PWR01", r1_top_pt[0], r1_top_pt[1], angle=90, uuid_n=4)
        + _sym("power:GND", "#PWR02", r2_bottom_pt[0], r2_bottom_pt[1], angle=90, uuid_n=5)
        + _sym("power:GND", "#PWR03", v_minus[0], v_minus[1], angle=90, uuid_n=6)
        + _sym("power:VCC", "#PWR04", v_plus[0], v_plus[1], angle=90, uuid_n=7)
        + _wire(vin_label_pt, in_minus)
        + _wire(out_pin, vout_label_pt)
        + _junction(*in_plus)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


def build_schmitt_inverter(gate_label="U1", in_label="VIN", out_label="VOUT") -> str:
    """Single already-Schmitt-input logic gate (e.g. 74HC14) -- far fewer
    nets than the op-amp version, included so the corpus isn't one topology
    repeated with different resistor values. Body only, see build_opamp_schmitt().
    """
    u_x, u_y = 100.0, 100.0
    in_pin = (u_x - 2.54, u_y)
    out_pin = (u_x + 2.54, u_y)
    vcc_pin = (u_x, u_y + 2.54)
    gnd_pin = (u_x, u_y - 2.54)
    vin_label_pt = (in_pin[0] - 2.46, in_pin[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:SchmittInverter", gate_label, u_x, u_y, value="74HC14", uuid_n=1)
        + _sym("power:VCC", "#PWR01", vcc_pin[0], vcc_pin[1], angle=90, uuid_n=2)
        + _sym("power:GND", "#PWR02", gnd_pin[0], gnd_pin[1], angle=90, uuid_n=3)
        + _wire(vin_label_pt, in_pin)
        + _wire(out_pin, vout_label_pt)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


def build_voltage_follower(in_label="VIN", out_label="VOUT") -> str:
    """Unity-gain op-amp voltage follower / buffer: IN- wired directly to
    OUT (100% negative feedback, gain = 1), Vin drives IN+ directly -- no
    resistors, the simplest possible op-amp circuit. Chinese term
    "電壓隨耦器" confirmed alongside 反相/非反相放大器 in the same real
    teaching-lab source already cited for those two (opentech.com.tw, see
    module docstring)."""
    u_x, u_y = 110.0, 100.0
    in_plus = (u_x - 2.54, u_y + 1.27)
    in_minus = (u_x - 2.54, u_y - 1.27)
    out_pin = (u_x + 2.54, u_y)
    v_plus = (u_x, u_y + 2.54)
    v_minus = (u_x, u_y - 2.54)

    vin_label_pt = (in_plus[0] - 2.46, in_plus[1])
    vout_label_pt = (out_pin[0] + 2.46, out_pin[1])

    body = (
        _sym("Sinco:OpAmp", "U1", u_x, u_y, value="LM358", uuid_n=1)
        + _sym("power:GND", "#PWR01", v_minus[0], v_minus[1], angle=90, uuid_n=2)
        + _sym("power:VCC", "#PWR02", v_plus[0], v_plus[1], angle=90, uuid_n=3)
        + _wire(vin_label_pt, in_plus)
        + _wire(in_minus, out_pin)
        + _wire(out_pin, vout_label_pt)
        + _label(in_label, *vin_label_pt)
        + _label(out_label, *vout_label_pt)
    )
    return body


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
        errors = [f for f in findings if f["severity"] == "error"]
        assert not errors, f"generated circuit failed circuit_rule_check: {errors}"


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
