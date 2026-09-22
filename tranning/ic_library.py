"""Known IC pinout/package definitions -- public datasheet pinouts, hardcoded.

No real .kicad_sch reference file for any MCU was available (user confirmed
2026-09-22), so every pin number/name/function below is transcribed directly
from the part's public datasheet, not derived from an existing KiCad library
file. Used by circuit_ic_generator.py (writes a synthetic lib_symbols block
+ places the part) and by circuit_rule_check.py's MCU-aware checks
(missing_decoupling_cap / missing_reset_pullup / missing_crystal_load_caps,
which key off pin["name"] patterns like "VCC"/"RESET"/"XTAL", so they work
on ANY part carrying conventionally-named pins, not only the ones listed
here).

Local pin coordinates follow the same convention circuit_rule_check.py
already reads out of real KiCad files (_parse_pin_node): (x, y) is the pin's
OUTER TIP, not its body-edge point -- transform_point()/_ic_body_point()
(circuit_kicad_to_schemdraw.py) already know how to turn a tip position into
a drawn lead, so this table only needs to place tips.

DIP package pins are numbered counter-clockwise from the top-left notch:
1..N/2 down the left side, N/2+1..N back up the right side. dip_pin_layout()
implements that numbering generically so a second DIP part only needs its
own pin list, not new placement code.
"""

PIN_PITCH_MM = 2.54          # KiCad's own on-grid pin spacing
BODY_HALF_WIDTH_MM = 5.08    # half the DIP body width; the tip sits one more pitch out
LEAD_LENGTH_MM = 2.54        # matches circuit_kicad_to_schemdraw.IC_LEAD_INSET_MM


def dip_pin_layout(pin_count: int) -> dict:
    """pin number (1-based) -> local (x, y) tip position for a standard
    DIP-N body, pin 1 at top-left, numbered counter-clockwise."""
    if pin_count % 2 != 0:
        raise ValueError(f"DIP package pin count must be even, got {pin_count}")
    rows = pin_count // 2
    half_span = (rows - 1) / 2 * PIN_PITCH_MM
    tip_x = BODY_HALF_WIDTH_MM + LEAD_LENGTH_MM
    layout = {}
    for i in range(rows):                 # left side: pins 1..rows, top to bottom
        layout[i + 1] = (-tip_x, round(half_span - i * PIN_PITCH_MM, 3))
    for j in range(rows):                 # right side: pins rows+1..2*rows, bottom to top
        layout[rows + 1 + j] = (tip_x, round(-half_span + j * PIN_PITCH_MM, 3))
    return layout


def _pin(number: int, name: str, function: str, electrical_type: str) -> dict:
    return {"number": str(number), "name": name, "function": function,
            "electrical_type": electrical_type}


# function values consumed by circuit_ic_generator.py / circuit_rule_check.py:
#   "power"  -- IC supply pin, needs a decoupling capacitor to a ground net
#   "ground" -- IC ground pin, wired straight to a power:GND symbol
#   "reset"  -- needs a pull-up resistor to a power net
#   "xtal"   -- oscillator pin, needs a load capacitor to ground
#   "aref"   -- decoupled like a power pin, but is not a supply itself
#   "gpio"   -- no MCU-specific rule applies

ATMEGA328P_PINS = [
    _pin(1, "PC6/RESET", "reset", "input"),
    _pin(2, "PD0/RXD", "gpio", "bidirectional"),
    _pin(3, "PD1/TXD", "gpio", "bidirectional"),
    _pin(4, "PD2/INT0", "gpio", "bidirectional"),
    _pin(5, "PD3/INT1", "gpio", "bidirectional"),
    _pin(6, "PD4", "gpio", "bidirectional"),
    _pin(7, "VCC", "power", "power_in"),
    _pin(8, "GND", "ground", "power_in"),
    _pin(9, "PB6/XTAL1", "xtal", "passive"),
    _pin(10, "PB7/XTAL2", "xtal", "passive"),
    _pin(11, "PD5", "gpio", "bidirectional"),
    _pin(12, "PD6", "gpio", "bidirectional"),
    _pin(13, "PD7", "gpio", "bidirectional"),
    _pin(14, "PB0", "gpio", "bidirectional"),
    _pin(15, "PB1", "gpio", "bidirectional"),
    _pin(16, "PB2", "gpio", "bidirectional"),
    _pin(17, "PB3/MOSI", "gpio", "bidirectional"),
    _pin(18, "PB4/MISO", "gpio", "bidirectional"),
    _pin(19, "PB5/SCK", "gpio", "bidirectional"),
    _pin(20, "AVCC", "power", "power_in"),
    _pin(21, "AREF", "aref", "passive"),
    _pin(22, "GND", "ground", "power_in"),
    _pin(23, "PC0/ADC0", "gpio", "bidirectional"),
    _pin(24, "PC1/ADC1", "gpio", "bidirectional"),
    _pin(25, "PC2/ADC2", "gpio", "bidirectional"),
    _pin(26, "PC3/ADC3", "gpio", "bidirectional"),
    _pin(27, "PC4/ADC4/SDA", "gpio", "bidirectional"),
    _pin(28, "PC5/ADC5/SCL", "gpio", "bidirectional"),
]

IC_LIBRARY = {
    # lib_id matches KiCad's own official symbol library naming, so a real
    # .kicad_sch using this part lines up with this table if one ever shows up.
    "MCU_Microchip_ATmega:ATmega328P-PU": {"package": "DIP-28", "pins": ATMEGA328P_PINS},
}

# friendly CLI/lookup names -> real lib_id
IC_ALIASES = {
    "atmega328p": "MCU_Microchip_ATmega:ATmega328P-PU",
}


def resolve_ic(name: str) -> str:
    return IC_ALIASES.get(name.lower(), name)


def get_pins_with_layout(lib_id: str) -> list:
    """Pin dicts (number/name/function/electrical_type/x/y) for a known
    part, tip coordinates filled in from dip_pin_layout()."""
    lib_id = resolve_ic(lib_id)
    entry = IC_LIBRARY[lib_id]
    layout = dip_pin_layout(len(entry["pins"]))
    return [{**pin, "x": layout[int(pin["number"])][0], "y": layout[int(pin["number"])][1]}
            for pin in entry["pins"]]
