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


def dip_pin_layout(pin_count: int, body_half_width: float = BODY_HALF_WIDTH_MM) -> dict:
    """pin number (1-based) -> local (x, y) tip position for a standard
    DIP-N body, pin 1 at top-left, numbered counter-clockwise."""
    if pin_count % 2 != 0:
        raise ValueError(f"DIP package pin count must be even, got {pin_count}")
    rows = pin_count // 2
    half_span = (rows - 1) / 2 * PIN_PITCH_MM
    tip_x = round(body_half_width + LEAD_LENGTH_MM, 3)
    layout = {}
    for i in range(rows):                 # left side: pins 1..rows, top to bottom
        layout[i + 1] = (-tip_x, round(half_span - i * PIN_PITCH_MM, 3))
    for j in range(rows):                 # right side: pins rows+1..2*rows, bottom to top
        layout[rows + 1 + j] = (tip_x, round(-half_span + j * PIN_PITCH_MM, 3))
    return layout


def header_pin_layout(pin_count: int, body_half_width: float = BODY_HALF_WIDTH_MM) -> dict:
    """pin number -> tip (x, y) for a 2-row pin header drawn the way KiCad's
    Conn_02xNN_Odd_Even symbol is: odd pins down the left side, even pins
    down the right side, pin 1 and pin 2 on the same top row."""
    if pin_count % 2 != 0:
        raise ValueError(f"2-row header pin count must be even, got {pin_count}")
    rows = pin_count // 2
    half_span = (rows - 1) / 2 * PIN_PITCH_MM
    tip_x = round(body_half_width + LEAD_LENGTH_MM, 3)
    layout = {}
    for i in range(rows):
        y = round(half_span - i * PIN_PITCH_MM, 3)
        layout[2 * i + 1] = (-tip_x, y)
        layout[2 * i + 2] = (tip_x, y)
    return layout


LAYOUTS = {"dip": dip_pin_layout, "header": header_pin_layout}


def _pin(number: int, name: str, function: str, electrical_type: str, net: str = None) -> dict:
    pin = {"number": str(number), "name": name, "function": function,
           "electrical_type": electrical_type}
    if net:
        pin["net"] = net
    return pin


# function values consumed by circuit_ic_generator.py / circuit_rule_check.py:
#   "power"  -- IC supply pin, needs a decoupling capacitor to a ground net
#   "ground" -- IC ground pin, wired straight to a power:GND symbol
#   "reset"  -- needs a pull-up resistor to a power net
#   "xtal"   -- oscillator pin, needs a load capacitor to ground
#   "aref"   -- decoupled like a power pin, but is not a supply itself
#   "gpio"   -- no MCU-specific rule applies
#   "nc"     -- not connected inside the part (or reserved, e.g. the ESP32
#               module's internal-flash pins); gets a no_connect marker
#   "supply_out" -- a supply the part PROVIDES (e.g. a Raspberry Pi header's
#               5V/3V3 pins): tied to its supply net, no decoupling cap needed
#
# Every power / supply_out pin is tied to the part's "supply" net unless the
# pin carries its own "net" (e.g. the Pi header has both +5V and +3V3 pins).

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


# ATmega2560 TQFP-100 (Arduino Mega 2560's MCU) -- datasheet "Pinout
# ATmega640/1280/2560" (100-pin TQFP). Drawn as a 2-sided schematic symbol
# (50 + 50), the usual way large MCUs are drawn in a schematic; the physical
# package is 4-sided but pin numbers are what matter electrically.
_M2560_NAMES = {
    1: "PG5", 2: "PE0/RXD0", 3: "PE1/TXD0", 4: "PE2", 5: "PE3", 6: "PE4", 7: "PE5", 8: "PE6", 9: "PE7",
    10: "VCC", 11: "GND", 12: "PH0/RXD2", 13: "PH1/TXD2", 14: "PH2", 15: "PH3", 16: "PH4", 17: "PH5",
    18: "PH6", 19: "PB0/SS", 20: "PB1/SCK", 21: "PB2/MOSI", 22: "PB3/MISO", 23: "PB4", 24: "PB5",
    25: "PB6", 26: "PB7", 27: "PH7", 28: "PG3", 29: "PG4", 30: "RESET", 31: "VCC", 32: "GND",
    33: "XTAL2", 34: "XTAL1", 35: "PL0", 36: "PL1", 37: "PL2", 38: "PL3", 39: "PL4", 40: "PL5",
    41: "PL6", 42: "PL7", 43: "PD0/SCL", 44: "PD1/SDA", 45: "PD2/RXD1", 46: "PD3/TXD1", 47: "PD4",
    48: "PD5", 49: "PD6", 50: "PD7", 51: "PG0", 52: "PG1", 53: "PC0", 54: "PC1", 55: "PC2", 56: "PC3",
    57: "PC4", 58: "PC5", 59: "PC6", 60: "PC7", 61: "VCC", 62: "GND", 63: "PJ0/RXD3", 64: "PJ1/TXD3",
    65: "PJ2", 66: "PJ3", 67: "PJ4", 68: "PJ5", 69: "PJ6", 70: "PG2", 71: "PA7", 72: "PA6", 73: "PA5",
    74: "PA4", 75: "PA3", 76: "PA2", 77: "PA1", 78: "PA0", 79: "PJ7", 80: "VCC", 81: "GND",
    82: "PK7/ADC15", 83: "PK6/ADC14", 84: "PK5/ADC13", 85: "PK4/ADC12", 86: "PK3/ADC11",
    87: "PK2/ADC10", 88: "PK1/ADC9", 89: "PK0/ADC8", 90: "PF7/ADC7", 91: "PF6/ADC6", 92: "PF5/ADC5",
    93: "PF4/ADC4", 94: "PF3/ADC3", 95: "PF2/ADC2", 96: "PF1/ADC1", 97: "PF0/ADC0", 98: "AREF",
    99: "GND", 100: "AVCC",
}


def _classify(name: str) -> tuple:
    if name in ("VCC", "AVCC"):
        return "power", "power_in"
    if name == "GND":
        return "ground", "power_in"
    if name == "RESET":
        return "reset", "input"
    if name.startswith("XTAL"):
        return "xtal", "passive"
    if name == "AREF":
        return "aref", "passive"
    return "gpio", "bidirectional"


ATMEGA2560_PINS = [_pin(n, name, *_classify(name)) for n, name in sorted(_M2560_NAMES.items())]

# ESP32-WROOM-32 module -- Espressif "ESP32-WROOM-32 Datasheet", pin
# definitions table (38 castellated pins; the exposed GND pad 39 is the
# same net as pins 1/15/38 and is not drawn separately). Pins 17-22
# (SD0-SD3/CMD/CLK) are wired to the module's internal SPI flash and must
# not be used, so they are marked "nc" like the real NC pin 32.
ESP32_WROOM_32_PINS = [
    _pin(1, "GND", "ground", "power_in"),
    _pin(2, "3V3", "power", "power_in"),
    _pin(3, "EN", "reset", "input"),
    _pin(4, "SENSOR_VP/IO36", "gpio", "input"),
    _pin(5, "SENSOR_VN/IO39", "gpio", "input"),
    _pin(6, "IO34", "gpio", "input"),
    _pin(7, "IO35", "gpio", "input"),
    _pin(8, "IO32", "gpio", "bidirectional"),
    _pin(9, "IO33", "gpio", "bidirectional"),
    _pin(10, "IO25", "gpio", "bidirectional"),
    _pin(11, "IO26", "gpio", "bidirectional"),
    _pin(12, "IO27", "gpio", "bidirectional"),
    _pin(13, "IO14", "gpio", "bidirectional"),
    _pin(14, "IO12", "gpio", "bidirectional"),
    _pin(15, "GND", "ground", "power_in"),
    _pin(16, "IO13", "gpio", "bidirectional"),
    _pin(17, "SD2/IO9", "nc", "no_connect"),
    _pin(18, "SD3/IO10", "nc", "no_connect"),
    _pin(19, "CMD/IO11", "nc", "no_connect"),
    _pin(20, "CLK/IO6", "nc", "no_connect"),
    _pin(21, "SD0/IO7", "nc", "no_connect"),
    _pin(22, "SD1/IO8", "nc", "no_connect"),
    _pin(23, "IO15", "gpio", "bidirectional"),
    _pin(24, "IO2", "gpio", "bidirectional"),
    _pin(25, "IO0", "gpio", "bidirectional"),
    _pin(26, "IO4", "gpio", "bidirectional"),
    _pin(27, "IO16", "gpio", "bidirectional"),
    _pin(28, "IO17", "gpio", "bidirectional"),
    _pin(29, "IO5", "gpio", "bidirectional"),
    _pin(30, "IO18", "gpio", "bidirectional"),
    _pin(31, "IO19", "gpio", "bidirectional"),
    _pin(32, "NC", "nc", "no_connect"),
    _pin(33, "IO21", "gpio", "bidirectional"),
    _pin(34, "RXD0/IO3", "gpio", "bidirectional"),
    _pin(35, "TXD0/IO1", "gpio", "bidirectional"),
    _pin(36, "IO22", "gpio", "bidirectional"),
    _pin(37, "IO23", "gpio", "bidirectional"),
    _pin(38, "GND", "ground", "power_in"),
]

# Raspberry Pi 5 40-pin GPIO header (same pinout as Pi 2/3/4) -- official
# raspberrypi.com GPIO pinout. Drawn from the HAT side: the Pi SUPPLIES
# 5V and 3V3 on this header, so those pins are "supply_out" (power_out).
_PI_HEADER = {
    1: ("3V3", "+3V3"), 2: ("5V", "+5V"), 3: ("GPIO2/SDA1", None), 4: ("5V", "+5V"),
    5: ("GPIO3/SCL1", None), 6: ("GND", None), 7: ("GPIO4", None), 8: ("GPIO14/TXD", None),
    9: ("GND", None), 10: ("GPIO15/RXD", None), 11: ("GPIO17", None), 12: ("GPIO18/PCM_CLK", None),
    13: ("GPIO27", None), 14: ("GND", None), 15: ("GPIO22", None), 16: ("GPIO23", None),
    17: ("3V3", "+3V3"), 18: ("GPIO24", None), 19: ("GPIO10/MOSI", None), 20: ("GND", None),
    21: ("GPIO9/MISO", None), 22: ("GPIO25", None), 23: ("GPIO11/SCLK", None), 24: ("GPIO8/CE0", None),
    25: ("GND", None), 26: ("GPIO7/CE1", None), 27: ("ID_SD", None), 28: ("ID_SC", None),
    29: ("GPIO5", None), 30: ("GND", None), 31: ("GPIO6", None), 32: ("GPIO12/PWM0", None),
    33: ("GPIO13/PWM1", None), 34: ("GND", None), 35: ("GPIO19/PCM_FS", None), 36: ("GPIO16", None),
    37: ("GPIO26", None), 38: ("GPIO20/PCM_DIN", None), 39: ("GND", None), 40: ("GPIO21/PCM_DOUT", None),
}
RPI_HEADER_PINS = [
    _pin(n, name, "supply_out", "power_out", net) if net else
    _pin(n, name, "ground", "power_in") if name == "GND" else
    _pin(n, name, "gpio", "bidirectional")
    for n, (name, net) in sorted(_PI_HEADER.items())
]

IC_LIBRARY = {
    # lib_id matches KiCad's own official symbol library naming, so a real
    # .kicad_sch using this part lines up with this table if one ever shows up.
    # "supply" = the net its power pins are tied to; "crystal" = value label
    # for the oscillator; "reset_cap" = also put a cap on the reset net
    # (Espressif's recommended RC delay on EN).
    "MCU_Microchip_ATmega:ATmega328P-PU": {"package": "DIP-28", "layout": "dip", "supply": "+5V",
                                           "crystal": "16MHz", "pins": ATMEGA328P_PINS},
    "MCU_Microchip_ATmega:ATmega2560-16A": {"package": "TQFP-100", "layout": "dip", "supply": "+5V",
                                            "crystal": "16MHz", "pins": ATMEGA2560_PINS},
    "RF_Module:ESP32-WROOM-32": {"package": "Module-38", "layout": "dip", "supply": "+3V3",
                                 "reset_cap": True, "pins": ESP32_WROOM_32_PINS},
    "Connector_Generic:Conn_02x20_Odd_Even": {"package": "PinHeader-2x20", "layout": "header",
                                              "supply": "+5V", "pins": RPI_HEADER_PINS},
}

# friendly CLI/lookup names -> real lib_id
IC_ALIASES = {
    "atmega328p": "MCU_Microchip_ATmega:ATmega328P-PU",
    "uno": "MCU_Microchip_ATmega:ATmega328P-PU",
    "atmega2560": "MCU_Microchip_ATmega:ATmega2560-16A",
    "mega2560": "MCU_Microchip_ATmega:ATmega2560-16A",
    "esp32": "RF_Module:ESP32-WROOM-32",
    "esp32-wroom-32": "RF_Module:ESP32-WROOM-32",
    "pi5": "Connector_Generic:Conn_02x20_Odd_Even",
    "rpi5": "Connector_Generic:Conn_02x20_Odd_Even",
    "raspberrypi5": "Connector_Generic:Conn_02x20_Odd_Even",
}


def resolve_ic(name: str) -> str:
    return IC_ALIASES.get(name.lower(), name)


def get_part(lib_id: str) -> dict:
    return IC_LIBRARY[resolve_ic(lib_id)]


PIN_NAME_CHAR_MM = 1.1      # rough width of one 1.27mm-font character
PIN_NAME_MARGIN_MM = 1.5    # gap between the two columns of pin names


def body_half_width(lib_id: str) -> float:
    """Half the IC body width, on the 2.54mm grid, wide enough that the
    left column of pin names and the right column never run into each
    other inside the box (KiCad draws pin names inside the body). A DIP-28
    with short names keeps the classic 5.08mm; a module whose pins are named
    "SENSOR_VP/IO36" gets a wider box instead of overlapping text."""
    entry = IC_LIBRARY[resolve_ic(lib_id)]
    longest = max(len(p["name"]) for p in entry["pins"])
    needed = longest * PIN_NAME_CHAR_MM + PIN_NAME_MARGIN_MM
    return round(max(BODY_HALF_WIDTH_MM, -(-needed // PIN_PITCH_MM) * PIN_PITCH_MM), 3)


def get_pins_with_layout(lib_id: str) -> list:
    """Pin dicts (number/name/function/electrical_type/x/y) for a known
    part, tip coordinates filled in from the part's layout function.
    Coordinates are KiCad LIBRARY coordinates (y up): pin 1 of a DIP has
    the largest y and appears at the top of the sheet."""
    lib_id = resolve_ic(lib_id)
    entry = IC_LIBRARY[lib_id]
    layout = LAYOUTS[entry.get("layout", "dip")](len(entry["pins"]), body_half_width(lib_id))
    return [{**pin, "x": layout[int(pin["number"])][0], "y": layout[int(pin["number"])][1]}
            for pin in entry["pins"]]
