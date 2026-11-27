"""Tests for ic_library.py: DIP pin-numbering geometry and the atmega328p
pinout table, checked against the public datasheet numbers by hand."""

from ic_library import dip_pin_layout, get_pins_with_layout, resolve_ic


def test_dip_pin_layout_rejects_odd_pin_count():
    try:
        dip_pin_layout(27)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dip_pin_layout_pin1_top_left_pin_n_top_right():
    layout = dip_pin_layout(28)
    # pin 1 (top-left) and pin 28 (top-right) sit on the same row (same y),
    # on opposite sides (mirrored x) -- standard DIP counter-clockwise numbering.
    assert layout[1][1] == layout[28][1]
    assert layout[1][0] == -layout[28][0]
    # pin 14 (bottom-left) and pin 15 (bottom-right) are the other shared row.
    assert layout[14][1] == layout[15][1]
    assert layout[14][0] == -layout[15][0]
    # left column pins are strictly decreasing in y going down the package.
    assert layout[1][1] > layout[2][1] > layout[14][1]


def test_resolve_ic_alias_and_passthrough():
    assert resolve_ic("atmega328p") == "MCU_Microchip_ATmega:ATmega328P-PU"
    assert resolve_ic("ATMEGA328P") == "MCU_Microchip_ATmega:ATmega328P-PU"
    assert resolve_ic("MCU_Microchip_ATmega:ATmega328P-PU") == "MCU_Microchip_ATmega:ATmega328P-PU"


def test_atmega328p_pin_count_and_key_pins():
    pins = get_pins_with_layout("atmega328p")
    assert len(pins) == 28
    by_number = {p["number"]: p for p in pins}
    assert by_number["7"]["name"] == "VCC" and by_number["7"]["function"] == "power"
    assert by_number["20"]["name"] == "AVCC" and by_number["20"]["function"] == "power"
    assert by_number["8"]["function"] == "ground" and by_number["22"]["function"] == "ground"
    assert by_number["1"]["function"] == "reset"
    assert by_number["9"]["function"] == "xtal" and by_number["10"]["function"] == "xtal"
    assert by_number["21"]["function"] == "aref"
    gpio_count = sum(1 for p in pins if p["function"] == "gpio")
    assert gpio_count == 20  # 28 total - 2 power - 2 gnd - 1 reset - 2 xtal - 1 aref


def test_atmega328p_no_duplicate_or_missing_pin_numbers():
    pins = get_pins_with_layout("atmega328p")
    numbers = sorted(int(p["number"]) for p in pins)
    assert numbers == list(range(1, 29))
