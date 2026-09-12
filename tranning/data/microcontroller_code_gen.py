"""Generates microcontroller code_pairs.json entries (Arduino/ESP32/Pi
Pico/STM32) and appends them to the existing data/code_pairs.json used by
tranning/chats.py's code checkpoint (CODE_OUT_DIR) -- same "extend the
existing model, don't build a new pipeline" choice you made for this task,
unlike the Schmitt-trigger/OPA circuit work which got its own pipeline
because that one needed a from-scratch ERC to validate against.

No compiler is installed in this environment (arduino-cli/platformio/gcc
all absent, confirmed 2026-09-12) so Arduino/ESP32/STM32 (C/C++) snippets
are best-effort: hand-written against well-established, stable APIs (Arduino
core, ESP32 Arduino-core 3.x's ledcAttach()/ledcWrite() -- verified against
espressif/arduino-esp32's own docs, the 2.x ledcSetup()+ledcAttachPin() two-
step API was removed in 3.x -- and ST's HAL naming), plus a cheap brace/paren
balance heuristic (not a real parser, just catches a transcription slip).
Pi Pico is MicroPython (your choice): every snippet actually IS validated,
for free, with Python's own ast.parse() -- a genuine syntax-correctness
check the other three platforms don't get.

STM32 snippets assume the peripheral (htimN/huartN/hi2cN) is already
initialized via CubeMX/HAL_*_Init(), matching how STM32 "write me code for
X" requests are normally scoped in practice (nobody re-asks for the whole
clock-tree/GPIO-init boilerplate every time) -- this keeps replies short
enough to fit the existing character-level GRU's max_len budget instead of
padding every example with ~40 lines of unrelated init code.
"""

import ast
import json
from pathlib import Path

CODE_PAIRS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "code_pairs.json"

_ARDUINO = {
    "digital_io": (
        "點燈與按鈕輸入",
        'const int LED_PIN = 13;\nconst int BUTTON_PIN = 2;\n\n'
        'void setup() {\n  pinMode(LED_PIN, OUTPUT);\n  pinMode(BUTTON_PIN, INPUT_PULLUP);\n}\n\n'
        'void loop() {\n  digitalWrite(LED_PIN, digitalRead(BUTTON_PIN) == LOW ? HIGH : LOW);\n}',
    ),
    "pwm": (
        "PWM 呼吸燈",
        'const int LED_PIN = 9;\n\nvoid setup() {\n  pinMode(LED_PIN, OUTPUT);\n}\n\n'
        'void loop() {\n  for (int duty = 0; duty <= 255; duty++) {\n    analogWrite(LED_PIN, duty);\n    delay(10);\n  }\n}',
    ),
    "uart": (
        "序列埠印出訊息",
        'void setup() {\n  Serial.begin(9600);\n}\n\nvoid loop() {\n  Serial.println("Hello from Arduino");\n  delay(1000);\n}',
    ),
    "i2c": (
        "讀取 I2C 感測器數值",
        '#include <Wire.h>\n\nconst int SENSOR_ADDR = 0x48;\n\n'
        'void setup() {\n  Wire.begin();\n  Serial.begin(9600);\n}\n\n'
        'void loop() {\n  Wire.requestFrom(SENSOR_ADDR, 2);\n  if (Wire.available() >= 2) {\n'
        '    int value = (Wire.read() << 8) | Wire.read();\n    Serial.println(value);\n  }\n  delay(500);\n}',
    ),
}

_ESP32 = {
    "digital_io": (
        "點燈與按鈕輸入",
        'const int LED_PIN = 2;\nconst int BUTTON_PIN = 4;\n\n'
        'void setup() {\n  pinMode(LED_PIN, OUTPUT);\n  pinMode(BUTTON_PIN, INPUT_PULLUP);\n}\n\n'
        'void loop() {\n  digitalWrite(LED_PIN, digitalRead(BUTTON_PIN) == LOW ? HIGH : LOW);\n}',
    ),
    "pwm": (
        "PWM 呼吸燈",
        'const int LED_PIN = 5;\n\nvoid setup() {\n  ledcAttach(LED_PIN, 5000, 8);\n}\n\n'
        'void loop() {\n  for (int duty = 0; duty <= 255; duty++) {\n    ledcWrite(LED_PIN, duty);\n    delay(10);\n  }\n}',
    ),
    "uart": (
        "序列埠印出訊息",
        'void setup() {\n  Serial.begin(115200);\n}\n\nvoid loop() {\n  Serial.println("Hello from ESP32");\n  delay(1000);\n}',
    ),
    "i2c": (
        "讀取 I2C 感測器數值",
        '#include <Wire.h>\n\nconst int SENSOR_ADDR = 0x48;\n\n'
        'void setup() {\n  Wire.begin();\n  Serial.begin(115200);\n}\n\n'
        'void loop() {\n  Wire.requestFrom(SENSOR_ADDR, 2);\n  if (Wire.available() >= 2) {\n'
        '    int value = (Wire.read() << 8) | Wire.read();\n    Serial.println(value);\n  }\n  delay(500);\n}',
    ),
}

_PICO = {
    "digital_io": (
        "點燈與按鈕輸入",
        'from machine import Pin\nimport time\n\n'
        'led = Pin(25, Pin.OUT)\nbutton = Pin(14, Pin.IN, Pin.PULL_UP)\n\n'
        'while True:\n    led.value(0 if button.value() else 1)\n    time.sleep(0.05)',
    ),
    "pwm": (
        "PWM 呼吸燈",
        'from machine import Pin, PWM\nimport time\n\n'
        'pwm = PWM(Pin(15))\npwm.freq(1000)\n\n'
        'while True:\n    for duty in range(0, 65535, 1024):\n        pwm.duty_u16(duty)\n        time.sleep(0.01)',
    ),
    "uart": (
        "序列埠印出訊息",
        'from machine import UART, Pin\nimport time\n\n'
        'uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))\n\n'
        'while True:\n    uart.write("Hello from Pico\\n")\n    time.sleep(1)',
    ),
    "i2c": (
        "讀取 I2C 感測器數值",
        'from machine import I2C, Pin\n\n'
        'i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)\ndata = i2c.readfrom(0x48, 2)\n'
        'value = (data[0] << 8) | data[1]\nprint(value)',
    ),
}

_STM32 = {
    "digital_io": (
        "點燈與按鈕輸入（假設 GPIO 已經初始化好）",
        'if (HAL_GPIO_ReadPin(BUTTON_GPIO_Port, BUTTON_Pin) == GPIO_PIN_RESET) {\n'
        '    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);\n} else {\n'
        '    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);\n}',
    ),
    "pwm": (
        "PWM 呼吸燈（假設 htim2 的 channel 1 已經設定成 PWM）",
        'HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);\n'
        'for (uint16_t duty = 0; duty <= 999; duty++) {\n'
        '    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, duty);\n    HAL_Delay(2);\n}',
    ),
    "uart": (
        "序列埠印出訊息（假設 huart2 已經設定好）",
        'char msg[] = "Hello from STM32\\r\\n";\n'
        'HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);\nHAL_Delay(1000);',
    ),
    "i2c": (
        "讀取 I2C 感測器數值（假設 hi2c1 已經設定好）",
        'uint8_t data[2];\nHAL_I2C_Master_Receive(&hi2c1, (0x48 << 1), data, 2, HAL_MAX_DELAY);\n'
        'int16_t value = (data[0] << 8) | data[1];',
    ),
}

_PROMPT_TEMPLATES = [
    "用 {platform} 寫一個{task}的程式",
    "幫我寫一段 {platform} 的{task}程式",
    "我要一個 {platform} 的{task}程式碼",
]

_PLATFORMS = [
    ("Arduino", _ARDUINO, "c_family"),
    ("ESP32", _ESP32, "c_family"),
    ("樹莓派 Pico", _PICO, "python"),
    ("STM32", _STM32, "c_family"),
]


def _braces_balanced(code: str) -> bool:
    """Cheap heuristic sanity net for the C/C++ snippets (no real compiler
    available, see module docstring) -- catches a transcription slip
    (missing/extra brace or paren), not a claim of syntactic correctness."""
    return code.count("{") == code.count("}") and code.count("(") == code.count(")")


def _validate(code: str, kind: str) -> None:
    if kind == "python":
        ast.parse(code)  # raises SyntaxError if genuinely broken
    else:
        assert _braces_balanced(code), f"unbalanced braces/parens:\n{code}"


def build_pairs() -> list[dict]:
    pairs = []
    for platform, tasks, kind in _PLATFORMS:
        for task_key, (task_label, code) in tasks.items():
            _validate(code, kind)
            for template in _PROMPT_TEMPLATES:
                pairs.append({"prompt": template.format(platform=platform, task=task_label), "reply": code})
    return pairs


def main():
    existing = json.loads(CODE_PAIRS_PATH.read_text(encoding="utf-8"))
    new_pairs = build_pairs()
    new_prompts = {p["prompt"] for p in new_pairs}
    existing = [p for p in existing if p["prompt"] not in new_prompts]  # idempotent re-run
    combined = existing + new_pairs
    CODE_PAIRS_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    longest_reply = max(len(p["reply"]) for p in combined)
    longest_prompt = max(len(p["prompt"]) for p in combined)
    print(f"{len(new_pairs)} new pairs appended -> {len(combined)} total in {CODE_PAIRS_PATH}")
    print(f"longest reply: {longest_reply} chars, longest prompt: {longest_prompt} chars")


if __name__ == "__main__":
    main()
