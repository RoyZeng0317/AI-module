"""word_problem_generator.py — elementary-level Chinese math word problem
generator AND solver (four arithmetic operations, unit conversion,
ratio/percentage/unit price). Deterministic arithmetic, NOT a neural model,
no checkpoint, nothing to train — same category as calculus_generator.py.

Why a generator instead of training data: this project already tried the
"train the model on Q&A pairs" route for calculus (see CLAUDE.md to-do #10)
and rejected it — sinco's chat models (chats.py's GRU, transformer_chat.py's
Transformer) are pattern-matchers over a small hand-written/scraped corpus;
feeding them word problems as training pairs only teaches them to recite the
exact problems they saw, not to actually compute a new problem's answer. A
word problem's answer instead comes from real Python arithmetic here, so it
is always correct and works on a problem sinco has never seen before, the
same way calculus_generator.py's sympy computation does.

Each generate_*() function returns a dict shaped exactly like
calculus_generator.generate_problem()'s output ({"topic", "topic_zh",
"question", "steps", "answer"}) so tools.py reuses
calculus_generator.format_question()/format_problem() unchanged — one
shared "出題 -> 解題" quiz flow across calculus / math word problems / logic
reasoning.

Usage:
    python word_problem_generator.py                     # one random-topic problem
    python word_problem_generator.py --topic arithmetic
    python word_problem_generator.py --topic ratio --count 5
    python word_problem_generator.py --topic unit_conversion --seed 42
"""

import argparse
import random

import calculus_generator as cg  # reuse format_question()/format_problem()

_NAMES = ["小明", "小華", "小美", "小強", "阿仁", "淑芬", "志偉", "雅婷"]
_OBJECTS = ["蘋果", "餅乾", "貼紙", "鉛筆", "氣球", "糖果", "橘子", "彈珠"]


def _rand_name(rng: random.Random) -> str:
    return rng.choice(_NAMES)


def _rand_object(rng: random.Random) -> str:
    return rng.choice(_OBJECTS)


def _pack(topic: str, topic_zh: str, question: str, steps: list, answer: str) -> dict:
    return {"topic": topic, "topic_zh": topic_zh, "question": question, "steps": steps, "answer": answer}


# ---------------------------------------------------------------------------
# arithmetic：加減乘除應用題
#
# 每個子題型底下有好幾種敘述場景（動詞/角色不同，數字順序與運算邏輯完全一致），
# 隨機挑一種——避免每次都是同一句話的填空版本，讓 sinco 看到的「加法題」不會
# 只有一種固定句型，同時仍然保證答案 100% 正確（跟哪種敘述場景無關，算的都
# 是同一組數字）。
# ---------------------------------------------------------------------------

_ADD_SCENARIOS = [("有", "又買了"), ("有", "又摘了"), ("原本有", "又收到朋友送的"), ("手上有", "又撿到")]
_SUB_SCENARIOS = [("有", "用掉了"), ("有", "吃掉了"), ("原本有", "送給朋友"), ("手上有", "弄丟了")]
_CONTAINERS = ["箱", "袋", "盒", "包"]
_DIV_RECIPIENTS = [("個朋友", "每人"), ("位同學", "每人"), ("個小朋友", "每人")]
_MONEY_HAVE_VERBS = ["身上有", "存了", "帶了"]


def _arith_addition(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    a, b = rng.randint(3, 50), rng.randint(3, 50)
    answer = a + b
    have_verb, gain_verb = rng.choice(_ADD_SCENARIOS)
    question = f"{name}{have_verb} {a} 顆{obj}，{gain_verb} {b} 顆，請問{name}現在一共有幾顆{obj}？"
    steps = [f"「一共」代表把兩次的數量相加：{a} + {b} = {answer}"]
    return _pack("arithmetic", "數學應用題（加法）", question, steps, f"{answer} 顆{obj}")


def _arith_subtraction(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    a = rng.randint(10, 60)
    b = rng.randint(1, a - 1)
    answer = a - b
    have_verb, lose_verb = rng.choice(_SUB_SCENARIOS)
    question = f"{name}{have_verb} {a} 顆{obj}，{lose_verb} {b} 顆，請問{name}還剩下幾顆{obj}？"
    steps = [f"「還剩下」代表從原有數量中減去用掉的部分：{a} - {b} = {answer}"]
    return _pack("arithmetic", "數學應用題（減法）", question, steps, f"{answer} 顆{obj}")


def _arith_multiplication(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    boxes = rng.randint(2, 9)
    per_box = rng.randint(2, 12)
    answer = boxes * per_box
    container = rng.choice(_CONTAINERS)
    question = f"{name}買了 {boxes} {container}{obj}，每{container}有 {per_box} 顆，請問{name}一共買了幾顆{obj}？"
    steps = [f"「每{container}都一樣多」代表用乘法：{container}數 × 每{container}數量 = {boxes} × {per_box} = {answer}"]
    return _pack("arithmetic", "數學應用題（乘法）", question, steps, f"{answer} 顆{obj}")


def _arith_division(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    people = rng.randint(2, 8)
    per_person = rng.randint(2, 12)
    total = people * per_person
    recipient, unit_each = rng.choice(_DIV_RECIPIENTS)
    question = f"{name}有 {total} 顆{obj}，要平分給 {people} {recipient}，請問{unit_each}可以分到幾顆{obj}？"
    steps = [f"「平分」代表用除法：總數 ÷ 人數 = {total} ÷ {people} = {per_person}"]
    return _pack("arithmetic", "數學應用題（除法）", question, steps, f"{per_person} 顆{obj}")


def _arith_two_step(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    money = rng.randint(50, 200)
    n = rng.randint(2, 8)
    price = rng.randint(2, money // n)
    cost = n * price
    answer = money - cost
    have_money_verb = rng.choice(_MONEY_HAVE_VERBS)
    question = (f"{name}{have_money_verb} {money} 元，買了 {n} 個{obj}，每個 {price} 元，"
                f"請問{name}還剩下多少元？")
    steps = [
        f"先算出買{obj}總共花了多少錢：{n} × {price} = {cost} 元",
        f"再從原本的錢裡扣掉花費：{money} - {cost} = {answer} 元",
    ]
    return _pack("arithmetic", "數學應用題（兩步驟：先乘後減）", question, steps, f"{answer} 元")


_ARITHMETIC_GENERATORS = [
    _arith_addition, _arith_subtraction, _arith_multiplication,
    _arith_division, _arith_two_step,
]


def _generate_arithmetic(rng: random.Random) -> dict:
    return rng.choice(_ARITHMETIC_GENERATORS)(rng)


# ---------------------------------------------------------------------------
# unit_conversion：常見單位換算（刻意只挑倍數關係固定的單位對，除法方向也保證整除）
# ---------------------------------------------------------------------------

_UNIT_CONVERSIONS = [
    ("公里", "公尺", 1000), ("公尺", "公分", 100), ("公斤", "公克", 1000),
    ("小時", "分鐘", 60), ("分鐘", "秒", 60), ("公升", "毫升", 1000),
]


_UNIT_QUESTION_TEMPLATES = ["{value} {from_unit}等於幾{to_unit}？", "{value} {from_unit}換算成{to_unit}是多少？"]


def _generate_unit_conversion(rng: random.Random) -> dict:
    big_unit, small_unit, factor = rng.choice(_UNIT_CONVERSIONS)
    to_small = rng.random() < 0.5
    template = rng.choice(_UNIT_QUESTION_TEMPLATES)
    if to_small:
        value = rng.randint(1, 20)
        answer = value * factor
        question = template.format(value=value, from_unit=big_unit, to_unit=small_unit)
        steps = [f"1 {big_unit} = {factor} {small_unit}，所以 {value} {big_unit} = {value} × {factor} = {answer} {small_unit}"]
        answer_str = f"{answer} {small_unit}"
    else:
        multiple = rng.randint(1, 20)
        value = multiple * factor
        answer = multiple
        question = template.format(value=value, from_unit=small_unit, to_unit=big_unit)
        steps = [f"1 {big_unit} = {factor} {small_unit}，所以 {value} {small_unit} = {value} ÷ {factor} = {answer} {big_unit}"]
        answer_str = f"{answer} {big_unit}"
    return _pack("unit_conversion", "數學應用題（單位換算）", question, steps, answer_str)


# ---------------------------------------------------------------------------
# ratio：比例／百分比／單價（base 一律取 20 的倍數，保證常見百分比整除不出小數）
# ---------------------------------------------------------------------------

_DISCOUNT_OBJ_PREFIXES = ["一件", "一個", "一份"]
_PERCENTAGE_TEMPLATES = ["{base} 的 {percent}% 是多少？", "{base} 乘以 {percent}% 等於多少？", "求 {base} 的 {percent}%？"]
_UNIT_PRICE_TEMPLATES = [
    "{name}花了 {total} 元買了 {n} 個{obj}，平均每個多少元？",
    "{name}用 {total} 元買了 {n} 個{obj}，每個多少錢？",
]


def _ratio_discount(rng: random.Random) -> dict:
    obj = _rand_object(rng)
    unit_price = rng.randint(2, 20) * 10  # 十位整數，打折後仍是整數
    discount = rng.randint(5, 9)  # X 折 = X/10 原價
    answer = unit_price * discount // 10
    prefix = rng.choice(_DISCOUNT_OBJ_PREFIXES)
    question = f"{prefix}{obj}原價 {unit_price} 元，打 {discount} 折後賣多少元？"
    steps = [f"「打 {discount} 折」代表付原價的 {discount}/10：{unit_price} × {discount} ÷ 10 = {answer} 元"]
    return _pack("ratio", "數學應用題（折扣）", question, steps, f"{answer} 元")


def _ratio_percentage(rng: random.Random) -> dict:
    base = rng.randint(1, 10) * 20  # 20 的倍數，10/20/25/50/75% 都能整除不出小數
    percent = rng.choice([10, 20, 25, 50, 75])
    answer = base * percent // 100
    question = rng.choice(_PERCENTAGE_TEMPLATES).format(base=base, percent=percent)
    steps = [f"「百分之 {percent}」代表 {percent}/100：{base} × {percent} ÷ 100 = {answer}"]
    return _pack("ratio", "數學應用題（百分比）", question, steps, str(answer))


def _ratio_unit_price(rng: random.Random) -> dict:
    name, obj = _rand_name(rng), _rand_object(rng)
    n = rng.randint(2, 10)
    unit_price = rng.randint(3, 20)
    total = n * unit_price
    question = rng.choice(_UNIT_PRICE_TEMPLATES).format(name=name, total=total, n=n, obj=obj)
    steps = [f"「平均每個」代表總價 ÷ 數量：{total} ÷ {n} = {unit_price} 元"]
    return _pack("ratio", "數學應用題（單價）", question, steps, f"{unit_price} 元")


def _ratio_allocation(rng: random.Random) -> dict:
    obj = _rand_object(rng)
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    while a == b:
        b = rng.randint(1, 5)
    unit = rng.randint(2, 10)
    total = (a + b) * unit
    share_a, share_b = a * unit, b * unit
    question = f"{total} 顆{obj}按照 {a}:{b} 的比例分給甲、乙兩人，甲、乙各分到幾顆？"
    steps = [
        f"比例 {a}:{b} 一共分成 {a + b} 份，每份 = {total} ÷ {a + b} = {unit} 顆",
        f"甲分到 {a} 份：{a} × {unit} = {share_a} 顆；乙分到 {b} 份：{b} × {unit} = {share_b} 顆",
    ]
    return _pack("ratio", "數學應用題（比例分配）", question, steps, f"甲 {share_a} 顆、乙 {share_b} 顆")


_RATIO_GENERATORS = [_ratio_discount, _ratio_percentage, _ratio_unit_price, _ratio_allocation]


def _generate_ratio(rng: random.Random) -> dict:
    return rng.choice(_RATIO_GENERATORS)(rng)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "arithmetic": _generate_arithmetic,
    "unit_conversion": _generate_unit_conversion,
    "ratio": _generate_ratio,
}


def generate_problem(topic: str | None = None, seed: int | None = None) -> dict:
    """Build one random elementary math word problem + worked solution.

    topic: "arithmetic" | "unit_conversion" | "ratio" | "random"/None.
    seed: pass the same int to get the exact same problem back (used by
    tests and by --seed on the CLI); None means a fresh random problem.
    """
    rng = random.Random(seed)
    if topic in (None, "random"):
        topic = rng.choice(list(GENERATORS))
    if topic not in GENERATORS:
        raise ValueError(f"未知主題：{topic!r}，可用選項：{', '.join(GENERATORS)}, random")
    return GENERATORS[topic](rng)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random elementary math word problem (arithmetic/unit_conversion/"
                     "ratio) with a worked solution. Pure Python arithmetic — no trained model."
    )
    parser.add_argument("--topic", choices=["arithmetic", "unit_conversion", "ratio", "random"], default="random")
    parser.add_argument("--seed", type=int, default=None, help="reproducible output when given")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    for i in range(args.count):
        seed = None if args.seed is None else args.seed + i
        problem = generate_problem(topic=args.topic, seed=seed)
        print(cg.format_problem(problem))
        print()


if __name__ == "__main__":
    main()
