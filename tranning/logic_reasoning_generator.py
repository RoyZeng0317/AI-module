"""logic_reasoning_generator.py — deterministic Chinese logic-reasoning
problem generator AND solver (spatial tracking / syllogistic deduction /
transitive comparison), inspired by Facebook AI Research's bAbI tasks
(synthetic reasoning benchmarks — task 1 "single supporting fact", task 15
"basic deduction", task 17/19-style comparison). Deterministic rule
evaluation, NOT a neural model, no checkpoint, nothing to train — same
category as calculus_generator.py / word_problem_generator.py.

Why deterministic generation instead of training data: same reasoning as
word_problem_generator.py's docstring — this project already found (see
CLAUDE.md to-do #10) that feeding sinco's small seq2seq/Transformer chat
models example Q&A pairs only teaches them to recite those exact examples,
not to actually reason about a new one. Every answer here comes from
evaluating the generated facts with plain Python control flow, so it is
always correct and generalizes to a problem sinco has never seen before.

bAbI itself deliberately uses made-up/randomized entities so a model can't
shortcut the reasoning task with memorized real-world facts — the deduction
generator below follows the same principle: it only ever asks about "所有的
{category} 都是 {superset}" relationships that are unconditionally true by
construction (the fixed, always-true category table below), never a claim
that could itself be factually wrong (e.g. "所有的鳥都會飛" is a false
premise — penguins exist — which would make the "correct" answer a lie).
_deduction_undetermined() additionally tests the classic "denying the
antecedent" fallacy (knowing X is NOT a member of the category does NOT let
you conclude X is not a member of the superset), which is real logical
rigor, not just trivia recall.

Each generate_*() function returns a dict shaped exactly like
calculus_generator.generate_problem()'s output, so tools.py reuses
calculus_generator.format_question()/format_problem() unchanged.

Usage:
    python logic_reasoning_generator.py                    # one random-topic problem
    python logic_reasoning_generator.py --topic spatial
    python logic_reasoning_generator.py --topic deduction --count 5
    python logic_reasoning_generator.py --topic comparison --seed 42
"""

import argparse
import random

import calculus_generator as cg  # reuse format_question()/format_problem()

_NAMES = ["小明", "小華", "小美", "小強", "阿仁", "淑芬", "志偉", "雅婷"]
_ROOMS = ["廚房", "臥室", "客廳", "書房", "浴室", "花園", "車庫", "陽台"]

# 一律「絕對為真」的分類關係，避免用到有例外/有爭議的現實世界事實
# （例如「所有的鳥都會飛」其實是假命題，企鵝就是反例）
_CATEGORY_SUPERSETS = {
    "動物": ["貓", "狗", "兔子", "老虎", "大象", "馬"],
    "水果": ["蘋果", "香蕉", "橘子", "西瓜", "葡萄", "芒果"],
    "花": ["玫瑰", "向日葵", "百合", "鬱金香", "菊花"],
    "交通工具": ["汽車", "公車", "火車", "腳踏車", "飛機"],
}


def _rand_name(rng: random.Random) -> str:
    return rng.choice(_NAMES)


def _pack(topic: str, topic_zh: str, question: str, steps: list, answer: str) -> dict:
    return {"topic": topic, "topic_zh": topic_zh, "question": question, "steps": steps, "answer": answer}


# ---------------------------------------------------------------------------
# spatial：bAbI task 1 風格，單一／連續位置追蹤
# ---------------------------------------------------------------------------

_MOVE_VERBS = ["走到了", "跑到了", "搬到了", "移動到了"]


def _generate_spatial(rng: random.Random) -> dict:
    name = _rand_name(rng)
    n_moves = rng.choice([1, 2])
    rooms = rng.sample(_ROOMS, n_moves + 1)
    facts = [f"{name}在{rooms[0]}。"]
    for prev, nxt in zip(rooms, rooms[1:]):
        verb = rng.choice(_MOVE_VERBS)
        facts.append(f"{name}從{prev}{verb}{nxt}。")
    question = "".join(facts) + f"請問{name}現在在哪裡？"
    answer = rooms[-1]
    steps = [
        "追蹤最後一次提到的地點（之前的位置都已經被最新的移動取代）：",
        f"{'→'.join(rooms)}，最後停留在「{answer}」",
    ]
    return _pack("spatial", "邏輯推理（空間位置追蹤）", question, steps, answer)


# ---------------------------------------------------------------------------
# deduction：bAbI task 15 風格，簡單三段論
# ---------------------------------------------------------------------------

_DEDUCTION_VALID_TEMPLATES = [
    "所有的{category}都是{superset}。{instance}是{category}。請問{instance}是不是{superset}？",
    "已知{instance}是{category}，而所有的{category}都是{superset}。請問{instance}是不是{superset}？",
    "因為所有的{category}都是{superset}，且{instance}是{category}，所以{instance}是不是{superset}？",
]


def _deduction_valid(rng: random.Random) -> dict:
    superset, members = rng.choice(list(_CATEGORY_SUPERSETS.items()))
    category = rng.choice(members)
    instance = _rand_name(rng)
    question = rng.choice(_DEDUCTION_VALID_TEMPLATES).format(category=category, superset=superset, instance=instance)
    steps = [
        f"大前提：所有的{category}都是{superset}",
        f"小前提：{instance}是{category}",
        f"三段論結論：{instance}一定是{superset}",
    ]
    return _pack("deduction", "邏輯推理（三段論演繹）", question, steps, f"是，{instance}是{superset}")


def _deduction_undetermined(rng: random.Random) -> dict:
    superset, members = rng.choice(list(_CATEGORY_SUPERSETS.items()))
    category = rng.choice(members)
    other_superset, other_members = rng.choice(
        [(k, v) for k, v in _CATEGORY_SUPERSETS.items() if k != superset]
    )
    other_category = rng.choice(other_members)
    instance = _rand_name(rng)
    question = (f"所有的{category}都是{superset}。{instance}不是{category}，而是{other_category}。"
                f"請問{instance}是不是{superset}？")
    steps = [
        f"大前提：所有的{category}都是{superset}（這句話沒有說「只有」{category}才是{superset}）",
        f"已知：{instance}不是{category}，是{other_category}",
        f"從「{instance}不是{category}」無法反推「{instance}不是{superset}」——"
        "這是邏輯上的否定前件謬誤，正確結論是「不一定／無法從已知條件判斷」",
    ]
    return _pack("deduction", "邏輯推理（三段論演繹，避免否定前件謬誤）", question, steps, "不一定（無法從已知條件判斷）")


_DEDUCTION_GENERATORS = [_deduction_valid, _deduction_undetermined]


def _generate_deduction(rng: random.Random) -> dict:
    return rng.choice(_DEDUCTION_GENERATORS)(rng)


# ---------------------------------------------------------------------------
# comparison：bAbI 風格，遞移律比較
# ---------------------------------------------------------------------------

_COMPARISONS = [
    ("高", "矮"), ("重", "輕"), ("大", "小"), ("快", "慢"), ("年紀大", "年紀小"),
]


_COMPARISON_TEMPLATES = [
    "{a}比{b}{more}。{b}比{c}{more}。請問{a}和{c}誰比較{more}？",
    "已知{a}比{b}{more}，且{b}比{c}{more}，請問{a}和{c}相比，誰比較{more}？",
    "{a}比{b}{more}，{b}又比{c}{more}，那麼{a}和{c}相比呢？",
]


def _generate_comparison(rng: random.Random) -> dict:
    a, b, c = rng.sample(_NAMES, 3)
    more, _less = rng.choice(_COMPARISONS)
    question = rng.choice(_COMPARISON_TEMPLATES).format(a=a, b=b, c=c, more=more)
    steps = [
        f"{a}比{b}{more}，{b}比{c}{more}",
        f"比較關係具有遞移性：{a} > {b} > {c}（依「{more}」排序）",
        f"所以{a}比{c}{more}",
    ]
    return _pack("comparison", "邏輯推理（遞移律比較）", question, steps, f"{a}比較{more}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "spatial": _generate_spatial,
    "deduction": _generate_deduction,
    "comparison": _generate_comparison,
}


def generate_problem(topic: str | None = None, seed: int | None = None) -> dict:
    """Build one random logic-reasoning problem + worked solution.

    topic: "spatial" | "deduction" | "comparison" | "random"/None.
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
        description="Generate a random logic-reasoning problem (spatial/deduction/comparison) "
                     "with a worked solution. Deterministic rule evaluation — no trained model."
    )
    parser.add_argument("--topic", choices=["spatial", "deduction", "comparison", "random"], default="random")
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
