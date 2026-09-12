"""Tests for word_problem_generator.py — every generated problem's stated
answer is independently recomputed here from the numbers embedded in the
question text (not just "did it run without crashing"), mirroring
test_calculus_generator.py's own-answer-verification pattern.
"""

import random
import re

import pytest

import word_problem_generator as wp


def _rng(seed):
    return random.Random(seed)


@pytest.mark.parametrize("seed", range(30))
def test_arithmetic_answer_matches_recomputation(seed):
    problem = wp._generate_arithmetic(_rng(seed))
    nums = [int(n) for n in re.findall(r"\d+", problem["question"])]
    answer_num = int(re.search(r"\d+", problem["answer"]).group())

    if problem["topic_zh"].endswith("（加法）"):
        assert answer_num == nums[0] + nums[1]
    elif problem["topic_zh"].endswith("（減法）"):
        assert answer_num == nums[0] - nums[1]
        assert answer_num >= 0
    elif problem["topic_zh"].endswith("（乘法）"):
        assert answer_num == nums[0] * nums[1]
    elif problem["topic_zh"].endswith("（除法）"):
        total, people = nums[0], nums[1]
        assert total % people == 0
        assert answer_num == total // people
    elif problem["topic_zh"].endswith("兩步驟：先乘後減）"):
        money, n, price = nums[0], nums[1], nums[2]
        assert answer_num == money - n * price
        assert answer_num >= 0
    else:
        pytest.fail(f"unexpected arithmetic subtype: {problem['topic_zh']}")


@pytest.mark.parametrize("seed", range(30))
def test_unit_conversion_answer_matches_recomputation(seed):
    problem = wp._generate_unit_conversion(_rng(seed))
    nums = [int(n) for n in re.findall(r"\d+", problem["question"])]
    value = nums[0]
    answer_num = int(re.search(r"\d+", problem["answer"]).group())

    matched = False
    for big_unit, small_unit, factor in wp._UNIT_CONVERSIONS:
        # question wording varies (see _UNIT_QUESTION_TEMPLATES), but both
        # units of the pair being converted always appear somewhere in it.
        has_big, has_small = big_unit in problem["question"], small_unit in problem["question"]
        if not (has_big and has_small):
            continue
        if problem["question"].startswith(f"{value} {big_unit}"):
            assert answer_num == value * factor
            matched = True
            break
        if problem["question"].startswith(f"{value} {small_unit}"):
            assert value % factor == 0
            assert answer_num == value // factor
            matched = True
            break
    assert matched, f"couldn't match generated question to a known unit pair: {problem['question']}"


@pytest.mark.parametrize("seed", range(30))
def test_ratio_answer_matches_recomputation(seed):
    problem = wp._generate_ratio(_rng(seed))
    nums = [int(n) for n in re.findall(r"\d+", problem["question"])]

    if problem["topic_zh"].endswith("（折扣）"):
        unit_price, discount = nums[0], nums[1]
        assert unit_price % 10 == 0
        answer_num = int(re.search(r"\d+", problem["answer"]).group())
        assert answer_num == unit_price * discount // 10
    elif problem["topic_zh"].endswith("（百分比）"):
        base, percent = nums[0], nums[1]
        assert (base * percent) % 100 == 0, "percentage must divide evenly, no rounding"
        answer_num = int(problem["answer"])
        assert answer_num == base * percent // 100
    elif problem["topic_zh"].endswith("（單價）"):
        total, n = nums[0], nums[1]
        assert total % n == 0
        answer_num = int(re.search(r"\d+", problem["answer"]).group())
        assert answer_num == total // n
    elif problem["topic_zh"].endswith("（比例分配）"):
        total, a, b = nums[0], nums[1], nums[2]
        assert total % (a + b) == 0
        unit = total // (a + b)
        shares = [int(n) for n in re.findall(r"\d+", problem["answer"])]
        assert shares == [a * unit, b * unit]
    else:
        pytest.fail(f"unexpected ratio subtype: {problem['topic_zh']}")


def test_generate_problem_topic_dispatch():
    assert wp.generate_problem(topic="arithmetic", seed=1)["topic"] == "arithmetic"
    assert wp.generate_problem(topic="unit_conversion", seed=1)["topic"] == "unit_conversion"
    assert wp.generate_problem(topic="ratio", seed=1)["topic"] == "ratio"


def test_generate_problem_random_topic_picks_a_known_topic():
    problem = wp.generate_problem(topic="random", seed=7)
    assert problem["topic"] in wp.GENERATORS


def test_generate_problem_unknown_topic_raises():
    with pytest.raises(ValueError):
        wp.generate_problem(topic="not_a_real_topic")


def test_generate_problem_same_seed_is_reproducible():
    a = wp.generate_problem(topic="arithmetic", seed=42)
    b = wp.generate_problem(topic="arithmetic", seed=42)
    assert a == b
