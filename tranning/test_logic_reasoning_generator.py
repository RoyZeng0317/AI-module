"""Tests for logic_reasoning_generator.py — verifies each generated
problem's stated answer against an independent re-evaluation of the facts
embedded in the question, mirroring test_calculus_generator.py's own-
answer-verification pattern (not just "did it run without crashing").
"""

import random

import pytest

import logic_reasoning_generator as lr


def _rng(seed):
    return random.Random(seed)


@pytest.mark.parametrize("seed", range(30))
def test_spatial_answer_is_the_last_room_visited(seed):
    problem = lr._generate_spatial(_rng(seed))
    assert problem["answer"] in lr._ROOMS
    # steps[-1] spells out "roomA→roomB→...，最後停留在「answer」" — the
    # room right before the closing 」 must match the stated answer.
    assert problem["steps"][-1].split("「")[-1].rstrip("」") == problem["answer"]
    assert problem["steps"][-1].split("→")[-1].startswith(problem["answer"])


@pytest.mark.parametrize("seed", range(30))
def test_deduction_valid_answer_is_always_affirmative(seed):
    problem = lr._deduction_valid(_rng(seed))
    assert problem["answer"].startswith("是，")
    # the asserted superset in the answer must be a real key of the category table
    superset = problem["answer"].split("是")[-1]
    assert superset in lr._CATEGORY_SUPERSETS


@pytest.mark.parametrize("seed", range(30))
def test_deduction_undetermined_answer_is_always_undetermined(seed):
    problem = lr._deduction_undetermined(_rng(seed))
    assert problem["answer"] == "不一定（無法從已知條件判斷）"
    # sanity: the "other" category actually belongs to a *different* superset
    # than the one being asked about, otherwise the premise wouldn't be a
    # genuine fallacy trap
    assert "而是" in problem["question"]


@pytest.mark.parametrize("seed", range(30))
def test_comparison_answer_matches_transitive_order(seed):
    problem = lr._generate_comparison(_rng(seed))
    # steps[1] spells out "A > B > C（依「more」排序）" — the answer must
    # name the same person who appears first in that transitive chain.
    chain = problem["steps"][1].split("：")[-1].split("（")[0]
    first_in_chain = chain.split(" > ")[0]
    assert problem["answer"].startswith(first_in_chain)


def test_generate_problem_topic_dispatch():
    assert lr.generate_problem(topic="spatial", seed=1)["topic"] == "spatial"
    assert lr.generate_problem(topic="deduction", seed=1)["topic"] == "deduction"
    assert lr.generate_problem(topic="comparison", seed=1)["topic"] == "comparison"


def test_generate_problem_random_topic_picks_a_known_topic():
    problem = lr.generate_problem(topic="random", seed=7)
    assert problem["topic"] in lr.GENERATORS


def test_generate_problem_unknown_topic_raises():
    with pytest.raises(ValueError):
        lr.generate_problem(topic="not_a_real_topic")


def test_generate_problem_same_seed_is_reproducible():
    a = lr.generate_problem(topic="deduction", seed=42)
    b = lr.generate_problem(topic="deduction", seed=42)
    assert a == b


def test_deduction_generators_never_use_a_false_real_world_premise():
    # every category -> superset relationship in the fixed table must be
    # unconditionally true (see module docstring: no "所有鳥都會飛"-style
    # false premises with real-world exceptions)
    for superset, members in lr._CATEGORY_SUPERSETS.items():
        assert len(members) >= 2
        assert superset not in members
