"""Tests for english_writing.py — CEFR grammar/vocabulary/essay generator +
rule-based checker. Verifies (a) every hand-authored bank item's own answer
passes its own checker (an item whose own correct answer fails its own
regex/accepted-list would be a real bug in the bank itself, not a corpus
typo, the same "verify the generator's own answer independently" spirit as
test_calculus_generator.py re-deriving every answer with
sp.diff/sp.integrate/sp.limit), and (b) generate_problem()/check_answer()'s
public contract (question never leaks the answer, wrong answers are
correctly rejected, same-seed reproducibility, unsupported topics raise).
"""

import random

import english_writing as ew


# ---------------------------------------------------------------------------
# generate_problem() contract
# ---------------------------------------------------------------------------

def test_generate_problem_same_seed_is_reproducible():
    a = ew.generate_problem(seed=7)
    b = ew.generate_problem(seed=7)
    assert a["question"] == b["question"]
    assert a["answer"] == b["answer"]


def test_generate_problem_unknown_topic_raises():
    try:
        ew.generate_problem(topic="not-a-real-topic", seed=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generate_problem_random_topic_picks_from_generators():
    for seed in range(10):
        problem = ew.generate_problem(seed=seed)
        assert problem["topic"] in ew.GENERATORS


def test_question_never_leaks_answer_or_steps():
    for topic in ew.GENERATORS:
        for seed in range(5):
            problem = ew.generate_problem(topic=topic, seed=seed)
            question_only = ew.cg.format_question(problem, heading="test")
            assert problem["answer"] not in question_only
            for step in problem["steps"]:
                assert step not in question_only


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------

def test_every_grammar_item_own_answer_passes_its_own_checker():
    for subtype, (label, items) in ew._GRAMMAR_BANK.items():
        for item in items:
            problem = ew._pack(
                "grammar", f"進階文法結構－{label}", "q", [item["explain"]], item["answer"],
                subtype=subtype, marker=item["marker"], base=item["base"],
            )
            result = ew.check_answer(problem, item["answer"])
            assert result["correct"], f"{subtype} 題庫項目自己的答案沒通過自己的 checker: {item['answer']}"


def test_grammar_checker_rejects_unmodified_base_sentence():
    problem = ew.generate_problem(topic="grammar", seed=1)
    result = ew.check_answer(problem, problem["base"])
    assert not result["correct"]


def test_grammar_subtype_selection_is_honored():
    for subtype in ew._GRAMMAR_BANK:
        problem = ew._generate_grammar(random.Random(0), subtype=subtype)
        assert problem["subtype"] == subtype


def test_grammar_unknown_subtype_raises():
    try:
        ew._generate_grammar(random.Random(0), subtype="not-a-real-subtype")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

def test_every_vocabulary_item_own_answers_pass_their_own_checker():
    for subtype, (label, items) in ew._VOCAB_BANK.items():
        for item in items:
            problem = ew._pack(
                "vocabulary", f"學術詞彙與慣用語－{label}", "q", [f"用法說明：{item['hint']}"],
                "／".join(item["answers"]), subtype=subtype, accepted=item["answers"],
            )
            for ans in item["answers"]:
                result = ew.check_answer(problem, ans)
                assert result["correct"], f"{subtype} 題庫項目答案 {ans!r} 沒通過自己的 checker"


def test_vocabulary_checker_is_case_and_punctuation_insensitive():
    problem = ew.generate_problem(topic="vocabulary", seed=2)
    correct_answer = problem["accepted"][0]
    result = ew.check_answer(problem, f"  {correct_answer.upper()}.  ")
    assert result["correct"]


def test_vocabulary_checker_rejects_wrong_word():
    problem = ew.generate_problem(topic="vocabulary", seed=2)
    result = ew.check_answer(problem, "definitely-not-the-right-word")
    assert not result["correct"]


def test_vocabulary_unknown_subtype_raises():
    try:
        ew._generate_vocabulary(random.Random(0), subtype="not-a-real-subtype")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# essay
# ---------------------------------------------------------------------------

def test_every_sample_answer_passes_its_own_word_count_and_marker_checks():
    for item in ew._ESSAY_ITEMS:
        problem = ew._pack(
            "essay", "篇章寫作", item["prompt"], ["s"], item["sample_answer"],
            min_words=item["min_words"], max_words=item["max_words"],
            required_markers=item["required_markers"], marker_label=item["marker_label"],
        )
        result = ew.check_answer(problem, item["sample_answer"])
        assert result["correct"], f"樣本範文自己沒通過自己的字數/標記檢查: {item['prompt'][:30]}..."


def test_essay_checker_rejects_too_short_answer():
    problem = ew.generate_problem(topic="essay", seed=3)
    result = ew.check_answer(problem, "Too short.")
    assert not result["correct"]


def test_essay_checker_rejects_missing_required_marker():
    problem = ew.generate_problem(topic="essay", seed=4)
    # pad with plain filler words (no linking word / conditional / Dear-Sincerely)
    # up to the item's max word count, so this fails ONLY the marker check.
    filler = " ".join(["word"] * problem["max_words"])
    result = ew.check_answer(problem, filler)
    assert not result["correct"]


def test_essay_word_count_counts_alphabetic_tokens_only():
    assert ew._word_count("Hello, world! This has six words.") == 6


def test_essay_generate_rejects_non_random_subtype():
    try:
        ew._generate_essay(random.Random(0), subtype="not-a-real-subtype")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# check_answer() dispatch / format_feedback()
# ---------------------------------------------------------------------------

def test_check_answer_unsupported_topic_raises():
    try:
        ew.check_answer({"topic": "unknown"}, "text")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_format_feedback_includes_the_feedback_text():
    problem = ew.generate_problem(topic="vocabulary", seed=5)
    result = ew.check_answer(problem, problem["accepted"][0])
    rendered = ew.format_feedback(result)
    assert result["feedback"] in rendered
    assert "批改" in rendered
