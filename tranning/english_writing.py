"""english_writing.py — CEFR B2-C2 English writing practice: advanced
grammar rewrites, academic vocabulary/collocation/idiom cloze, and short
essay/email writing prompts, with a RULE-BASED (not model) checker for
grading a user's own submitted answer. Deterministic item bank + rng pick,
NOT a neural model, no checkpoint, nothing to train.

Why an item bank instead of randomly *generating* new English sentences:
calculus_generator.py/word_problem_generator.py/logic_reasoning_generator.py
can all guarantee a randomly-assembled problem's answer is correct because
the underlying domain has a closed-form checker (sympy, arithmetic, rule
evaluation) that verifies arbitrary generated input. There is no such
checker for "is this an arbitrary grammatically correct English sentence" —
composing a brand-new sentence via random slot substitution risks producing
broken English, which is exactly the failure mode this project already
diagnosed in sinco's own seq2seq models (CLAUDE.md to-do #12/#16: a model
generating free-form English without real language understanding produces
disfluent or wrong output). Randomly generating novel C2-level grammar
sentences would just move that same risk into a different (rule-based
instead of neural) generator.

Instead, every grammar/vocabulary/essay item below is a hand-authored,
individually-verified example (correctness checked when writing the file,
not by a generic solver at call time) — the same "item bank + rng.choice"
shape calculus_generator._LIMIT_GENERATORS already uses to pick among 4
hand-written pedagogical limit subtypes. The only thing randomized is
*which* pre-verified item is served on a given call, not the English text
itself; test_english_writing.py independently re-checks every bank item's
own answer against its own checker (mirroring test_calculus_generator.py
re-deriving every answer with sp.diff/sp.integrate/sp.limit) so a typo in
the bank is caught by CI rather than by a user mid-conversation.

Three topics, matching the CEFR C2 coverage decided for this feature:
  - "grammar": advanced structures a B2 learner has usually not automated
    yet — subjunctive mood, inversion, participle clauses, non-restrictive
    relative clauses, cleft/emphatic sentences. Each item is a
    (base sentence, target rewrite, structural marker regex, rule
    explanation) tuple; checking is "does the user's rewrite contain the
    required structural marker", not full grammatical parsing.
  - "vocabulary": academic word list / collocation / idiom cloze — a
    sentence with a blank plus a Traditional-Chinese meaning hint; checking
    is exact-match (case/punctuation-insensitive) against an accepted-answer
    list (a cloze item can have more than one acceptable synonym).
  - "essay": short essay/email writing prompts with a word-count range and
    a required structural element (an advanced linking word, a conditional
    sentence, or formal letter formatting). Grading here is DELIBERATELY
    limited to checklist items a program can verify objectively (word
    count, presence of a required marker) — sinco has no way to judge
    whether an essay's *argument* is actually good, and pretending
    otherwise would be exactly the kind of false confidence this project's
    other generators (see calculus_solver.py's SolveError, rather than
    guessing, when sympy can't find a closed-form answer) deliberately
    avoid. The sample_answer is shown as a reference model response, not as
    "the one true correct answer" the way calculus's answer is.

generate_problem()/format_question()/format_problem() share the exact same
dict shape ({"topic", "topic_zh", "question", "steps", "answer"}, plus
extra topic-specific keys check_answer() needs) that calculus_generator.py
established, so tools.py's existing "出題 -> 解題" quiz plumbing
(_last_quiz_problem, format_question/format_problem) works unchanged; only
the additional grading step ("批改"/"check my answer") is new, wired to
check_answer()/format_feedback() below.

Usage:
    python english_writing.py                          # one random-topic problem
    python english_writing.py --topic grammar
    python english_writing.py --topic vocabulary --count 5
    python english_writing.py --topic essay --seed 42   # reproducible
"""

import argparse
import random
import re

import calculus_generator as cg  # reuse format_question()/format_problem() for the CLI


def _pack(topic: str, topic_zh: str, question: str, steps: list, answer: str, **extra) -> dict:
    problem = {"topic": topic, "topic_zh": topic_zh, "question": question, "steps": steps, "answer": answer}
    problem.update(extra)
    return problem


def _normalize(text: str) -> str:
    """大小寫、頭尾標點/空白都忽略，純字串正規化——不是真正的語言理解，只是
    讓使用者多打一個句點或大小寫不同不會被誤判成答錯。"""
    text = text.strip().lower()
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# grammar：進階文法結構改寫題（CEFR C1-C2）——見上方模組 docstring 說明為何
# 用「題庫＋隨機抽選」而非隨機生成全新句子。
# ---------------------------------------------------------------------------

_SUBJUNCTIVE_ITEMS = [
    dict(
        base="I don't know the answer, so I can't help you.",
        answer="If I knew the answer, I could help you.",
        marker=re.compile(r"\bif i knew\b.*\bcould\b", re.IGNORECASE | re.DOTALL),
        explain="與現在事實相反的假設：If + 主詞 + 過去式動詞, 主詞 + could/would + 原形動詞。",
    ),
    dict(
        base="I didn't study, so I failed the exam.",
        answer="If I had studied, I would have passed the exam.",
        marker=re.compile(r"\bif i had studied\b.*\bwould have\b", re.IGNORECASE | re.DOTALL),
        explain="與過去事實相反的假設：If + 主詞 + had + 過去分詞, 主詞 + would have + 過去分詞。",
    ),
    dict(
        base="I regret not knowing the truth earlier.",
        answer="I wish I had known the truth earlier.",
        marker=re.compile(r"\bi wish i had known\b", re.IGNORECASE),
        explain="wish + 主詞 + had + 過去分詞：表達對過去已發生事情的遺憾。",
    ),
    dict(
        base="It is essential that she attends the meeting.",
        answer="It is essential that she attend the meeting.",
        marker=re.compile(r"\bthat she attend\b", re.IGNORECASE),
        explain="表建議/必要性的假設語氣（mandative subjunctive）：that 子句中的動詞用原形，"
                "不隨主詞變化（she attend，不是 attends）。",
    ),
    dict(
        base="If you don't hurry, you will regret it, but let's not assume that will happen.",
        answer="Were you to hurry, you would not regret it.",
        marker=re.compile(r"\bwere you to hurry\b.*\bwould\b", re.IGNORECASE | re.DOTALL),
        explain="正式倒裝假設語氣：Were + 主詞 + to + 原形動詞, 主詞 + would + 原形動詞"
                "（if 可省略，用倒裝代替）。",
    ),
]

_INVERSION_ITEMS = [
    dict(
        base="I have never seen such a beautiful sunset before.",
        answer="Never have I seen such a beautiful sunset before.",
        marker=re.compile(r"^never have i seen\b", re.IGNORECASE),
        explain="否定副詞 Never 放句首要倒裝：Never + have/has + 主詞 + 過去分詞。",
    ),
    dict(
        base="He not only finished the project early but also impressed the client.",
        answer="Not only did he finish the project early, but he also impressed the client.",
        marker=re.compile(r"^not only did he finish\b", re.IGNORECASE),
        explain="Not only 放句首要倒裝：Not only + did/do/does + 主詞 + 原形動詞, ... but (also) ...。",
    ),
    dict(
        base="If I had known the truth, I would have acted differently.",
        answer="Had I known the truth, I would have acted differently.",
        marker=re.compile(r"^had i known the truth\b", re.IGNORECASE),
        explain="省略 If 的倒裝假設語氣：Had + 主詞 + 過去分詞（等同 If + 主詞 + had + 過去分詞）。",
    ),
    dict(
        base="She rarely complains about her workload.",
        answer="Rarely does she complain about her workload.",
        marker=re.compile(r"^rarely does she complain\b", re.IGNORECASE),
        explain="頻率副詞 Rarely 放句首要倒裝：Rarely + does/do/did + 主詞 + 原形動詞。",
    ),
    dict(
        base="I had hardly arrived home when the phone rang.",
        answer="Hardly had I arrived home when the phone rang.",
        marker=re.compile(r"^hardly had i arrived home\b", re.IGNORECASE),
        explain="Hardly...when 句型的倒裝：Hardly + had + 主詞 + 過去分詞 + when + 過去式子句。",
    ),
]

_PARTICIPLE_ITEMS = [
    dict(
        base="He finished his homework, and then he went out to play.",
        answer="Having finished his homework, he went out to play.",
        marker=re.compile(r"^having finished his homework\b", re.IGNORECASE),
        explain="完成分詞構句 Having + 過去分詞，表示先發生的動作，用來連接兩個有先後關係的子句。",
    ),
    dict(
        base="She was exhausted by the long journey, so she fell asleep immediately.",
        answer="Exhausted by the long journey, she fell asleep immediately.",
        marker=re.compile(r"^exhausted by the long journey\b", re.IGNORECASE),
        explain="過去分詞構句（被動意義）：省略 being/having been，直接用過去分詞開頭。",
    ),
    dict(
        base="Because he did not know what to say, he remained silent.",
        answer="Not knowing what to say, he remained silent.",
        marker=re.compile(r"^not knowing what to say\b", re.IGNORECASE),
        explain="否定分詞構句：Not + 現在分詞，放在分詞前面表示否定。",
    ),
    dict(
        base="As the results were disappointing, the team decided to revise their strategy.",
        answer="The results being disappointing, the team decided to revise their strategy.",
        marker=re.compile(r"^the results being disappointing\b", re.IGNORECASE),
        explain="獨立分詞構句（主詞跟主要子句不同時要保留）：主詞 + being/現在分詞，"
                "用來取代 as/since 引導的原因子句。",
    ),
]

_RELATIVE_ITEMS = [
    dict(
        base="My brother lives in London. He is a doctor.",
        answer="My brother, who lives in London, is a doctor.",
        marker=re.compile(r"my brother, who lives in london,", re.IGNORECASE),
        explain="非限定關係子句（補充說明，用逗號隔開）：先行詞 + , who/which/whose ... , + 主要子句。",
    ),
    dict(
        base="The book was written by a famous author. I borrowed the book from the library.",
        answer="The book, which I borrowed from the library, was written by a famous author.",
        marker=re.compile(r"the book, which i borrowed from the library,", re.IGNORECASE),
        explain="非限定關係子句修飾事物用 which（不能用 that）：先行詞 + , which ... ,。",
    ),
    dict(
        base="The woman's car was stolen. She reported it to the police.",
        answer="The woman, whose car was stolen, reported it to the police.",
        marker=re.compile(r"the woman, whose car was stolen,", re.IGNORECASE),
        explain="所有格關係代名詞 whose：表示「先行詞的...」，連接兩個共用所有格關係的子句。",
    ),
]

_CLEFT_ITEMS = [
    dict(
        base="John broke the window, not anyone else.",
        answer="It was John who broke the window.",
        marker=re.compile(r"^it was john who broke the window\b", re.IGNORECASE),
        explain="分裂句（強調句型）：It is/was + 強調的部分 + who/that + 其餘子句，"
                "用來強調某個特定的人事物。",
    ),
    dict(
        base="I need a rest, more than anything else.",
        answer="What I need is a rest.",
        marker=re.compile(r"^what i need is a rest\b", re.IGNORECASE),
        explain="What 開頭的分裂句：What + 主詞 + 動詞 + is/was + 強調的部分，用來強調受詞或需求。",
    ),
    dict(
        base="The lack of communication caused the project to fail, not anything else.",
        answer="It was the lack of communication that caused the project to fail.",
        marker=re.compile(r"^it was the lack of communication that caused\b", re.IGNORECASE),
        explain="分裂句強調原因：It was + 強調的原因 + that + 其餘子句。",
    ),
]

_GRAMMAR_BANK = {
    "subjunctive": ("假設語氣", _SUBJUNCTIVE_ITEMS),
    "inversion": ("倒裝句", _INVERSION_ITEMS),
    "participle": ("分詞構句", _PARTICIPLE_ITEMS),
    "relative": ("關係子句", _RELATIVE_ITEMS),
    "cleft": ("分裂句（強調句型）", _CLEFT_ITEMS),
}


def _generate_grammar(rng: random.Random, subtype: str | None = None) -> dict:
    if subtype is None or subtype == "random":
        subtype = rng.choice(list(_GRAMMAR_BANK))
    if subtype not in _GRAMMAR_BANK:
        raise ValueError(f"未知文法子類型：{subtype!r}，可用選項：{', '.join(_GRAMMAR_BANK)}, random")
    label, items = _GRAMMAR_BANK[subtype]
    item = rng.choice(items)
    question = f"請用「{label}」改寫下列句子，改寫後意思要相同：\n{item['base']}"
    steps = [item["explain"]]
    return _pack(
        "grammar", f"進階文法結構－{label}（CEFR C1-C2）", question, steps, item["answer"],
        subtype=subtype, marker=item["marker"], base=item["base"],
    )


def _check_grammar(problem: dict, user_text: str) -> dict:
    marker: re.Pattern = problem["marker"]
    normalized = _normalize(user_text)
    correct = bool(marker.search(normalized))
    if correct:
        feedback = f"正確！你的改寫符合「{problem['topic_zh']}」的用法。"
    else:
        feedback = (
            f"還不符合「{problem['topic_zh']}」的句型要求。\n"
            f"文法重點：{problem['steps'][0]}\n"
            f"參考答案：{problem['answer']}"
        )
    return {"correct": correct, "feedback": feedback}


# ---------------------------------------------------------------------------
# vocabulary：學術詞彙／搭配詞／慣用語填空題（CEFR C1-C2）
# ---------------------------------------------------------------------------

_ACADEMIC_VOCAB_ITEMS = [
    dict(sentence="After hours of debate, the committee finally reached a _____.",
         answers=["consensus"], hint="共識、一致的意見"),
    dict(sentence="Her claim was largely _____, lacking any concrete supporting evidence.",
         answers=["speculative"], hint="推測性的、缺乏根據的"),
    dict(sentence="The new policy has had a _____ effect on small businesses in the region.",
         answers=["detrimental"], hint="有害的、不利的"),
    dict(sentence="Researchers must remain _____ when interpreting ambiguous data.",
         answers=["objective", "impartial"], hint="客觀的、不偏頗的"),
    dict(sentence="The report provides a _____ analysis of the economic downturn.",
         answers=["comprehensive", "thorough"], hint="全面的、詳盡的"),
    dict(sentence="It is difficult to _____ the exact cause of the phenomenon.",
         answers=["ascertain", "determine"], hint="確定、查明"),
]

_COLLOCATION_ITEMS = [
    dict(sentence="She had to _____ a difficult decision under time pressure.",
         answers=["make"], hint="表示『做決定』的固定搭配動詞，不是逐字對應中文『做』的 do"),
    dict(sentence="The company decided to _____ a new marketing strategy.",
         answers=["adopt", "implement"], hint="表示『採用／實施』某個策略時常用的動詞"),
    dict(sentence="He managed to _____ a compromise between the two departments.",
         answers=["reach", "strike"], hint="表示『達成』妥協時常用的動詞"),
    dict(sentence="It's important to _____ attention to the small details.",
         answers=["pay"], hint="表示『注意、留意』的固定搭配動詞，不是逐字對應中文『給』的 give"),
    dict(sentence="The manager decided to _____ responsibility for the mistake.",
         answers=["take"], hint="表示『承擔責任』的固定搭配動詞"),
]

_IDIOM_ITEMS = [
    dict(sentence="Don't worry about the small mistake — fixing it won't _____.",
         answers=["break the bank", "cost a fortune"], hint="意思是『花費過多、傾家蕩產』的慣用語"),
    dict(sentence="He tends to _____ under pressure instead of staying calm.",
         answers=["fall apart", "crack", "buckle"], hint="意思是『崩潰、撐不住』的動詞片語"),
    dict(sentence="After the long meeting, we finally managed to _____ and reach an agreement.",
         answers=["see eye to eye"], hint="意思是『意見一致』的慣用語"),
    dict(sentence="I think we should _____ before making such an important decision.",
         answers=["weigh the pros and cons", "think twice"], hint="意思是『三思而後行、仔細考慮利弊』的慣用語"),
    dict(sentence="She always manages to _____ even in the most difficult situations.",
         answers=["keep her cool", "stay calm"], hint="意思是『保持冷靜』的慣用語"),
]

_VOCAB_BANK = {
    "academic": ("學術詞彙", _ACADEMIC_VOCAB_ITEMS),
    "collocation": ("搭配詞", _COLLOCATION_ITEMS),
    "idiom": ("慣用語", _IDIOM_ITEMS),
}


def _generate_vocabulary(rng: random.Random, subtype: str | None = None) -> dict:
    if subtype is None or subtype == "random":
        subtype = rng.choice(list(_VOCAB_BANK))
    if subtype not in _VOCAB_BANK:
        raise ValueError(f"未知詞彙子類型：{subtype!r}，可用選項：{', '.join(_VOCAB_BANK)}, random")
    label, items = _VOCAB_BANK[subtype]
    item = rng.choice(items)
    question = f"請完成填空（提示：{item['hint']}）：\n{item['sentence']}"
    steps = [f"用法說明：{item['hint']}"]
    answer = "／".join(item["answers"])
    return _pack(
        "vocabulary", f"學術詞彙與慣用語－{label}（CEFR C1-C2）", question, steps, answer,
        subtype=subtype, accepted=item["answers"],
    )


def _check_vocabulary(problem: dict, user_text: str) -> dict:
    accepted = {_normalize(a) for a in problem["accepted"]}
    correct = _normalize(user_text) in accepted
    if correct:
        feedback = "正確！"
    else:
        feedback = f"不正確。正確答案：{problem['answer']}\n{problem['steps'][-1]}"
    return {"correct": correct, "feedback": feedback}


# ---------------------------------------------------------------------------
# essay：篇章寫作題（CEFR B2-C2，短文/書信），規則式批改
#
# 批改只能檢查「結構性、可程式判斷」的條件（字數範圍、是否使用指定的轉折語/
# 文法結構/書信格式），不是真正評分文章論述品質好壞——見模組 docstring。
# ---------------------------------------------------------------------------

_ESSAY_ITEMS = [
    dict(
        kind="essay",
        prompt="To what extent do you agree that social media does more harm than good? "
               "Write a paragraph (80-120 words) stating your position with supporting reasons.",
        min_words=80, max_words=120,
        required_markers=[
            re.compile(r"\bhowever\b", re.IGNORECASE), re.compile(r"\bnevertheless\b", re.IGNORECASE),
            re.compile(r"\bfurthermore\b", re.IGNORECASE), re.compile(r"\bmoreover\b", re.IGNORECASE),
            re.compile(r"\bconsequently\b", re.IGNORECASE), re.compile(r"\bin contrast\b", re.IGNORECASE),
            re.compile(r"\bon the other hand\b", re.IGNORECASE),
        ],
        marker_label="至少一個進階轉折/連接詞（如 however / furthermore / consequently / on the other hand）",
        sample_answer=(
            "I largely agree that social media does more harm than good. Firstly, "
            "it often fosters shallow comparisons that damage users' self-esteem, "
            "especially among teenagers who measure their own worth against "
            "carefully curated online images. Furthermore, the addictive design of "
            "these platforms reduces the time people spend on meaningful, "
            "face-to-face relationships and genuine community activities. However, "
            "social media does have some benefits, such as connecting distant "
            "friends and family across time zones and borders. On balance, though, "
            "I believe the psychological costs outweigh these advantages, and users "
            "should actively moderate their daily usage accordingly."
        ),
    ),
    dict(
        kind="essay",
        prompt="Write a short essay (100-150 words) discussing one advantage AND one "
               "disadvantage of remote working, using at least one conditional sentence.",
        min_words=100, max_words=150,
        required_markers=[
            re.compile(r"\bif\b.*\bwould\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bif\b.*\bcould\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bif\b.*\bwill\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bwere\b.*\bwould\b", re.IGNORECASE | re.DOTALL),
        ],
        marker_label="至少一個條件句（if ... would/could/will，或倒裝 were ... would）",
        sample_answer=(
            "Remote working offers a significant advantage in terms of flexibility: "
            "employees can better balance their professional and personal lives "
            "without wasting hours commuting every single day. If companies "
            "embraced this model more widely, workers would likely report higher "
            "job satisfaction and improved overall wellbeing. However, remote "
            "working also has a notable disadvantage — it can lead to social "
            "isolation and weaker team cohesion, since spontaneous face-to-face "
            "interactions rarely happen when everyone works from home. Overall, "
            "organisations should weigh these trade-offs carefully before adopting "
            "a fully remote policy, perhaps opting for a hybrid arrangement instead "
            "of an all-or-nothing approach."
        ),
    ),
    dict(
        kind="email",
        prompt="Write a formal email (60-100 words) to a client apologising for a delayed "
               "shipment and proposing a solution.",
        min_words=60, max_words=100,
        required_markers=[
            re.compile(r"\bdear\b", re.IGNORECASE),
            re.compile(r"\bsincerely\b|\bregards\b|\byours faithfully\b", re.IGNORECASE),
        ],
        marker_label="正式書信格式（開頭 Dear ...，結尾 Sincerely / Regards / Yours faithfully）",
        sample_answer=(
            "Dear Mr. Anderson,\n\n"
            "I am writing to sincerely apologise for the delay in delivering your "
            "recent order. Due to an unexpected shortage of raw materials at our "
            "supplier's warehouse, the shipment has unfortunately been postponed "
            "by five business days. As compensation for this inconvenience, we "
            "would like to offer you a 10% discount on your next purchase with us. "
            "We sincerely appreciate your patience and continued understanding "
            "throughout this matter.\n\n"
            "Sincerely,\nCustomer Service Team"
        ),
    ),
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def _generate_essay(rng: random.Random, subtype: str | None = None) -> dict:
    if subtype not in (None, "random"):
        raise ValueError("essay 主題目前只有 random（單一題庫，不分子類型）")
    item = rng.choice(_ESSAY_ITEMS)
    question = (
        f"{item['prompt']}\n"
        f"（字數：{item['min_words']}-{item['max_words']} 字；需包含：{item['marker_label']}）"
    )
    steps = [
        "寫作檢查重點（規則式檢查，非內容品質評分）：",
        f"1. 字數是否落在 {item['min_words']}-{item['max_words']} 字之間",
        f"2. 是否包含 {item['marker_label']}",
        "3. 是否有清楚的立場/結構（開頭表明立場、中間給理由、結尾總結）——這一項無法由程式判斷，"
        "只能自行對照範文檢查",
    ]
    return _pack(
        "essay", f"篇章寫作－{item['kind']}（CEFR B2-C2）", question, steps, item["sample_answer"],
        min_words=item["min_words"], max_words=item["max_words"],
        required_markers=item["required_markers"], marker_label=item["marker_label"],
    )


def _check_essay(problem: dict, user_text: str) -> dict:
    count = _word_count(user_text)
    in_range = problem["min_words"] <= count <= problem["max_words"]
    has_marker = any(m.search(user_text) for m in problem["required_markers"])

    lines = [
        f"字數：{count}（要求 {problem['min_words']}-{problem['max_words']}）"
        + ("　✔ 符合" if in_range else "　✘ 不符合字數要求"),
        ("✔ 已包含 " if has_marker else "✘ 未偵測到 ") + problem["marker_label"],
    ]
    if not (in_range and has_marker):
        lines.append(f"參考範文：\n{problem['answer']}")
    lines.append("（注意：這只是規則式結構檢查，不是內容/論述品質評分——sinco 沒有能力像真人老師"
                 "一樣評價文章寫得好不好，只能檢查上面列出的客觀項目。）")
    return {"correct": in_range and has_marker, "feedback": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "grammar": _generate_grammar,
    "vocabulary": _generate_vocabulary,
    "essay": _generate_essay,
}

_CHECKERS = {
    "grammar": _check_grammar,
    "vocabulary": _check_vocabulary,
    "essay": _check_essay,
}


def generate_problem(topic: str | None = None, subtype: str | None = None, seed: int | None = None) -> dict:
    """Build one random English-writing practice item.

    topic: "grammar" | "vocabulary" | "essay" | "random"/None (pick one at
    random). subtype: an optional topic-specific sub-selector (e.g.
    "inversion" for grammar, "idiom" for vocabulary) — None/"random" lets
    the topic's own generator pick. seed: pass the same int to get the exact
    same item back (used by tests and --seed on the CLI).
    """
    rng = random.Random(seed)
    if topic in (None, "random"):
        topic = rng.choice(list(GENERATORS))
    if topic not in GENERATORS:
        raise ValueError(f"未知主題：{topic!r}，可用選項：{', '.join(GENERATORS)}, random")
    return GENERATORS[topic](rng, subtype=subtype)


def check_answer(problem: dict, user_text: str) -> dict:
    """Rule-based grading of `user_text` against the given problem (must be
    one this module generated — dispatches on problem["topic"]). Returns
    {"correct": bool, "feedback": str}.
    """
    topic = problem.get("topic")
    if topic not in _CHECKERS:
        raise ValueError(f"不支援批改的主題：{topic!r}")
    return _CHECKERS[topic](problem, user_text)


def format_feedback(result: dict) -> str:
    return "\n".join(["【sinco 英文寫作批改】", "", result["feedback"]])


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random CEFR B2-C2 English writing practice item "
                     "(grammar/vocabulary/essay). Deterministic item bank — no trained model."
    )
    parser.add_argument("--topic", choices=["grammar", "vocabulary", "essay", "random"], default="random")
    parser.add_argument("--subtype", default=None)
    parser.add_argument("--seed", type=int, default=None, help="reproducible output when given")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    for i in range(args.count):
        seed = None if args.seed is None else args.seed + i
        problem = generate_problem(topic=args.topic, subtype=args.subtype, seed=seed)
        print(cg.format_problem(problem))
        print()


if __name__ == "__main__":
    main()
