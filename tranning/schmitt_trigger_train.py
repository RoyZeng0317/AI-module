"""Schmitt-trigger circuit design -- pretrain + fine-tune orchestration.

Extends the sinco Transformer (transformer_chat.py's GPT, same architecture,
zero new model code -- Rule 06 self-built requirement) to answer "design a
Schmitt trigger circuit" with a real .kicad_sch S-expression file instead of
prose, continuing the generative step to-do #23 deliberately deferred
("完全沒有動生成式模型那一段...之後你要談生成的時候再回來做").

No real Schmitt-trigger dataset exists (2026-09-11, see to_do_list.md #29),
so tranning/data/schmitt_trigger_gen.py hand-designs the topology in Python
(coordinate arithmetic, not typed-by-hand S-expressions) and self-validates
every example through circuit_rule_check.py's ERC before it ever reaches
training -- run `python data/schmitt_trigger_gen.py` first (or via `train`
below, which regenerates it every time so the corpus can never go stale
against the generator).

Same two-stage recipe as transformer_chat.py: pretrain() teaches the model
the .kicad_sch S-expression syntax itself (unsupervised, over the raw
generated corpus), finetune() teaches "given this Chinese design request,
produce this circuit" (supervised, data/schmitt_trigger_pairs.json). Both
stages just call transformer_chat.py's existing, already-tested functions --
this file only supplies domain-specific hyperparameters and paths, and a
design()/check_design() convenience wrapper that runs the output straight
back through circuit_rule_check.py so "did it produce something structurally
sane" is a real automated check, not a human eyeballing S-expression text.

block_size=1024 (not transformer_chat.py's 512 default): measured directly
against the generated corpus (see schmitt_trigger_gen.py) -- the longest
prompt+<sep>+reply+<eos> sequence BPE-encodes to ~750 tokens because
coordinate numbers dominate the text and don't compress the way repeated
words do; 512 would silently truncate every example's tail (a garbled,
un-parseable circuit) before it ever reached the loss.

Attempt history (honest record, see to_do_list.md #29 for the full log):
attempt 1 (60/80 epochs, replies = FULL .kicad_sch text incl. lib_symbols,
block_size=2048) converged to train_loss=0.87 with val loss still falling --
generated output had mismatched parens (208 open / 257 close), i.e. never
learned to close the document. attempt 2 found the CLI's own argparse
defaults hadn't been updated to match train()'s new epoch counts, so it
silently re-ran the same low epoch counts (a bug in this file, not in
transformer_chat.py) -- fixed. attempt 3 (150/400 epochs, same full-file
target) reached train_loss=0.50 but generated output was still unparsable
even under near-greedy decoding (temperature=0.05, top_k=1) -- confirmed via
a direct reply() probe that this was genuine underfitting on a too-large
target sequence, not a sampling-randomness problem (repetition_penalty raised
to 1.8 for that probe did kill an earlier "-0000-0000-..." UUID-loop
degeneration, but the token *order* was still wrong from the very first few
tokens). attempt 4 (this version) addresses the root cause instead of just
training longer: the SFT target is now BODY ONLY (placed symbols/wires/
labels) -- lib_symbols is ~50% of a full file and is byte-for-byte identical
boilerplate across every example, so asking the model to reproduce it from
memory on every request was wasting most of its limited capacity/training
signal on the one part that never varies. wrap_kicad() (schmitt_trigger_gen.py)
re-attaches that fixed preamble programmatically at generation time instead.
n_layer/n_embd raised to 6/192 (from 4/128) now that block_size dropped from
2048 to 1024 leaves headroom in the 8GB budget (attention memory is
O(block_size^2), see Rule 06 / DEFAULT_GPU_MEM_FRACTION in transformer_chat.py).

Honest scope even after attempt 4: 18 pairs over 6 unique hand-designed
circuits (5 op-amp hysteresis-value variants + 1 logic-gate inverter) is a
training-pipeline validation, not a claim that this generalizes to
Schmitt-trigger requests outside those two topologies -- same "scaffold
first, expand data later" status as circuit_diagram_train.py / OCR.py /
road_sign_train.py when they were first written.

Usage:
    python schmitt_trigger_train.py train                  # regenerate corpus, pretrain, finetune
    python schmitt_trigger_train.py design --prompt "幫我設計一個史密特觸發電路，R1 用 10k，R2 用 100k"
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
import transformer_chat as tc  # noqa: E402
from circuit_rule_check import check_schematic  # noqa: E402
from schmitt_trigger_gen import wrap_kicad, PLACEHOLDER_R1, PLACEHOLDER_R2  # noqa: E402

_VALUE_RE = r"([0-9]+(?:\.[0-9]+)?\s*[kKmMuUpP]?)"
_R1_PROMPT_RE = re.compile(r"R1\s*(?:[=是為]|用|設[為成]?)\s*" + _VALUE_RE)
_R2_PROMPT_RE = re.compile(r"R2\s*(?:[=是為]|用|設[為成]?)\s*" + _VALUE_RE)


def fill_values(text: str, prompt: str) -> str:
    """Substitutes schmitt_trigger_gen.py's PLACEHOLDER_R1/PLACEHOLDER_R2
    tokens with the value the user actually typed, parsed straight out of
    their own prompt -- see build_corpus()'s docstring for why the model
    is never trusted to reproduce the number itself. No-op for whichever
    placeholder isn't found in either the prompt or the text (e.g. the
    Schmitt-inverter topology's reply has neither placeholder at all)."""
    r1 = _R1_PROMPT_RE.search(prompt)
    r2 = _R2_PROMPT_RE.search(prompt)
    if r1:
        text = text.replace(PLACEHOLDER_R1, r1.group(1).strip())
    if r2:
        text = text.replace(PLACEHOLDER_R2, r2.group(1).strip())
    return text

HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "data" / "schmitt_trigger_corpus.txt"
PAIRS_PATH = HERE.parent / "data" / "schmitt_trigger_pairs.json"
PRETRAIN_DIR = HERE / "schmitt_pretrain_runs"
FINETUNE_DIR = HERE / "schmitt_chat_runs"

BLOCK_SIZE = 1024
N_LAYER = 6
N_EMBD = 192
N_HEAD = 6
VOCAB_SIZE = 2000


def regenerate_corpus() -> None:
    subprocess.run([sys.executable, str(HERE / "data" / "schmitt_trigger_gen.py")], check=True)


def train(pretrain_epochs: int = 150, finetune_epochs: int = 400, device: str | None = None) -> None:
    regenerate_corpus()

    print("=== pretrain: learning .kicad_sch S-expression syntax ===", flush=True)
    tc.pretrain(
        CORPUS_PATH, out_dir=PRETRAIN_DIR, epochs=pretrain_epochs, batch_size=2,
        block_size=BLOCK_SIZE, n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        dropout=0.1, lr=3e-4, weight_decay=0.01, vocab_size=VOCAB_SIZE,
        val_split=0.15, patience=25, device=device,
    )

    print("=== finetune: 設計要求 -> 電路 S-expression ===", flush=True)
    # 18 examples over 6 unique circuits is small enough that this stage is
    # meant to reach near-memorization (same target as chats.py's GRU on its
    # 28 hand-written pairs, train_loss ~0.0001), not to generalize -- attempt
    # 1 (60 epochs, see to_do_list.md #29) stopped at train_loss=0.87 with val
    # loss still falling, i.e. still underfit, and the generated output came
    # out with mismatched parens. attempt 4 (body-only target, larger model)
    # made it WORSE: a random 15%-of-18 val split put a disproportionate
    # share of the minority "inverter" topology (3 of 18 pairs) into val,
    # val_loss flatlined at 1.30 almost immediately and patience=40 correctly
    # early-stopped against a genuinely diverging (not just noisy) val signal
    # -- but which 2-3 examples land in val is non-deterministic (finetune()
    # calls random.shuffle() unseeded), so the *specific* prompt tested by
    # hand could land in either split from one run to the next. Fix: pass
    # val_data_path=PAIRS_PATH so ALL 18 pairs are both train and the loss
    # used for early stopping -- same "reuse train as val" fallback
    # transformer_chat.pretrain() already uses for a too-small corpus,
    # applied here deliberately because near-total memorization of these 18
    # pairs IS the stated goal at this data scale, not held-out generalization.
    tc.finetune(
        PAIRS_PATH, pretrain_dir=PRETRAIN_DIR, out_dir=FINETUNE_DIR,
        epochs=finetune_epochs, batch_size=2, lr=1e-4, weight_decay=0.01,
        dropout=0.1, val_data_path=PAIRS_PATH, patience=40, device=device,
    )


def design(prompt: str, max_new_tokens: int = 900, temperature: float = 0.2, device: str | None = None) -> str:
    """Returns the model's generated BODY text (placed symbols/wires/labels
    only) -- the fixed lib_symbols preamble is never part of what the model
    has to reproduce (see schmitt_trigger_gen.build_opamp_schmitt() and
    to_do_list.md #29 attempt 4); wrap with wrap_kicad() (check_design()
    already does) to get a real .kicad_sch file. low temperature / narrow
    top_k: with only a few dozen training examples the goal is
    near-memorization (see train()'s comment), so generation should stay
    close to the model's most-confident path rather than sampling broadly.

    fill_values() runs on the raw output before returning: the model only
    ever has to generate PLACEHOLDER_R1/PLACEHOLDER_R2 (build_corpus()'s
    docstring explains why), the real value comes from parsing `prompt`
    itself.
    """
    raw = tc.reply(
        prompt, out_dir=FINETUNE_DIR, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=5, top_p=0.9, repetition_penalty=1.3, device=device,
    )
    return fill_values(raw, prompt)


def check_design(body_text: str, tmp_path: Path) -> list:
    """Wraps the model's generated BODY text with the fixed lib_symbols
    preamble (wrap_kicad()), writes a real .kicad_sch, and runs the
    from-scratch ERC (circuit_rule_check.py) over it -- parse failures count
    as findings too (wrapped as a single "unparsable" finding) instead of
    raising, so a caller can always inspect the result."""
    tmp_path.write_text(wrap_kicad(body_text), encoding="utf-8")
    try:
        return check_schematic(tmp_path)
    except Exception as exc:
        return [{"rule": "unparsable", "severity": "error", "message": str(exc), "point": None}]


def main():
    parser = argparse.ArgumentParser(description="Schmitt-trigger circuit design: pretrain + finetune + generate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="regenerate corpus, pretrain, then finetune")
    p_train.add_argument("--pretrain-epochs", type=int, default=150)
    p_train.add_argument("--finetune-epochs", type=int, default=400)
    p_train.add_argument("--device", default=None)

    p_design = sub.add_parser("design", help="generate a circuit from a design request and rule-check it")
    p_design.add_argument("--prompt", required=True)
    p_design.add_argument("--out", type=Path, default=HERE / "schmitt_design_output.kicad_sch")
    p_design.add_argument("--device", default=None)

    args = parser.parse_args()
    if args.command == "train":
        train(args.pretrain_epochs, args.finetune_epochs, device=args.device)
    elif args.command == "design":
        text = design(args.prompt, device=args.device)
        findings = check_design(text, args.out)
        print(f"generated circuit written to {args.out}")
        if not findings:
            print("circuit_rule_check: no findings")
        else:
            for f in findings:
                print(f'  [{f["severity"]}] {f["rule"]}: {f["message"]}')


if __name__ == "__main__":
    main()
