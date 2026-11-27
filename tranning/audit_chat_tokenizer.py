"""Measure UNK and supervised-answer truncation without training a model."""
import argparse, json
from pathlib import Path

from bpe_tokenizer import BPETokenizer, UNK

ROOT = Path(__file__).resolve().parents[1]


def audit_rows(rows, tokenizer, block_size):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    tokens = unknown = truncated = empty_targets = 0
    issues = []
    longest = 0
    characters = set()
    for index, row in enumerate(rows):
        characters.update(row['prompt'] + row['reply'])
        prompt, answer = tokenizer.encode(row['prompt']), tokenizer.encode(row['reply'])
        length = len(prompt) + len(answer) + 2  # SEP and EOS
        missing = (prompt + answer).count(UNK)
        retained = min(len(answer) + 1, max(0, block_size - len(prompt)))
        tokens += len(prompt) + len(answer)
        unknown += missing
        longest = max(longest, length)
        truncated += length > block_size + 1
        empty_targets += retained == 0
        if missing or length > block_size + 1:
            issues.append(dict(index=index, prompt=row['prompt'], tokens=length,
                               unknown_tokens=missing, retained_answer_targets=retained))
    return dict(rows=len(rows), tokens=tokens, unknown_tokens=unknown,
                unknown_ratio=unknown / max(tokens, 1), longest_sequence=longest,
                truncated_rows=truncated, empty_answer_targets=empty_targets,
                unknown_characters=sorted(char for char in characters if tokenizer.encode(char) == [UNK]),
                issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.data_dir, args.tokenizer_dir, args.output):
        if not path.resolve().is_relative_to(ROOT):
            parser.error('All paths must stay inside this project')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing report')
    tokenizer = BPETokenizer.load(args.tokenizer_dir)
    config = json.loads((args.tokenizer_dir / 'config.json').read_text(encoding='utf-8'))
    report = dict(tokenizer_dir=str(args.tokenizer_dir), block_size=config['block_size'])
    for split in ('train', 'val'):
        rows = json.loads((args.data_dir / f'{split}.json').read_text(encoding='utf-8'))
        report[split] = audit_rows(rows, tokenizer, config['block_size'])
    report['meaning'] = 'Format coverage only; not a measure of comprehension or reply quality.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({split: {key: value for key, value in report[split].items() if key != 'issues'}
                      for split in ('train', 'val')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
