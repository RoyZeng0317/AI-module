"""Prepare isolated replay data and report tokenizer coverage; never train."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tranning'))
from bpe_tokenizer import BPETokenizer, UNK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--replay', type=Path, default=ROOT / 'data/pairs.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/word_sense/prepared')
    args = parser.parse_args()
    base = args.base.resolve()
    config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    if config.get('context_format') not in (None, 'legacy'):
        raise ValueError('Legacy finetune preparation requires legacy GPT base; use dialogue_v1 trainer for that format')
    tokenizer = BPETokenizer.load(base)
    limit = config['block_size'] + 1
    data_dir = ROOT / 'data/word_sense'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    replay = read(args.replay)
    if not replay or not all(isinstance(p.get('prompt'), str) and isinstance(p.get('reply'), str) for p in replay):
        raise ValueError('Replay must be nonempty prompt/reply pairs')
    groups = {split: read(data_dir / f'sft_{split}.json') for split in ('train', 'val', 'test')}
    heldout = {p['prompt'] for split in ('val', 'test') for p in groups[split]}
    replay = [dict(prompt=p['prompt'], reply=p['reply']) for p in replay if p['prompt'] not in heldout]
    groups['replay'] = replay
    report = dict(base=str(base), block_size=config['block_size'], groups={})
    for name, pairs in groups.items():
        counts = dict(total=len(pairs), no_unk=0, untruncated=0, effective=0, reply_fully_removed=0)
        for pair in pairs:
            prompt, reply = tokenizer.encode(pair['prompt']), tokenizer.encode(pair['reply'])
            clean = UNK not in prompt + reply
            fits = len(prompt) + len(reply) + 2 <= limit
            counts['no_unk'] += clean
            counts['untruncated'] += fits
            counts['effective'] += clean and fits
            counts['reply_fully_removed'] += len(prompt) + 1 >= limit
        report['groups'][name] = counts
    # Strict gate for the new tiny fixtures; preserve original replay verbatim.
    valid = all(report['groups'][s]['effective'] == report['groups'][s]['total'] for s in ('train', 'val', 'test'))
    out = args.out_dir.resolve()
    if not out.is_relative_to(ROOT) or out == data_dir:
        raise ValueError('Use a separate output directory inside project')
    out.mkdir(parents=True, exist_ok=False)
    report['ready'] = valid
    (out / 'coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not valid:
        raise ValueError(f'UNK/truncation found; see {out / "coverage.json"}; no training file written')
    for filename, pairs in [('train_replay.json', replay + groups['train']), ('val.json', groups['val'])]:
        (out / filename).write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
