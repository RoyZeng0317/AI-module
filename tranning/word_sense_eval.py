"""Train-only prototype classification and same-term, same-sense retrieval.

These synthetic fixtures verify an evaluation workflow, not general language quality.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[1] / 'data/word_sense'
AMBIGUOUS = 'insufficient_context'


def load_split(name, data_dir=DATA):
    if name not in {'train', 'val', 'test'}:
        raise ValueError('split must be train, val or test')
    return json.loads((Path(data_dir) / f'{name}.json').read_text(encoding='utf-8'))


def classification_metrics(expected, predicted, labels):
    if len(expected) != len(predicted) or not expected:
        raise ValueError('expected and predicted must be nonempty and equally sized')
    if (set(expected) | set(predicted)) - set(labels):
        raise ValueError('unknown label')
    scores = {}
    for label in labels:
        tp = sum(a == b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        scores[label] = dict(precision=precision, recall=recall,
                             f1=2 * tp / max(2 * tp + fp + fn, 1), support=expected.count(label))
    return dict(macro_f1=sum(s['f1'] for s in scores.values()) / len(labels), per_label=scores)


def evaluate(encoder, train, queries, mode='span'):
    if not train or not queries or mode not in {'span', 'sentence'}:
        raise ValueError('nonempty train/queries and span or sentence mode required')
    if {r['id'] for r in train} & {r['id'] for r in queries}:
        raise ValueError('train/query overlap')
    def encode(rows):
        texts = [r['text'] for r in rows]
        values = (encoder.encode_spans(texts, [tuple(r['span']) for r in rows]) if mode == 'span'
                  else encoder.encode_sentences(texts))
        if values.ndim != 2 or len(values) != len(rows) or not torch.isfinite(values).all():
            raise ValueError('encoder must return finite N,D vectors')
        if (values.norm(dim=1) == 0).any():
            raise ValueError('zero vector')
        return F.normalize(values.float(), dim=-1)
    candidates, vectors = encode(train), encode(queries)
    keys = sorted({(r['term'], r['label']) for r in train})
    prototypes = torch.stack([F.normalize(candidates[[i for i, r in enumerate(train)
                                if (r['term'], r['label']) == key]].mean(0), dim=0) for key in keys])
    predicted, reciprocal, hits = [], [], []
    for row, vector in zip(queries, vectors):
        valid = [i for i, key in enumerate(keys) if key[0] == row['term']]
        if not valid:
            raise ValueError('query term absent from training')
        best = int(torch.argmax(prototypes[valid] @ vector))
        predicted.append(keys[valid[best]][1])
        if row['label'] == AMBIGUOUS:
            continue  # no single-sense retrieval ground truth
        indices = [i for i, r in enumerate(train) if r['term'] == row['term'] and r['label'] != AMBIGUOUS]
        ranked = sorted(indices, key=lambda i: (-float(candidates[i] @ vector), train[i]['id']))
        ranks = [rank for rank, i in enumerate(ranked, 1) if train[i]['label'] == row['label']]
        if not ranks:
            raise ValueError('query sense absent from training retrieval candidates')
        reciprocal.append(1 / ranks[0])
        hits.append(int(ranks[0] == 1))
    result = classification_metrics([r['label'] for r in queries], predicted, sorted({r['label'] for r in train}))
    result.update(mode=mode, predictions=[dict(id=r['id'], expected=r['label'], predicted=p)
                                         for r, p in zip(queries, predicted)],
                  retrieval=dict(mrr=sum(reciprocal) / max(len(reciprocal), 1),
                                 hit_at_1=sum(hits) / max(len(hits), 1), count=len(hits),
                                 excluded_ambiguous=sum(r['label'] == AMBIGUOUS for r in queries)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model-dir')
    source.add_argument('--chat-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--mode', choices=['span', 'sentence'], default='span')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-length', type=int, default=256)
    args = parser.parse_args()
    if args.chat_dir:
        if not (args.chat_dir / 'model.pt').is_file():
            raise FileNotFoundError('chat checkpoint model.pt missing')
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from transformer_chat import reply
        pairs = json.loads((DATA / f'sft_{args.split}.json').read_text(encoding='utf-8'))
        result = dict(task='raw_chat_generation', split=args.split, checkpoint=str(args.chat_dir),
                      quality_evaluated=False, rows=[dict(id=row['id'], prompt=pair['prompt'],
                      expected_label=row['label'], reference=pair['reply'], raw_reply=reply(
                          pair['prompt'], out_dir=args.chat_dir, temperature=1., top_k=1,
                          top_p=1., repetition_penalty=1., device=args.device))
                          for row, pair in zip(load_split(args.split), pairs)])
    else:
        if __package__:
            from .contextual_embeddings import ContextualEncoder
        else:
            from contextual_embeddings import ContextualEncoder
        encoder = ContextualEncoder(args.model_dir, device=args.device, max_length=args.max_length)
        result = evaluate(encoder, load_split('train'), load_split(args.split), args.mode)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
