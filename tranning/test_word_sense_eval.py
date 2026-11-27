import json
import torch
import pytest
from tranning.word_sense_eval import DATA, load_split, evaluate, classification_metrics


def test_dataset_isolation_and_sft_alignment():
    splits = [load_split(s) for s in ('train', 'val', 'test')]
    for i, rows in enumerate(splits):
        for other in splits[i + 1:]:
            assert not {r['family'] for r in rows} & {r['family'] for r in other}
            assert not {r['text'] for r in rows} & {r['text'] for r in other}
        for row in rows:
            a, b = row['span']
            assert row['text'][a:b] == row['term'] and '?' not in row['text']
        split = ('train', 'val', 'test')[i]
        pairs = json.loads((DATA / f'sft_{split}.json').read_text(encoding='utf-8'))
        assert len(pairs) == len(rows)
        assert all(set(p) == {'prompt', 'reply'} and r['text'] in p['prompt'] and p['reply'] == r['reply']
                   for p, r in zip(pairs, rows))
        assert {r['label'] for r in rows} == {r['label'] for r in splits[0]}


def test_metrics_count_false_positives_and_missing_predictions():
    result = classification_metrics(['a', 'a', 'b'], ['a', 'a', 'a'], ['a', 'b'])
    assert result['macro_f1'] == pytest.approx(.4)
    assert result['per_label']['b']['support'] == 1
    with pytest.raises(ValueError):
        classification_metrics(['a'], [], ['a'])


def test_fake_encoder_pipeline_both_modes_and_ambiguity():
    train, queries = load_split('train'), load_split('test')
    labels = sorted({r['label'] for r in train})
    # An oracle fixture lookup tests plumbing only, never model semantics.
    mapping = {r['text']: labels.index(r['label']) for r in train + queries}
    class FakeEncoder:
        def encode_sentences(self, texts):
            return torch.eye(len(labels))[[mapping[t] for t in texts]]
        def encode_spans(self, texts, spans):
            assert len(texts) == len(spans)
            return self.encode_sentences(texts)
    for mode in ('sentence', 'span'):
        result = evaluate(FakeEncoder(), train, queries, mode)
        assert result['macro_f1'] == 1
        assert result['retrieval'] == dict(mrr=1, hit_at_1=1, count=6, excluded_ambiguous=3)
    with pytest.raises(ValueError, match='overlap'):
        evaluate(FakeEncoder(), train, train)
