"""Manually append one word-sense example and cache its context vector.

Appends to data/word_sense/train.json and sft_train.json (never overwrites
existing rows, unlike build_word_sense.py). The context vector is computed
with ContextualEncoder and cached in data/word_sense/train_vectors.json,
keyed by id, so the vector library grows alongside the training fixtures.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/word_sense'
TRAIN, SFT_TRAIN, VECTORS = DATA / 'train.json', DATA / 'sft_train.json', DATA / 'train_vectors.json'
REPLIES = {'apple_phone': '指的是蘋果品牌手機。', 'apple_fruit': '指的是可食用的蘋果水果。',
           'mouse_device': '指的是操作電腦的滑鼠。', 'mouse_animal': '指的是會活動的鼠類動物。',
           'virus_computer': '指的是感染電腦的惡意程式。', 'virus_biological': '指的是感染生物的病毒。',
           'insufficient_context': '資訊不足，請補充用途、行為或其他語境。'}


def locate_span(text, term, occurrence):
    starts = [i for i in range(len(text)) if text.startswith(term, i)]
    if not starts:
        raise ValueError(f'term {term!r} not found in text')
    if occurrence is None:
        if len(starts) > 1:
            raise ValueError(f'term {term!r} appears {len(starts)} times; pass --occurrence 0-based')
        occurrence = 0
    if not 0 <= occurrence < len(starts):
        raise ValueError(f'occurrence {occurrence} out of range ({len(starts)} matches)')
    start = starts[occurrence]
    return [start, start + len(term)]


def next_manual_id(rows):
    return f"manual_{sum(1 for r in rows if r['id'].startswith('manual_'))}"


def add_entry(text, term, label, reply=None, family='manual', occurrence=None,
              new_label=False, model_dir='models/bge-small-zh-v1.5', compute_vector=True, max_length=512):
    if label not in REPLIES and not new_label:
        raise ValueError(f'unknown label {label!r}; pass new_label=True to add a new sense label')
    reply = reply or REPLIES.get(label)
    if not reply:
        raise ValueError('reply is required for a new label')

    rows = json.loads(TRAIN.read_text(encoding='utf-8'))
    if any(r['text'] == text and r['term'] == term for r in rows):
        raise ValueError('duplicate text+term already in train.json')

    entry = dict(id=next_manual_id(rows), family=family, term=term, text=text,
                 span=locate_span(text, term, occurrence), label=label, reply=reply)

    vector = None
    if compute_vector:
        from tranning.contextual_embeddings import ContextualEncoder
        encoder = ContextualEncoder(model_dir, max_length=max_length)
        vector = encoder.encode_spans([text], [tuple(entry['span'])])[0].tolist()

    rows.append(entry)
    TRAIN.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    pairs = json.loads(SFT_TRAIN.read_text(encoding='utf-8'))
    pairs.append(dict(prompt=f'請依語境說明「{term}」的意思：{text}', reply=reply))
    SFT_TRAIN.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if vector is not None:
        store = json.loads(VECTORS.read_text(encoding='utf-8')) if VECTORS.exists() else {}
        store[entry['id']] = dict(term=term, label=label, span=entry['span'], vector=vector)
        VECTORS.write_text(json.dumps(store, ensure_ascii=False) + '\n', encoding='utf-8')

    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--term', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reply', default=None, help='required when --label is new')
    parser.add_argument('--family', default='manual')
    parser.add_argument('--occurrence', type=int, default=None, help='0-based index when term repeats in text')
    parser.add_argument('--new-label', action='store_true', help='allow a label outside the known set')
    parser.add_argument('--model-dir', default='models/bge-small-zh-v1.5')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--no-vector', action='store_true', help='skip context-vector computation')
    args = parser.parse_args()
    try:
        entry = add_entry(args.text, args.term, args.label, args.reply, args.family, args.occurrence,
                          args.new_label, args.model_dir, not args.no_vector, args.max_length)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"added {entry['id']} to train.json" + ('' if args.no_vector else ' (vector cached in train_vectors.json)'))


if __name__ == '__main__':
    main()
