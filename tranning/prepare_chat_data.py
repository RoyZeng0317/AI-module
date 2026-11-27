"""Prepare auditable, group-disjoint chat data without changing source files."""
import argparse, hashlib, json, random, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def local_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All paths must stay inside the project")
    return path


def prepare(base, extra, out_dir, seed=42, val_ratio=.15, overrides=None, eval_path=None):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 < val_ratio < 1:
        raise ValueError("seed must be an integer; val_ratio must be between 0 and 1")
    destination = local_path(out_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    sources = [local_path(base), local_path(extra)]
    override_path = local_path(overrides) if overrides else None
    eval_paths = list(eval_path) if isinstance(eval_path, (list, tuple)) else [eval_path] if eval_path else []
    default_eval = ROOT / 'data/chat_eval_cases.json'
    if default_eval.exists():
        eval_paths.append(default_eval)
    protected_paths = list(dict.fromkeys(local_path(path) for path in eval_paths))
    if any(path in sources + ([override_path] if override_path else []) for path in protected_paths):
        raise ValueError("Evaluation data cannot be a training source")
    hashes, exclusions, records = {}, [], []

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {path}")
        return data

    replacements = {}
    if override_path:
        for row in read(override_path):
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and normalize(row[k]) for k in ('prompt', 'reply')):
                raise ValueError("Invalid override record")
            key = normalize(row['prompt'])
            if key in replacements:
                raise ValueError("Duplicate override prompt")
            replacements[key] = row
    eval_rows = [row for path in protected_paths for row in read(path) if isinstance(row, dict)]
    protected = {normalize(row['prompt']) for row in eval_rows if isinstance(row.get('prompt'), str)}
    protected_labels = {str(row['group_id']) for row in eval_rows if row.get('group_id')}
    applied = set()
    for path in sources:
        for index, original in enumerate(read(path)):
            location = f"{path.relative_to(ROOT)}:{index}"
            if not isinstance(original, dict) or not all(isinstance(original.get(k), str) and normalize(original[k]) for k in ('prompt', 'reply')):
                exclusions.append(dict(location=location, reason='malformed_or_blank', record=original))
                continue
            row = dict(original)
            key = normalize(row['prompt'])
            if key in replacements:
                exclusions.append(dict(location=location, reason='replaced_by_override', record=original))
                row.update(replacements[key]); applied.add(key)
            row.update(location=location)
            records.append(row)
    if replacements.keys() - applied:
        raise ValueError("Override prompt not found in sources")
    # Build components before any filtering: excluded duplicates/conflicts can bridge families.
    components, linked = list(range(len(records))), {}

    def component(i):
        while components[i] != i:
            components[i] = components[components[i]]; i = components[i]
        return i

    for i, row in enumerate(records):
        row_keys = [('prompt', normalize(row['prompt'])), ('reply', normalize(row['reply']))]
        if row.get('group_id'):
            row_keys.append(('group', str(row['group_id'])))
        for key in row_keys:
            if key in linked:
                components[component(i)] = component(linked[key])
            else:
                linked[key] = i
    blocked = {component(i) for i, row in enumerate(records)
               if normalize(row['prompt']) in protected or str(row.get('group_id', '')) in protected_labels}
    eligible = []
    for i, row in enumerate(records):
        row['source_component'] = component(i)
        if component(i) in blocked:
            reason = 'protected_eval_prompt' if normalize(row['prompt']) in protected else 'protected_eval_family'
            exclusions.append(dict(reason=reason, record=row))
        elif re.search('[A-Za-z]', row['prompt']) and not re.search('[\u3400-\u9fff]', row['prompt']) and re.search('[\u3400-\u9fff]', row['reply']) and not (row.get('topic') == 'translation' or row.get('translation') is True or re.search(r'\b(translat\w*|chinese|mandarin)\b', row['prompt'], re.I)):
            exclusions.append(dict(reason='english_prompt_cjk_reply', record=row))
        else:
            eligible.append(row)
    records = eligible
    answers = {}
    for row in records:
        answers.setdefault(normalize(row['prompt']), set()).add(normalize(row['reply']))
    conflicts = {key for key, values in answers.items() if len(values) > 1}
    clean, seen = [], {}
    for row in records:
        pair = normalize(row['prompt']), normalize(row['reply'])
        reason = 'conflicting_prompt' if pair[0] in conflicts else 'duplicate_pair' if pair in seen else None
        if reason:
            exclusions.append(dict(reason=reason, record=row))
            if reason == 'duplicate_pair' and row.get('group_id'):
                seen[pair].setdefault('duplicate_group_ids', []).append(row['group_id'])
        else:
            clean.append(row); seen[pair] = row
    parent = list(range(len(clean)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    keys = {}
    for i, row in enumerate(clean):
        group_keys = [('prompt', normalize(row['prompt'])), ('reply', normalize(row['reply']))]
        group_keys.append(('source_component', row['source_component']))
        if row.get('group_id'):
            group_keys.append(('group', str(row['group_id'])))
        group_keys.extend(('group', str(value)) for value in row.get('duplicate_group_ids', []))
        for key in group_keys:
            if key in keys:
                parent[find(i)] = find(keys[key])
            else:
                keys[key] = i
    groups = {}
    for i, row in enumerate(clean):
        groups.setdefault(find(i), []).append(row)
    ordered = list(groups.values())
    if len(ordered) < 2:
        raise ValueError("At least two independent groups are needed for nonempty train/val")
    random.Random(seed).shuffle(ordered)
    validation_groups = max(1, min(len(ordered)-1, round(len(ordered)*val_ratio)))
    splits = {'val': ordered[:validation_groups], 'train': ordered[validation_groups:]}
    manifest = []
    for split, group_list in splits.items():
        for group in group_list:
            group_key = hashlib.sha256('\n'.join(sorted(row['location'] for row in group)).encode()).hexdigest()[:16]
            manifest.extend(dict(row, split=split, split_group=group_key) for row in group)
    report = dict(seed=seed, requested_val_ratio=val_ratio, source_sha256=hashes,
                  counts={split: sum(row['split'] == split for row in manifest) for split in splits},
                  groups=len(groups), largest_group=max(map(len, groups.values())), exclusions=exclusions,
                  limitations='Explicit group_id and normalized identical prompts/replies only; semantic paraphrases require human review.')
    report['language_counts'] = {}
    for row in manifest:
        language = row.get('language') or ('zh-Hant-or-CJK' if re.search('[\u3400-\u9fff]', row['prompt']) else 'en-or-other')
        report['language_counts'][language] = report['language_counts'].get(language, 0) + 1
    report['disjoint'] = {field: not ({normalize(row[field]) for row in manifest if row['split'] == 'train'} & {normalize(row[field]) for row in manifest if row['split'] == 'val'}) for field in ('prompt', 'reply')}
    destination.mkdir(parents=True, exist_ok=False)
    for split in splits:
        (destination / f'{split}.json').write_text(json.dumps([{key: row[key] for key in ('prompt', 'reply')} for row in manifest if row['split'] == split], ensure_ascii=False, indent=2), encoding='utf-8')
    for name, value in [('manifest', manifest), ('audit', report)]:
        (destination / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'extra', 'out-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--overrides'); parser.add_argument('--eval', dest='eval_path', action='append')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--val-ratio', type=float, default=.15)
    args = vars(parser.parse_args())
    try:
        report = prepare(**args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ('counts', 'groups', 'largest_group')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
