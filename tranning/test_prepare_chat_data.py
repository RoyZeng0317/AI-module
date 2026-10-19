import json, tempfile
from pathlib import Path
import pytest
from prepare_chat_data import ROOT, prepare


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(dir=ROOT / 'team-run') as directory:
        yield Path(directory)


def write(path, rows):
    path.write_text(json.dumps(rows), encoding='utf-8')
    return path


def test_quarantine_and_transitive_groups(workspace):
    rows = [dict(prompt='A', reply='one', group_id='family'), dict(prompt='B', reply='two', group_id='family'),
            dict(prompt='C', reply='two'), dict(prompt='D', reply='three'), dict(prompt='Conflict?', reply='x'),
            dict(prompt='CONFLICT', reply='y'), dict(prompt='Protected?', reply='z'), dict(prompt='hello', reply='你好'),
            dict(prompt='', reply='blank'), dict(prompt='A!', reply='one'), dict(prompt='Other heldout', reply='never train')]
    base = write(workspace/'base.json', rows)
    extra = write(workspace/'extra.json', [])
    evaluation = write(workspace/'eval.json', [dict(prompt='protected')])
    second_eval = write(workspace/'eval2.json', [dict(prompt='Other heldout')])
    report = prepare(base, extra, workspace/'out', eval_path=[evaluation, second_eval])
    manifest = json.loads((workspace/'out/manifest.json').read_text())
    assert len(manifest) == 4
    assert len({row['split'] for row in manifest if row['prompt'] in ['A', 'B', 'C']}) == 1
    assert set(report['counts']) == {'train', 'val'} and min(report['counts'].values()) > 0
    assert {entry['reason'] for entry in report['exclusions']} == {'conflicting_prompt', 'protected_eval_prompt', 'english_prompt_cjk_reply', 'malformed_or_blank', 'duplicate_pair'}
    assert json.loads(base.read_text()) == rows
    with pytest.raises(FileExistsError):
        prepare(base, extra, workspace/'out')


def test_overrides_and_rejections(workspace):
    base = write(workspace/'base.json', [dict(prompt='hi', reply='你好'), dict(prompt='Bye', reply='Goodbye')])
    extra = write(workspace/'extra.json', [])
    override = write(workspace/'override.json', [dict(prompt='hi', reply='Hello', reason='English reply')])
    report = prepare(base, extra, workspace/'out', overrides=override)
    assert report['exclusions'][0]['reason'] == 'replaced_by_override'
    for options in [dict(seed='42'), dict(val_ratio=0), dict(val_ratio=1), dict(eval_path=base)]:
        with pytest.raises(ValueError):
            prepare(base, extra, workspace/'bad', **options)
    write(override, [dict(prompt='missing', reply='Missing')])
    with pytest.raises(ValueError, match='not found'):
        prepare(base, extra, workspace/'bad', overrides=override)
    write(base, [dict(prompt='a', reply='same'), dict(prompt='b', reply='same')])
    with pytest.raises(ValueError, match='two independent'):
        prepare(base, extra, workspace/'bad')
    assert not (workspace/'bad').exists()


def test_duplicate_metadata_keeps_group_bridge(workspace):
    base = write(workspace/'base.json', [dict(prompt='a', reply='one', group_id='left'),
        dict(prompt='A!', reply='one', group_id='right'), dict(prompt='b', reply='two', group_id='right'),
        dict(prompt='c', reply='three')])
    extra = write(workspace/'extra.json', [])
    prepare(base, extra, workspace/'out')
    manifest = json.loads((workspace/'out/manifest.json').read_text())
    assert len({row['split_group'] for row in manifest if row['prompt'] in ('a', 'b')}) == 1


def test_eval_quarantines_whole_transitive_family_and_labels(workspace):
    rows = [dict(prompt='Held out', reply='first', group_id='g1'),
        dict(prompt='Paraphrase', reply='second', group_id='g1'),
        dict(prompt='PARAPHRASE!', reply='second', group_id='g2'),
        dict(prompt='Translation cousin', reply='third', group_id='g2'),
        dict(prompt='Reply cousin', reply='third'),
        dict(prompt='Label only match', reply='fourth', group_id='eval-label'),
        dict(prompt='Independent a', reply='answer a'), dict(prompt='Independent b', reply='answer b')]
    base = write(workspace/'base.json', rows)
    extra = write(workspace/'extra.json', [])
    evaluation = write(workspace/'eval.json', [dict(prompt='Held out'), dict(prompt='Unseen question', group_id='eval-label')])
    report = prepare(base, extra, workspace/'out', eval_path=evaluation)
    manifest = json.loads((workspace/'out/manifest.json').read_text())
    assert {row['prompt'] for row in manifest} == {'Independent a', 'Independent b'}
    isolated = [entry for entry in report['exclusions'] if entry['reason'].startswith('protected_eval')]
    assert len(isolated) == 6
    assert any(entry['record']['prompt'] == 'Reply cousin' for entry in isolated)
