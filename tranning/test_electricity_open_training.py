import json
from pathlib import Path
import pytest
from electricity_open_training import records, prepare, preflight, DATA, BASE, train


def test_groups_and_reserved_example():
    rows=list(records())
    assert [sum(r[0]==s for r in rows) for s in ('train','dev','test')]==[22,10,10]
    for column in (2,3,4):
        groups=[{r[column] for r in rows if r[0]==s} for s in ('train','dev','test')]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert all('100歐姆' not in r[4] for r in rows)


def test_real_tokenizer_budget_and_unk():
    report=preflight()
    assert all(r['max_tokens']<=513 for r in report.values())


def test_version_cannot_overwrite():
    with pytest.raises(ValueError,match='已存在'): prepare(DATA)


def test_training_cannot_overwrite_base():
    with pytest.raises(ValueError,match='輸出'): train(output=BASE)


def test_private_targets_not_in_question_files():
    for split in ('train','dev','test'):
        for line in (DATA/f'{split}.questions.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            assert not {'target','reply','answer'} & row.keys()


def test_toy_wrapper_records_actual_updates(tmp_path, monkeypatch):
    import torch, transformer_chat as gpt
    def toy_loop(model, train_loader, val_loader, optimizer, *args, **kwargs):
        model(torch.ones(1,1)).sum().backward(); optimizer.step()
        return [dict(epoch=1,train_loss=2.0,val_loss=2.1)]
    def toy_finetune(data_path, **kw):
        model=torch.nn.Linear(1,1)
        history=gpt._run_epochs(model,None,None,torch.optim.SGD(model.parameters(),lr=.01))
        target=kw['out_dir']
        torch.save(model.state_dict(),target/'model.pt')
        (target/'history.json').write_text(json.dumps(history),encoding='utf-8')
    monkeypatch.setattr(gpt,'_run_epochs',toy_loop)
    monkeypatch.setattr(gpt,'finetune',toy_finetune)
    output=tmp_path/'toy'
    train(output=output,epochs=1)
    report=json.loads((output/'run.json').read_text(encoding='utf-8'))
    assert report['updates']['steps']==1 and report['best_epoch']==1
    assert report['replay_count']==22
