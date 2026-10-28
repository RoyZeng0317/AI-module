"""開放式電學：生成資料、只讀基準與受控 SFT；正式 train 須先獲使用者核准。"""
import argparse, json, math, random
from fractions import Fraction
from pathlib import Path
from electricity_dataset import ROOT, inside_project, object_hash, file_hash, write_json, write_jsonl, read_jsonl

DATA = ROOT / 'data/electricity_open/v2'
BASE = ROOT / 'tranning/gpt_chat_runs'
OUT = ROOT / 'tranning/gpt_electricity_open_runs/pilot_v1'
SEED = 43


def records():
    """模板與數值組皆跨 split 隔離；同一三支路拓撲，不能宣稱拓撲泛化。"""
    templates = [('電阻為{r}歐姆，電壓{v}V，求電流並列式。', '電阻{r}歐姆通過{i}A，求電壓。', '電壓{v}V、電流{i}A，求電阻。'),
                 ('請算出{v}V加在{r}歐姆時的電流。', '一個{r}歐姆電阻的電流是{i}A，兩端電壓多少？', '若兩端{v}V流過{i}A，阻值多少歐姆？'),
                 ('負載阻值{r}歐姆且端電壓{v}V，請說明流過幾安培。', '已知負載{r}歐姆及{i}A，請說明端電壓。', '負載端電壓{v}V且電流{i}A，請說明阻值。')]
    cases = [[(10, 2), (30, 9), (40, 5), (60, 3), (80, 12), (120, 30)], [(15, 6), (25, 2)], [(35, 14), (45, 9)]]
    for s, split in enumerate(('train', 'dev', 'test')):
        for n, (r, v) in enumerate(cases[s]):
            i = Fraction(v, r)
            for kind in range(3):
                q = templates[s][kind].format(r=r, v=v, i=i)
                target = (f'依歐姆定律，I=V/R={v}/{r}={i} A（{i*1000} mA）。', f'依歐姆定律，V=IR=({i})*{r}={v} V。', f'依歐姆定律，R=V/I={v}/({i})={r} 歐姆。')[kind]
                yield split, f'{split}-ohm-{n}-{kind}', f'{split}-template-{kind}', f'{r}:{v}', q, target, 'numeric'
        unitq = ('電阻2千歐姆，電壓6V，求電流。', '求電流：8V電壓接4千歐姆電阻。', '15V接在3千歐姆上，電流多少？')[s]
        ur, uv = ((2000,6),(4000,8),(3000,15))[s]
        yield split, f'{split}-units', f'{split}-units', f'unit-{s}', unitq, f'先換算電阻為{ur}歐姆。I=V/R={uv}/{ur}={Fraction(uv,ur)} A（{Fraction(uv*1000,ur)} mA）。', 'numeric'
        for k, (q, t) in enumerate([(('只知道電阻，能求電流嗎？', '電流未知且只有阻值，如何求電壓？', '只給電壓，請求阻值。')[s], '資訊不足，請補充另一個已知的電壓、電流或電阻數值。'), (('電阻0歐姆加5V，求電流。', '將7V接在零歐姆，電流多少？', '阻值為0且電壓9V，請算電流。')[s], '零電阻不能用有限電流的歐姆定律直接相除；理想電源短路，需提供實際內阻。')]):
            yield split, f'{split}-invalid-{k}', f'{split}-invalid-{k}', f'invalid-{s}-{k}', q, t, 'manual'
        vl, vr, rl, rr, rc = ((12, 6, 2, 4, 3), (15, 5, 3, 6, 4), (18, 9, 4, 8, 6))[s]
        x = (Fraction(vl, rl)+Fraction(vr, rr))/(Fraction(1, rl)+Fraction(1, rr)+Fraction(1, rc))
        i1, i2 = (vl-x)/rl, (x-vr)/rr
        q = f'以地為0V，左節點L={vl}V，右節點R={vr}V。{rl}歐姆接L與X，{rr}歐姆接X與R，{rc}歐姆接X與地。' + ('請由節點電壓法轉成順時針網目電流法。', '請把X節點方程改寫為左右順時針網目方程。', '請比較節點法與兩個順時針網目電流的關係並求解。')[s]
        t = f'節點法：(X-{vl})/{rl}+(X-{vr})/{rr}+X/{rc}=0，X={x} V。左右順時針電流I1、I2，共用支路向下電流I1-I2。網目式：{rl+rc}I1-{rc}I2={vl}；-{rc}I1+{rr+rc}I2=-{vr}。I1=({vl}-X)/{rl}={i1} A，I2=(X-{vr})/{rr}={i2} A，共用電流={i1-i2} A。'
        yield split, f'{split}-mesh', f'{split}-mesh', f'mesh-{vl}:{vr}:{rl}:{rr}:{rc}', q, t, 'worked'


def prepare(destination=DATA):
    destination = inside_project(destination)
    if destination.exists():
        raise ValueError('版本目錄已存在，不覆寫')
    rows = list(records())
    for split in ('train', 'dev', 'test'):
        questions, keys = [], []
        for sp, ident, group, case, question, target, grading in rows:
            if sp != split: continue
            questions.append(dict(id=ident, split=sp, template_group=group, numeric_group=case, question=question, question_sha256=object_hash(question)))
            keys.append(dict(id=ident, target=target, grading=grading))
        write_jsonl(destination / f'{split}.questions.jsonl', questions)
        write_jsonl(destination / 'private' / f'{split}.keys.jsonl', keys)
    write_json(destination / 'manifest.json', dict(version=2, seed=SEED, source='explicit synthetic generator, not archive extraction', generator_sha256=file_hash(__file__), files={str(p.relative_to(destination)):file_hash(p) for p in destination.rglob('*.jsonl')}))


def preflight(dataset=DATA, base=BASE):
    from bpe_tokenizer import BPETokenizer, UNK
    tok = BPETokenizer.load(base)
    config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if file_hash(dataset / name) != digest: raise ValueError(f'資料指紋不符：{name}')
    groups, numbers, prompts, report = set(), set(), set(), {}
    for split in ('train', 'dev', 'test'):
        questions = read_jsonl(dataset / f'{split}.questions.jsonl')
        keys = {r['id']:r for r in read_jsonl(dataset / 'private' / f'{split}.keys.jsonl')}
        sg, sn = {r['template_group'] for r in questions}, {r['numeric_group'] for r in questions}
        if groups & sg or numbers & sn: raise ValueError('split 群組洩漏')
        groups |= sg; numbers |= sn
        if set(keys) != {r['id'] for r in questions}: raise ValueError('題目與鍵不一致')
        maximum = 0
        for row in questions:
            if set(row) != {'id','split','template_group','numeric_group','question','question_sha256'} or row['split'] != split or object_hash(row['question']) != row['question_sha256']: raise ValueError('公開題目欄位不符')
            if row['question'] in prompts: raise ValueError('重複題目')
            prompts.add(row['question'])
            ids = tok.encode(row['question']) + [4] + tok.encode(keys[row['id']]['target']) + [2]
            if UNK in ids or len(ids) > config['block_size']+1: raise ValueError(f"UNK 或超長，禁止截斷：{row['id']} ({len(ids)})")
            maximum = max(maximum, len(ids))
        report[split] = dict(count=len(questions), max_tokens=maximum)
    return report


def baseline(destination, base=BASE, questions=None, dataset=DATA):
    """不開啟答案檔；每次原始回答立即 append+fsync，完成後才可另行評閱。"""
    import os, torch
    import transformer_chat as gpt
    destination = inside_project(destination)
    prompts = questions or ['電阻100歐姆，電壓20V，依歐姆定律求電流。', '電阻2千歐姆，電壓6V，求電流。', '只知道電阻，能求電流嗎？', '以地為0V，左節點L=12V，右節點R=6V。2歐姆接L與X，4歐姆接X與R，3歐姆接X與地。請由節點電壓法轉成順時針網目電流法。', '你好', '你是誰', '今天心情不好，可以陪我聊聊嗎？']
    random.seed(SEED); torch.manual_seed(SEED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        for prompt in prompts:
            answer = gpt.reply(prompt, out_dir=base, max_new_tokens=180, temperature=0.7, top_k=30, top_p=0.85, device='cpu')
            stream.write(json.dumps(dict(prompt=prompt, question_sha256=object_hash(prompt), reply=answer, checkpoint_sha256=file_hash(base/'model.pt'), config_sha256=file_hash(base/'config.json'), tokenizer_sha256=file_hash(base/'bpe_vocab.json'), dataset_sha256=file_hash(dataset/'manifest.json'), code_sha256=file_hash(__file__), seed=SEED, generation=dict(max_new_tokens=180,temperature=.7,top_k=30,top_p=.85,device='cpu'), raw_gpt=True),ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def train(dataset=DATA, base=BASE, output=OUT, epochs=3):
    import torch
    import transformer_chat as gpt
    output, base = inside_project(output), inside_project(base)
    if output.exists() or output == base or output.is_relative_to(base) or not 1 <= epochs <= 3: raise ValueError('輸出必須全新、不可覆寫基底，epochs限1至3')
    attempts = read_jsonl(dataset/'train.attempts.jsonl')
    expected = {r['question'] for r in read_jsonl(dataset/'train.questions.jsonl')}
    fingerprints=dict(checkpoint_sha256=file_hash(base/'model.pt'),config_sha256=file_hash(base/'config.json'),tokenizer_sha256=file_hash(base/'bpe_vocab.json'),dataset_sha256=file_hash(dataset/'manifest.json'))
    if len(attempts)!=len(expected) or {r['prompt'] for r in attempts} != expected or any(not r.get('raw_gpt') or not r.get('code_sha256') or r.get('generation')!=dict(max_new_tokens=180,temperature=.7,top_k=30,top_p=.85,device='cpu') or r.get('question_sha256')!=object_hash(r['prompt']) or any(r.get(k)!=v for k,v in fingerprints.items()) for r in attempts): raise ValueError('必須先保存同版本與基底全部 train 原始作答')
    report = preflight(dataset, base)
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    def pairs(split):
        keys = {k['id']:k['target'] for k in read_jsonl(dataset/'private'/f'{split}.keys.jsonl')}
        return [dict(prompt=q['question'],reply=keys[q['id']]) for q in read_jsonl(dataset/f'{split}.questions.jsonl')]
    training, validation = pairs('train'), pairs('dev')
    tok = gpt.BPETokenizer.load(base)
    replay = json.loads((ROOT/'data/pairs.json').read_text(encoding='utf-8'))
    forbidden = ('歐姆','电阻','電阻','電流','电流','電壓','电压','網目','节点','節點','ohm','100Ω','20V')
    heldout = {q['question'] for split in ('dev','test') for q in read_jsonl(dataset/f'{split}.questions.jsonl')}
    replay = [p for p in replay if set(p) >= {'prompt','reply'} and p['prompt'] not in heldout and not any(t.casefold() in (p['prompt']+p['reply']).casefold() for t in forbidden) and 3 not in tok.encode(p['prompt']+p['reply']) and len(tok.encode(p['prompt']))+len(tok.encode(p['reply']))+2 <= 513]
    replay = random.Random(SEED).sample(replay, min(len(training), len(replay)))
    output.mkdir(parents=True)
    write_json(output/'train.private.json', training+replay); write_json(output/'dev.private.json', validation)
    original, updates = gpt._run_epochs, {'steps':0}
    def checked_loop(model, train_loader, val_loader, optimizer, *args, **kwargs):
        original_step = optimizer.step
        def checked_step(*a, **kw):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('梯度非有限值')
            result = original_step(*a, **kw); updates['steps'] += 1
            if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('權重非有限值')
            return result
        optimizer.step = checked_step
        history = original(model, train_loader, val_loader, optimizer, *args, **kwargs)
        if not history or updates['steps'] < 1 or any(not math.isfinite(r[k]) for r in history for k in ('train_loss','val_loss')): raise ValueError('訓練無有效更新或loss非有限值')
        return history
    gpt._run_epochs = checked_loop
    try:
        gpt.finetune(output/'train.private.json', pretrain_dir=base, out_dir=output, val_data_path=output/'dev.private.json', epochs=epochs, batch_size=2, lr=1e-5, freeze_embeddings=True, freeze_layers=7, typo_noise_prob=0, gpu_mem_fraction=0.75, device='cuda', patience=3)
    finally: gpt._run_epochs = original
    history=json.loads((output/'history.json').read_text(encoding='utf-8'))
    best=min(history,key=lambda r:r['val_loss'])
    write_json(output/'run.json', dict(seed=SEED, epochs=epochs, updates=updates, preflight=report, replay_count=len(replay), replay_source_sha256=file_hash(ROOT/'data/pairs.json'), checkpoint_sha256=file_hash(output/'model.pt'), base_sha256=file_hash(base/'model.pt'), best_epoch=best['epoch'], best_metrics=best, quality='尚須未見開放題與聊天回歸評估'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['prepare','preflight','baseline','attempt-train','train']); p.add_argument('--dataset',type=Path,default=DATA); p.add_argument('--base',type=Path,default=BASE); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--epochs',type=int,default=3)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.dataset)
    elif a.action=='preflight': print(json.dumps(preflight(a.dataset,a.base),ensure_ascii=False))
    elif a.action=='baseline': baseline(a.output,a.base)
    elif a.action=='attempt-train': baseline(a.dataset/'train.attempts.jsonl',a.base,[r['question'] for r in read_jsonl(a.dataset/'train.questions.jsonl')],a.dataset)
    else: train(a.dataset,a.base,a.output,a.epochs)
