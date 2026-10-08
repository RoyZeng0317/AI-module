# Sinco 情緒訓練資料匯入

2026-10-08 已實際下載、轉換並產生以下訓練資料。原本 `data/pairs.json` 保留，新的資料可直接由 `transformer_chat.py finetune --data ... --val-data ...` 讀取。沒有啟動模型訓練。

| 用途／來源 | train | val | test |
| --- | ---: | ---: | ---: |
| 對話回覆合計 | 27,934 | 3,664 | 3,766 |
| ↳ ESConv | 4,548 | 516 | 613 |
| ↳ EmpatheticDialogues | 17,675 | 2,743 | 2,521 |
| ↳ DailyDialog | 5,179 | 339 | 557 |
| ↳ 原本 Sinco 資料（分組後、符合長度） | 532 | 66 | 75 |
| CPED 情緒辨識 | 83,964 | 10,016 | 26,214 |
| GoEmotions 情緒辨識 | 43,166 | 5,412 | 5,422 |

這些是去重與過濾後的實際筆數，不是原始資料集規模。原本資料未被刪除；新集合只納入符合長度限制的舊對話。來源、下載 SHA256、版本 pin 與詳細過濾筆數見 `source_downloads.json`、`import_report.json`。

## 檔案與使用

- `pairs_train.json`／`pairs_val.json`／`pairs_test.json`：新的完整回覆訓練集，包含符合規則的既有 Sinco 對話，欄位仍有 `prompt`、`reply`，額外保存來源、conversation_id、split 等。
- `external_pairs_train.json`／`external_pairs_val.json`／`external_pairs_test.json`：僅外部對話資料的版本，適合隔離比較外部資料。
- `classification/cped_{train,val,test}.jsonl`：繁體中文（OpenCC s2t 字型轉換）的細粒度情緒標籤，保留原始 sentiment／對話行為。
- `classification/goemotions_{train,val,test}.jsonl`：英文多標籤情緒辨識，保留全部標籤，沒有強行壓成 0~5。
- `licenses/`：來源授權文字。來源為研究資料，原始／整理後資料檔不納入本 PR 的公開版本控制；交付 ZIP 內提供實際資料，亦可使用下載腳本重建。

在專案根目錄重建（原始下載約 64MB，輸出加原始資料合計約 134MB）：

```bash
python -m pip install -r tranning/emotion_dataset_requirements.txt
python tranning/import_emotion_datasets.py --tokenizer-dir tranning/gpt_pretrain_runs_v2
```

若沒有這份 BPE tokenizer，省略 `--tokenizer-dir`；split 隔離仍會驗證，但不會產生 tokenizer 長度／UNK 報告。來源檔依 SHA256 驗證，已有正確快取時不重下載；來源內容改變會拒絕，不會靜默改用新版。

本機微調指令（需先備妥自己的 model.pt；本次沒有執行）：

```bash
python tranning/transformer_chat.py finetune --data data/emotion_datasets/pairs_train.json --val-data data/emotion_datasets/pairs_val.json --pretrain-dir tranning/gpt_pretrain_runs_v2 --out-dir tranning/gpt_emotion_dataset_runs --epochs 12 --batch-size 8 --patience 3 --checkpoint-every 1 --dropout 0.2 --weight-decay 0.05 --gpu-mem-fraction 0.85
```

務必傳 `--val-data`；不要讓訓練腳本再把 train 拆成隨機驗證集。`pairs_test.json` 是保留的最終測試資料，不可給訓練、early stopping、pretrain 或 rehearsal。

目前 `MentalHealth/action.py` 仍是 0~5 分數的迴歸模型，**不能直接讀取上面的細粒度／多標籤 JSONL**。辨識資料已整理好，但要以這些標籤訓練，需要另做對應的分類頭與 loader；本次沒有把 categorical labels 擅自改成不存在的強度分數。

## 語言與品質

回覆 train 中英文 27,445 組、繁中 489 組；**外部回覆資料沒有翻成中文**。這份新集合適合英文／雙語回覆實驗，不能視為新增 27,000 多組繁中安慰話術。要改善中文模型，還需挑選與人工翻譯、審閱英文資料，並在中文獨立測試集上比較。原本中文主檔維持可用。

CPED 來自電視劇。實際查看 comfort 標註後發現部分台詞仍敷衍或責備，因此這次只作情緒辨識資料，不把它們自動當成理想安慰回覆。OpenCC 僅做簡繁轉換，不是英文翻譯，也不改写語氣。

- ESConv：合併連續 seeker／supporter 輪次，僅保留 Question、Affirmation and Reassurance、Reflection of feelings、Restatement or Paraphrasing 策略的回覆區塊，排除混入其他策略的整段回覆；過短問候／反馈不納入。
- EmpatheticDialogues：每段對話只取首次自述 → 第一個聆聽者回覆。後面的指代通常需要完整歷史，目前聊天模型不吃歷史，故不把那些短續句硬拆成獨立 pairs。還原 `_comma_`。
- DailyDialog：只取 user → system、user 有情緒標註的相鄰句對；排除回覆標成 anger／disgust 的句對。
- 自動過濾不等於逐筆人工審閱，也不代表所有回覆都已達到同理品質。

## 資料隔離與驗證

CPED、EmpatheticDialogues、DailyDialog、GoEmotions 保留來源的官方 splits。ESConv 沒有隨主 JSON 提供 train/val/test，本次按完整對話文字 SHA256 + seed 42 約 80/10/10 切分；既有 Sinco 按正規化 prompt 分組同樣切分，不拆散相同 prompt。

去重優先保留 test，再 val，再 train：跨 split 的相同正規化 prompt／text 從較低優先集合移除；不搬動對話至不同集合。同 split 相同 prompt 的不同回覆可保留，完全相同的句對只留一筆。對話群組若跨 source splits，匯入直接報錯。

实际檢查：回覆資料與各辨識資料的 train/val/test 之間，正規化文字重疊與 conversation_id 重疊皆為 0。用現有 gpt_pretrain_runs_v2 的 BPE 檢查全部回覆資料，沒有超過 512-token block 的句對；各外部來源 UNK 率低於 0.01%。過長資料被排除而非截斷，詳見 `validation_report.json`。

既有 Sinco 部分對話可能已進入 pretrain_v2；因此評估新外部資料時可優先看 `external_pairs_test.json`。若未來預訓練新 corpus，任何這些 val/test 都不能再混入。

```bash
python -m pytest -q tranning/test_import_emotion_datasets.py
```

## 來源與授權

- ESConv：[官方來源](https://github.com/thu-coai/Emotional-Support-Conversation)。CC BY-NC 4.0；README 另要求資料與程式作學術研究使用。未使用 FailedESConv。請保留作者、論文與授權。
- EmpatheticDialogues：[官方來源](https://github.com/facebookresearch/EmpatheticDialogues)。CC BY-NC 4.0；原 tar.gz 下載 URL 與 hash 保存於 manifest。
- CPED：[官方來源](https://github.com/scutcyr/CPED)。repository Apache 2.0；本次只使用文字與標籤，不下載影音。
- GoEmotions：[Google 官方來源](https://github.com/google-research/google-research/tree/master/goemotions)。repository Apache 2.0，資料來自人類標註 Reddit 评论。
- DailyDialog：[原始資料卡](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)，CC BY-NC-SA 4.0，研究用途；原作者舊官網已停放，使用 [ConvLab 整理版](https://github.com/ConvLab/ConvLab-3/tree/master/data/unified_datasets/dailydialog)。原資料卡與整理版版本皆固定並記錄；沒有把 ConvLab 的程式授權當成 DailyDialog 資料授權。

此 PR 提供匯入程式、版本／授權與結果報告，沒有把非商業資料當成無限制的公開語料。若之後要商用，必須另選相容授權來源或取得授權。
