# 字詞語意資料與評估

這是人工撰寫的繁體中文小型示範資料，不是外部語料或泛化能力證據。共有38筆：train20、val9、test9。六種確定詞義：蘋果手機／水果、鼠的電腦裝置／動物、病毒的惡意程式／生物病原；另有 `insufficient_context`。鼠是為跨動物／裝置設計的簡稱，日常繁體中文較常說滑鼠，不能把此集合視為自然分布。

## 檔案

- `data/word_sense/{train,val,test}.json`：分類與檢索共用標註，含唯一id、family、term、text、半開字元span、label與說明reply。span是Python字元索引，不是UTF-8位元組位置。
- `data/word_sense/sft_{train,val,test}.json`：只有標準prompt/reply欄位，供既有GPT finetune；不修改原本pairs。
- `tranning/build_word_sense.py`：可重建上述六個檔案，會覆寫這六個示範檔。

train兩個家族為購買／目的、照護指示；val是事件與否定報告；test是地點衝突的詢問／轉述。不同split未共用句型家族，並非只替換量詞後隨機切分。蘋果具有台／顆與品牌／水果線索，目標詞後方包含關鍵線索；val含明確否定，test含容易誤導的店面或職業。每個split皆有三種資訊不足情形。家族只有四個且資料非常少，獨立句型仍不代表真正獨立領域。

## 本地encoder評估

從專案根目錄執行，model-dir需已備妥本機tokenizer與權重：

```powershell
.venv/Scripts/python.exe -m tranning.word_sense_eval --model-dir models/bge-small-zh-v1.5 --split val --mode span
.venv/Scripts/python.exe -m tranning.word_sense_eval --model-dir models/bge-small-zh-v1.5 --split test --mode sentence
```

`span`與`sentence`需分別報告，不混為同一能力。分類用train中的同詞、同標籤向量均值建立原型，再以餘弦相似度預測；資訊不足也是一個原型類別，不是根據關鍵字或test調整門檻。輸出七類各自precision／recall／F1／support及macro-F1，並保留逐筆預測。

檢索候選僅使用train中同詞的確定詞義句子，以餘弦排序，報same-sense MRR與Hit@1。資訊不足無唯一檢索真值，明確排除並報排除數；仍納入分類。相同分數以id排序確保可重現。這是跨詞義檢索測試，不是一般開放領域搜尋。test不參與原型建立、訓練或門檻選擇；val僅供選擇設定。

聊天原始生成與以上評估分開：不呼叫smart_reply、工具或回覆模板，也不把標註reply當模型輸出。可用下列命令保存val的GPT原始回覆，由人工檢查詞義、資訊不足追問與一般聊天保留能力。固定temperature1、top_k1、top_p1、repetition_penalty1，不做substring評分。test留待設定定案後再執行。

```powershell
.venv/Scripts/python.exe -m tranning.word_sense_eval --chat-dir tranning/gpt_chat_runs --split val --device cpu --output output/word-sense/chat_val_baseline.json
```

## 手動新增一筆詞義範例（累加進向量庫）

`tranning/add_word_sense_entry.py` 讓你手動補一筆「句子＋目標詞＋詞義」，會自動用本機 `ContextualEncoder` 算出上下文向量，累加（不覆寫）進 `train.json`／`sft_train.json`，向量另存 `data/word_sense/train_vectors.json`（以 id 為 key）：

```powershell
.venv/Scripts/python.exe -m tranning.add_word_sense_entry --text "我在果汁店訂了一箱蘋果。" --term "蘋果" --label apple_fruit
```

已知 label：`apple_phone`／`apple_fruit`／`mouse_device`／`mouse_animal`／`virus_computer`／`virus_biological`／`insufficient_context`，會自動套用對應 `reply`；要新增新的 label 需加 `--new-label` 並自己提供 `--reply`。目標詞在句中出現超過一次要加 `--occurrence`（0起算）指定是第幾次。重複的 text+term 會被擋下，避免無聲重複累積。`--no-vector` 可以只新增文字範例、先不算向量（例如本機還沒有模型可用時）。

## SFT準備與核准後實驗

先選實際已存在的legacy GPT基底，不能把BGE目錄當GPT。`prepare_word_sense.py --base <GPT目錄>`不訓練，會檢查六個SFT檔及原pairs replay的UNK、截斷與有效筆數，輸出coverage.json。新fixture任何UNK／截斷都阻擋訓練檔輸出；原replay保留並報coverage。dialogue_v1基底會明確拒絕，避免錯誤訓練格式。需另用現有context trainer處理該格式。

```powershell
.venv/Scripts/python.exe tranning/prepare_word_sense.py --base tranning/gpt_chat_runs --out-dir data/word_sense/prepared
```

已以實際legacy基底`tranning/gpt_chat_runs`執行準備，block_size512。新train20／val9／test9全部無UNK、無截斷、有效38筆。原pairs replay742筆中723筆無UNK，19筆含UNK；742筆皆無截斷或reply全被截去。train_replay共762筆，保留原742筆並加入新20筆；有效743筆，新消歧資料約2.62%。報告位於`data/word_sense/prepared/coverage.json`。19筆舊UNK未被宣稱修復。先前以gpt_chat_runs_v2準備的18筆版本保留在prepared_legacy_v2_audit，僅為稽核，不用於本輪命令。

輸出目錄必須不存在，重跑請另選新的out-dir。原pairs全量replay與新train20筆合併到獨立train_replay.json，val9筆獨立保存，test不加入。訓練前仍需一般聊天保留能力對照。

以下是需要本輪核准後才能執行的精確設定範本；pretrain-dir必須與coverage的base一致，out-dir需全新，並先確認可訓練層數（範例freeze-layers0避免小模型全被凍結）：

```powershell
.venv/Scripts/python.exe tranning/transformer_chat.py finetune --data data/word_sense/prepared/train_replay.json --val-data data/word_sense/prepared/val.json --pretrain-dir tranning/gpt_chat_runs --out-dir tranning/gpt_word_sense_runs/pilot_v1 --epochs 5 --lr 5e-5 --batch-size 8 --gpu-mem-fraction .85 --patience 2 --checkpoint-every 1 --freeze-layers 0 --typo-noise-prob 0
```

不覆寫正式權重，不因小型val loss變好就切換正式聊天；本次未執行訓練。0.85為GPU記憶體配置上限設定，並非保證不OOM。若OOM、NaN、品質退步或coverage失敗，停止並保留紀錄。
