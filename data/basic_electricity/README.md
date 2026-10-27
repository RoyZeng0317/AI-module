# 基本電學：先作答，再核對

2026-10-27。本階段完成資料與唯讀基準，沒有訓練或切換正式模型。

## 可用範圍

使用者 ZIP 共43檔：33 PDF（402頁）、7 DOCX、2 XLSX、1 Python。原檔在 `source/Basic electricity/`，逐檔 SHA 與格式風險在 `v1/source_inventory.json`、`v1/source_index.json`；附件 Python 與試算表公式均未執行。

人工逐題核對24題無圖選擇題，跨103／104／105／107／109年度與習作第八章。以來源群組切分，seed=43：train18、dev3、test3。題目保留原文符號、負號與單位；NFKC僅作比對，不改寫實際輸入。試卷03／04檔名差異以首頁科目、年份及題號核對，不以檔名猜答案。官方送分、掃描題、圖形／承上題、缺失公式及未配對內容均未納入；其他內容保留為來源層級隔離紀錄，不宣稱43檔皆已變成可訓練題。

- `v1/questions/{train,dev,test}.jsonl`：只有題幹、四選項、題目來源與版本；沒有正解或詳解。
- `v1/evaluator/keys.jsonl`：評分鍵與答案出處；`reviewed_bank.json` 是可重用人工核對種子，`extracted/` 保留43檔的只讀抽取證據。
- `v1/quarantine.jsonl`：43筆**來源剩餘內容**隔離紀錄，不是43道錯題。
- `v1/reports/duplicate_pages.json`：121組具有足夠文字的重複頁證據；圖像／低字數頁不只憑文字 hash 判定相同。
- `v1/reports/contamination.json`：檢查目前14份 pairs／corpus 檔，未見題幹完全重複；歷史訓練曝光仍未知，不能保證模型從未見過同類題。

「隔離評分鍵」是軟體資料流分離，沒有加密或阻止本機使用者讀檔。整理來源時會核對原解答；**推論程序只讀 questions，先將原始作答 fsync 落地，另一個 grade 指令才開 keys**。不得把 evaluator／原始詳解／test 當成 tokenizer、pretrain 或一般 pairs 語料。

## 操作

以下在專案根目錄執行，輸出路徑必須未存在；避免覆寫首次作答。每輪最多12題。

```powershell
# 第一步：只給題目，直接呼叫自建 GPT，不經搜尋／工具／RAG
.\.venv\Scripts\python.exe tranning/electricity_practice.py attempt --split train --limit 9 --output output/basic-electricity/new_train_attempts.jsonl

# 第二步：讀取已保存作答後才核對；糾錯提示不含正確選項或詳解
.\.venv\Scripts\python.exe tranning/electricity_practice.py grade --attempts output/basic-electricity/new_train_attempts.jsonl --output output/basic-electricity/new_train_grade.json --feedback output/basic-electricity/new_train_feedback.jsonl

# 重建資料需新版本，使用持久保存的來源清單及人工種子
.\.venv\Scripts\python.exe tranning/electricity_dataset.py --inventory data/basic_electricity/v1/source_inventory.json --bank data/basic_electricity/v1/evaluator/reviewed_bank.json --out data/basic_electricity/v2
```

dev只評估，不產生糾錯；test凍結，禁止回饋或第二次有效作答。本次test三題尚未作答。超過完整題目＋SEP＋64回答token的上下文預算，或出現UNK字元，均明確略過，不截斷題目／刪除單位。當前 runner 只驗證正式 legacy GPT。選擇題需唯一明確的最後答案，矛盾、多答案、列舉不給分；自由數值、單位推導等未有可靠契約者一律待人工，不用子字串比對。

## 實際首答基準

直接讀 `tranning/gpt_chat_runs`，temperature=0、seed=43、max_new_tokens=64。train首9題＋dev3題共12題，完整prompt為108–192／109–146 token（含SEP），UNK=0，無上下文略過。**0/12產生合法最終答案**，原文包括「好嗎？」「當然可以值得嗎？」等偏題回覆；不是只差輸出格式，更不能說模型已會基本電學。

原始紀錄在 `output/basic-electricity/baseline_train.jsonl`、`baseline_dev.jsonl`，各自評分報告與僅train的受限feedback另存。分母包含所有已完成但格式無效的回覆，不只挑解析成功者。另保存三筆一般聊天原始基準 `chat_retention_baseline.jsonl`，供未來比較是否遺忘；這三筆不是全面聊天品質驗收。

資料manifest SHA256：`5ae9b62143406555e3270509295b2541a5429ab36b7abfabe2a53f1bc898cdf7`。正式模型、tokenizer與受保護的使用者修改檔案均未改動。下一步僅為 [待核准提案](PILOT_PROPOSAL.md)，沒有可偷跑的訓練指令。
