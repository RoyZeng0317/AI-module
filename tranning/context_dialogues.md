# 本地多輪資料與原始生成評估

2026-10-26狀態：工程串接已完成，但兩次有界微調未達模型切換門檻，正式權重保持原版。實驗版只供明確選用測試，不能宣稱已理解上下文；見[實驗結果與操作](../team-run/transformer-context/EXPERIMENT.md)。

此工具教學與測試姓名、物品指代、偏好更正及格式延續四種有限任務。資料由本地演算法產生，不使用外部模型，也不直接替模型回答。

## 已凍結資料

預設目錄為 `data/context_dialogues/`，seed為42：

| 檔案 | 監督記錄數 | 用途 |
|---|---:|---|
| train.json | 1,356 | 4類任務，每類339筆 |
| val.json | 200 | 訓練選擇與診斷，每類50筆 |
| test.json | 200 | 最後一次決策，每類50筆 |
| regression.json | 20 | 既有pairs最前20筆，含身份／名稱／能力等單輪回歸 |
| replay.json | 722 | 原pairs剩餘資料，可在微調時回放 |

每個對話保留初始事實回憶與修正後回憶兩個監督輪。先依完整對話、具體模板variant與實體組合分組，才展開記錄；train/val/test三種group及完整輸入的跨split交集均為零。相同任務類型與部分字元仍會共用，不能宣稱完全未見語言泛化。

200筆val或test中有152種獨立的 `(history,prompt)`；不同修正對話會共用初始資訊輪，因此不能把200筆稱為200種獨立輸入。評估同時列出加權率與去重後率，正式決策應看 `unique_input_summary`。

`manifest.json`記錄各檔SHA256、seed、group數、tokenizer詞表hash與UNK統計。現有tokenizer檢查UNK為0，最長prompt156tokens，最長回答含EOS13tokens。資料未新增token或修改embedding。

產生新的資料位置（非空目錄拒絕覆寫）：

```powershell
.\.venv\Scripts\python.exe -X utf8 tranning/context_dialogues.py --out-dir data/context_dialogues_new --seed 42
```

格式使用 `context_format.build_prompt()`，沒有在此工具重複實作角色序列化。每筆含dialogue_id/task/template_id/entity_group/history/prompt/reply/turn_index；缺少前文時的期待答案另列no_history_reply。

## 評估用法

舊模型同多輪格式的公平基準，及候選val比較：

```powershell
.\.venv\Scripts\python.exe -X utf8 tranning/context_evaluate.py --data data/context_dialogues/val.json --model-dir tranning/gpt_chat_runs --report output/transformer-context/baseline_val.json --device cuda
```

只需替換 `--model-dir` 與報告名稱即可評估候選。`--legacy`表示單輪回歸、不強制format；若checkpoint本身是dialogue_v1，仍使用其單輪role格式，不偽裝為舊格式：

```powershell
.\.venv\Scripts\python.exe -X utf8 tranning/context_evaluate.py --data data/context_dialogues/regression.json --model-dir tranning/gpt_chat_runs --report output/transformer-context/baseline_regression.json --device cuda --legacy
```

預設context生成上限20tokens，單輪回歸60；可用 `--max-new-tokens` 指定1至60。生成固定top_k=1、top_p=1、temperature=1、repetition_penalty=1。它只呼叫原始Transformer，不走精確句型或輸出修正。

報告列出：

- `summary`：所有記錄的task平衡macro exact／task_success，及重複輸入數。
- `unique_input_summary`：去除重複輸入的macro exact／task_success，作為主要gate依據。
- `rows`：完整前文、末問、期待答案、模型生成及分數。
- `history_removed`：移除history後，期待變為缺資訊澄清，不能仍要求原姓名。
- `history_swapped`：替換為同末問、不同答案的另一個history，期待答案同步改變。
- `contrastive_success`：原前文與替換前文都答對且答案真的不同，排除只猜固定答案。
- metadata：資料、模型、config、tokenizer詞表及merges、評估程式hash與生成參數。

Exact只作NFKC、外側空白及句尾句點／驚嘆號正規化；task_success另外接受事先固定的少數完整句型，例如「你叫小明」。不以contains把「你不是小明，是小安」判成小明正確；格式任務沒有寬鬆句型別名。

同一固定greedy輸入的生成可快取，避免反事實檢查重複計算；這不會改寫模型答案。`--no-counterfactual`可停用反事實檢查，`--limit`可用於val診斷，但有限樣本結果不視為完整test驗收。

## 誠實限制

模板任務只是上下文能力的起點，不能證明模型普遍理解概念、事實正確或強制遵守指令。單輪回歸集是既有資料，不是獨立世界知識測試；仍需同一預先固定集比較舊模型與候選，不能因基準分低而移除困難題。test與評估器在pilot前凍結，不能根據test結果反覆改模板或放寬評分。
