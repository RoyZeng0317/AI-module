# 資料拆分與分析能力：基礎架構紀錄（2026-11-17）

對應 to_do_list.md **#53**。這一輪先搭基礎架構，細部之後再修。

## 一句話結論

sinco 現在可以對資料檔做**分析**、**train/val/test 拆分**和 **k-fold 分配**。可以用 `/data` 指令，也可以直接講「幫我分析 data/mental.csv」。數字全部由程式計算，不交給模型生成（17M 的模型會亂編統計值）。也沒有呼叫外部 API，符合 Rule 06。

## 怎麼用

| 入口 | 寫法 |
|---|---|
| 桌面 GUI／終端機 CLI／網頁 | `/data analyze data/mental.csv`（輸入 `/data ` 後按 Tab 選 analyze／split／kfold） |
| 直接聊天 | 「幫我分析 data/mental.csv」「把 data/mental.csv 依 score 分層拆分」「data/mental.csv 做 5-fold 交叉驗證」 |
| 命令列 | `python tranning/data_analysis.py analyze data/mental.csv --save-report` |

```
/data analyze <檔案或資料夾> [--target 欄位]
/data split   <檔案或資料夾> [--target 欄位] [--group 欄位] [--ratios 0.7 0.15 0.15] [--seed 42]
/data kfold   <檔案或資料夾> [--k 5] [--target 欄位] [--group 欄位] [--seed 42]
```

- 支援格式：`.csv` `.tsv` `.json` `.jsonl` `.xlsx`（需要 openpyxl）`.txt`，以及影像資料夾（每類一個子資料夾，或已分好 train/val/test）。
- 中文 CSV 會自動判斷 UTF-8 或 Big5／cp950 編碼。
- 輸出一律寫在 `output/data_analysis/`，原始資料不會被改動。可以讀專案外的檔案，但不會寫到專案外（規則 01）。
- 路徑直接寫，不要加 `<>`（PowerShell 會報錯）。路徑有空白時用 `"..."` 包起來。

## 能做什麼

**分析（analyze）**
- 筆數、欄位型別（數值／類別／文字／巢狀）、缺值、唯一值。
- 數值欄位：平均、標準差、四分位數、IQR 離群值。
- 類別欄位：前 10 名。文字欄位：長度範圍。
- 目標欄位：類別分佈、不平衡比、標籤衝突（特徵相同但標籤不同）。不指定時，會自動找 `label`、`class`、`score` 等欄位名。
- 數值欄位之間的 Pearson 相關係數。
- 自動提醒：重複列、缺值多、常數欄、疑似 ID 欄、類別太少無法分層、兩欄幾乎重複。
- 影像資料夾：各類別張數、尺寸取樣；如果已經分割，順便跑 #52 的洩漏檢查。

**拆分（split）**
- 依比例切成 train/val/test；也可以只給兩個比例，切成 train/test。
- `--target`：分層，各類別在每個 split 的比例一致。
- `--group`：同一組（例如同一個使用者）不會跨 split。
- **沒給 `--group` 時，特徵完全相同的列（包括重複列）會自動視為同一組**，所以重複資料不會被拆到 train 和 test 兩邊。
- 切完自動檢查跨 split 的洩漏。影像資料夾沿用 `data_split.py`。

**k-fold**
- 分層＋分組分配，沿用 #52 的 `kfold.py`，輸出每一列分到第幾個 fold。

## 用專案真實資料跑出的結果

各份完整報告都在這個資料夾裡。

| 資料 | 發現 |
|---|---|
| `data/mental.csv`（443 筆） | ⚠️ **45 列完全重複**；score 極度不平衡（4 分 190 筆、0 分只有 1 筆，相差 190 倍）；3、5、0 分都少於 10 筆 → [報告](data_mental_analyze_report.md) |
| ↳ 拆分示範 | 依 score 分層，train/val/test = 309/65/69，**跨 split 洩漏 0 處**。0 分只有 1 筆，只能放在 train → [報告](data_mental_split_report.md)、[輸出](mental_split_demo/) |
| ↳ 5-fold 示範 | 每個 fold 88–89 筆，類別比例一致；0 分只能出現在一個 fold（有提醒） → [報告](data_mental_kfold_report.md) |
| `data/pairs.json`（742 筆） | 沒有重複；回覆長度 2–77 字（跟 #43-1b 的截斷問題一致） → [報告](data_pairs_analyze_report.md) |
| `data/chat_eval_cases.json`（24 筆） | `history`、`requires_network`、`context_required` 缺值 88%；`id` 欄每列都不同 → [報告](data_chat_eval_cases_analyze_report.md) |
| `data/code_pairs.json`（118 筆） | 沒有重複 → [報告](data_code_pairs_analyze_report.md) |
| `data/context_dialogues_v2/train.json`（4098 筆） | 四類任務各約 25%，很平衡，沒有標籤衝突 → [報告](context_dialogues_v2_train_analyze_report.md) |
| `tranning/data.csv`（60 筆） | height 和 weight 的相關係數 0.935 → [報告](tranning_data_analyze_report.md) |
| `data/dataset`（2400 張圖） | 4 類各 500/100 張，尺寸都是 224×224，洩漏 0 處 → [報告](data_dataset_analyze_report.md) |

**建議優先處理**：`mental.csv` 拿去訓練情緒模型前，先處理 45 列重複資料，並決定 score 0、3、5 這幾個樣本極少的類別要合併還是補資料。目前 `/data split` 已經保證重複列不會造成洩漏，但類別太少的問題只能靠補資料解決。

## 架構

```
使用者 ──┬─ 桌面 GUI  lib/components/command.py   _run_data()（背景執行緒）
         ├─ 終端機    lib/components/cli.py        /data
         ├─ 網頁      web/backend/app.py           _run_local_command("/data …")
         └─ 一般聊天  tranning/chats.py            smart_reply_traced() → route_data_request()
                              │
                  lib/components/data_tools.py   run_data()（共用入口）
                              │
                  tranning/data_analysis.py      讀檔／型別推斷／統計／拆分／k-fold／報告
                              ├─ tranning/kfold.py          分層＋分組 fold 分配（#52）
                              ├─ tranning/leakage_check.py  影像洩漏檢查（#52）
                              └─ tranning/data_split.py     影像資料夾拆分（既有）
```

- 自然語言只有在「訊息有分析／拆分／k-fold 的字眼」**而且**「提到一個實際存在的資料檔或影像資料夾」時才會觸發，其他句子照常交給聊天模型。像「我今天心情不好」「分析一下人生」都不會被攔下。
- 自然語言觸發時，回覆的 trace 會寫明「交給 data_analysis.py 計算」，不會假裝是模型想出來的。

## 測試

- `tranning/test_data_analysis.py`：10 項，涵蓋讀檔格式、統計、提醒、分層、分組、重複列不洩漏、輸出不出專案、影像資料夾、指令解析和自然語言路由。
- `lib/components/test_data_tools.py`：4 項，涵蓋 GUI 和 CLI 的指令登記與 Tab 選項。
- `web/backend/tests/test_app.py`：新增 1 項 `/data` 測試。
- 跟既有的 CLI、`chats.py`、網頁測試一起跑，共 **97 項通過**。
- 這台雲端機器的 Python 沒有 tkinter，GUI 的 `test_command.py` 跑不了。GUI 接線改用讀原始碼的方式確認，**還需要你在本機開 GUI 實際打一次 `/data`**。

## 已知限制（之後細修的候選）

1. 型別推斷是規則式的：唯一值 ≤ 20 而且字短 → 類別；全部是數字 → 數值。像 `score` 這種「數字型的類別」會同時給數值統計和類別分佈。日期欄位目前會被當成文字。
2. 自然語言抓目標欄位只認「依 X」「按 X」「目標欄位 X」「--target X」這幾種寫法，比例（例如「八二分」）還不能用口語指定，要用 `/data split ... --ratios 0.8 0.2`。
3. 相關係數只算數值對數值；類別和數值之間（例如 ANOVA、Cramér's V）還沒做。
4. 文字欄位只統計長度，沒有做斷詞或高頻詞。
5. 讀 `.xlsx` 需要另外安裝 openpyxl；大檔（數十萬列）是純 Python 計算，速度沒有特別優化。
6. 分析結果只是描述統計，不會自動幫你清理資料（例如去重、補缺值），避免在你不知道的情況下改到資料。

## 檔案清單

新增：`tranning/data_analysis.py`、`lib/components/data_tools.py`、`tranning/test_data_analysis.py`、`lib/components/test_data_tools.py`、本資料夾的報告與示範輸出。
修改：`tranning/chats.py`（加一段路由）、`lib/components/command.py`、`lib/components/cli.py`、`web/backend/app.py`、`web/backend/tests/test_app.py`。
