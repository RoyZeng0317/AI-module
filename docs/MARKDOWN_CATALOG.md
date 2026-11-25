# Markdown 檔案分類清冊

供其他 AI／開發工具使用的全專案 Markdown 索引。此清冊列出路徑、用途與分類狀態，不取代原始文件內容。

## 清冊範圍

- 更新日期：2026-11-19。
- 現有來源 Markdown：289 份；副檔名比對不分大小寫。
- 排除 `.git/`、`.venv/`、其他虛擬環境、`node_modules/`、`build/` 及工具快取。
- `team-run/` 即使在 `.gitignore` 內仍列入；它是任務證據，不能因 Git 忽略而刪除或移出專案。
- `output/` 下由 pytest 建立的 Markdown fixture 也列入，但標記為產生物，不是正式文件。
- 清冊本身不列入來源數，避免自我列舉。

## 狀態代碼

| 狀態 | 意義 |
|---|---|
| `KEEP_*` | 依功能與路徑相依性保留於現行目錄。 |
| `REVIEW` | 用途或連結相依仍需確認，先不移動。 |

## 已遷移路徑對照

| 遷移前 | 目前位置 | 說明 |
|---|---|---|
| `重點筆記/` | `docs/tutorials/重點筆記/` | 整組移動12份筆記，保留相對章節連結。 |
| `前端部屬scp.md` | `docs/deployment/前端部屬scp.md` | 更新3處原始碼說明引用。 |
| 根目錄10份 DOCX | `docs/reports/`、`docs/tutorials/`、`docs/business/` | 文件分類批次；DOCX 不屬於本 Markdown 檔案數。 |
| `baisc/`、`martix/`、`jieba/` 與根目錄教學程式 | `examples/` | 練習類別遷移；另見 `examples/README.md`。 |

## 分布（目前路徑）

| 頂層位置 | Markdown 數量 |
|---|---:|
| `[root]/` | 6 |
| `Agent/` | 14 |
| `app/` | 13 |
| `data/` | 18 |
| `docs/` | 15 |
| `examples/` | 1 |
| `lib/` | 5 |
| `machine_migration/` | 1 |
| `models/` | 1 |
| `output/` | 38 |
| `Rule/` | 1 |
| `shape-vision/` | 1 |
| `sinco-emotion-training-data/` | 4 |
| `sinco-emotion-training-data-2/` | 4 |
| `team-run/` | 159 |
| `tranning/` | 4 |
| `web/` | 3 |
| `專題測試結果/` | 1 |

## 逐檔分類

### 歷史錯誤、成功紀錄與索引（14）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `Agent/ErrorLog/Failed/2026-08-13-函式庫無法匯入.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Failed/2026-08-24.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Failed/2026-08-27-web前端無法使用terminal進行指令啟動.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Failed/2026-08-28-web前端的主控台錯誤.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Failed/2026-09-29-Sinco過擬合問題.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Failed/2026-10-08-emotion-dataset-import.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Success/2026-08-13-lib_main匯入路徑修正.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Success/2026-08-13-prompt_toolkit匯入錯誤修正.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Success/2026-09-27-jieba_main多項錯誤修正.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Success/2026-10-25-YOLO物品標記與小資料訓練修正.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ErrorLog/Success/ErrorLog.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/Index/2026-08-13-Project Advise.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/Index/2026-10-08-emotion-dataset-import.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |
| `Agent/ToDoList/2026-10-28.md` | 歷史錯誤、成功紀錄與索引 | `Agent/` | `KEEP_ARCHIVE` |

### 專案規則與根目錄入口（6）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `AGENTS.md` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |
| `CLAUDE.md` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |
| `ErrorLog.md` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |
| `README.MD` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |
| `to_do_list.md` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |
| `UserPersonality.md` | 專案規則與根目錄入口 | `根目錄` | `KEEP_ROOT` |

### 應用提示詞／斜線指令內容（13）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `app/command/Agent_Workflow.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/AI_Assistant.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/Architecture.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/BugFix.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/CodeReview.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/git_workfkow.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/memorize.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/PR.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/release.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/rules.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/Security.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/system.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |
| `app/command/task.md` | 應用提示詞／斜線指令內容 | `app/command/` | `KEEP_WITH_APP` |

### 資料集、教材來源與資料評估（18）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `data/basic_electricity/PILOT_PROPOSAL.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/basic_electricity/README.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/circuit-theory/Circuit Theory-Circuit-Element.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/circuit-theory/Circuit Theory-Circuit-Variables.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/circuit-theory/Simple-Resistive-Circuits.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/electricity_open/NODAL_MESH_GUIDE.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/electricity_open/PILOT_V1_RESULT.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/electricity_open/README.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/Emotion_datasets/README.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/imagegen_shapes/COLLAB_REVIEW_2026-10-30.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/imagegen_shapes/PILOT_V1_RESULT.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/imagegen_shapes/PILOT_V2_PROPOSAL.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/imagegen_shapes/PILOT_V2_RESULT.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/imagegen_shapes/README.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/Linear/ch1.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/Linear/ch2.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/Linear/ch3.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |
| `data/word_sense/README.md` | 資料集、教材來源與資料評估 | `data/` | `KEEP_WITH_DATA` |

### 部署操作手冊（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `docs/deployment/前端部屬scp.md` | 部署操作手冊 | `docs/deployment/` | `KEEP_WITH_DOCS` |

### 專案架構與文件索引（2）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `docs/PROJECT_LAYOUT.md` | 專案架構與文件索引 | `docs/` | `KEEP_CENTRAL_DOCS` |
| `docs/README.md` | 專案架構與文件索引 | `docs/` | `KEEP_CENTRAL_DOCS` |

### AI／機器學習課程筆記（12）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `docs/tutorials/重點筆記/00_學習地圖與基礎.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/01_決策樹.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/02_梯度下降.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/03_CNN.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/04_RNN.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/05_LSTM.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/06_Transformer.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/07_RAG.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/08_OCR.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/09_總整理與測驗.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/10_交叉商損失.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |
| `docs/tutorials/重點筆記/README.md` | AI／機器學習課程筆記 | `docs/tutorials/重點筆記/` | `KEEP_TUTORIAL_SET` |

### 教學練習／封存範例索引（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `examples/README.md` | 教學練習／封存範例索引 | `examples/` | `KEEP_EXAMPLES` |

### 程式模組操作與功能說明（5）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `lib/IMAGE_GENERATION.md` | 程式模組操作與功能說明 | `lib/` | `KEEP_WITH_MODULE` |
| `lib/opencv2/IMAGE_PROCESSING.md` | 程式模組操作與功能說明 | `lib/` | `KEEP_WITH_MODULE` |
| `lib/RAG/README.md` | 程式模組操作與功能說明 | `lib/` | `KEEP_WITH_MODULE` |
| `lib/yolo/CHARACTERS.md` | 程式模組操作與功能說明 | `lib/` | `KEEP_WITH_MODULE` |
| `lib/yolo/README.md` | 程式模組操作與功能說明 | `lib/` | `KEEP_WITH_MODULE` |

### 換機遷移清單與檔案清冊（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `machine_migration/MIGRATION_MANIFEST.md` | 換機遷移清單與檔案清冊 | `machine_migration/` | `KEEP_WITH_MIGRATION` |

### 本機模型卡與權重說明（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `models/bge-small-zh-v1.5/README.md` | 本機模型卡與權重說明 | `models/` | `KEEP_WITH_MODEL` |

### 分析、評估與任務輸出（4）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `output/chat-eval-2026-11-17/ASSESSMENT.md` | 分析、評估與任務輸出 | `output/` | `KEEP_WITH_OUTPUT` |
| `output/transformer-audit/REVIEW.md` | 分析、評估與任務輸出 | `output/` | `KEEP_WITH_OUTPUT` |
| `output/transformer-audit/STATUS.md` | 分析、評估與任務輸出 | `output/` | `KEEP_WITH_OUTPUT` |
| `output/user_circuit_dataset/README.md` | 分析、評估與任務輸出 | `output/` | `KEEP_WITH_OUTPUT` |

### 測試產生的 Markdown fixture（34）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `output/context-core-final-tests/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_chat_export_handles_empty0/empty.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_chat_with_explicit_path_s0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_chat_without_arg_opens_sa0/picked.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_init_with_explicit_folder0/some_project/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_init_without_arg_opens_fo0/picked/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_run_chat_with_explicit_pa0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-final-tests/test_run_chat_without_arg_uses0/output/chats/chat_20261026_003552.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-core-tests/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_chat_export_handles_empty0/empty.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_chat_with_explicit_path_s0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_chat_without_arg_opens_sa0/picked.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_init_with_explicit_folder0/some_project/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_init_without_arg_opens_fo0/picked/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_run_chat_with_explicit_pa0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests-final/test_run_chat_without_arg_uses0/output/chats/chat_20261026_003435.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_chat_export_handles_empty0/empty.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_chat_with_explicit_path_s0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_chat_without_arg_opens_sa0/picked.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_init_with_explicit_folder0/some_project/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_init_without_arg_opens_fo0/picked/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_run_chat_with_explicit_pa0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-frontends-tests/test_run_chat_without_arg_uses0/output/chats/chat_20261026_003406.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_chat_export_handles_empty0/empty.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_chat_with_explicit_path_s0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_chat_without_arg_opens_sa0/picked.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_init_with_explicit_folder0/some_project/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_init_without_arg_opens_fo0/picked/CLAUDE.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_run_chat_with_explicit_pa0/exported.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/context-review-tests/test_run_chat_without_arg_uses0/output/chats/chat_20261026_003926.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/xss-review-tests/regression-final/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/xss-review-tests/regression/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |
| `output/xss-tests/pytest-app-regression/test_chat_endpoint_preview_com0/note.md` | 測試產生的 Markdown fixture | `output/` | `KEEP_GENERATED` |

### 基礎規則文件（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `Rule/Basic.md` | 基礎規則文件 | `Rule/` | `KEEP_RULES` |

### 獨立子專案文件（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `shape-vision/README.md` | 獨立子專案文件 | `shape-vision/` | `KEEP_SUBPROJECT` |

### 匯入資料集封存包文件（8）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `sinco-emotion-training-data-2/Agent/ErrorLog/2026-10-08-emotion-dataset-import.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data-2/Agent/Index/2026-10-08-emotion-dataset-import.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data-2/data/emotion_datasets/README.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data-2/README.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data/Agent/ErrorLog/2026-10-08-emotion-dataset-import.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data/Agent/Index/2026-10-08-emotion-dataset-import.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data/data/emotion_datasets/README.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |
| `sinco-emotion-training-data/README.md` | 匯入資料集封存包文件 | `資料集封存目錄` | `KEEP_WITH_DATA_ARCHIVE` |

### 任務計畫、研究、審查與執行紀錄（159）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `team-run/basic-electricity/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-electricity/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-electricity/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-electricity/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-electricity/TRAINER_RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-reasoning-svg/CLOUD_HANDOFF.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-reasoning-svg/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/basic-reasoning-svg/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/BUILD_CHAT.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/BUILD_SEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/BUILD_UI.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/character-detection/BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/character-detection/DETECTION_FIX.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/character-detection/IMAGE_GUI_BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/character-detection/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/chat-layout/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/chat-layout/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/chat-layout/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/chat-layout/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/circuit-blocks/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/circuit-theory/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/BUILD_A.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/BUILD_B.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/NODAL_MESH_GUIDE.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/PILOT_V1_FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/PILOT_V1_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/electricity-open/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/grok-claude-ui/STATUS.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-processing/BUILD_CONVOLUTION.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-processing/BUILD_HISTOGRAM.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-processing/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-processing/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-processing/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/LIVE_REPLY.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/PROBE.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-search/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-web/BUILD_BACKEND.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-web/BUILD_FRONTEND.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-web/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-web/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/image-web/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/BUILD_EVALUATION.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/BUILD_MODEL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/PILOT_PROPOSAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/PILOT_V2_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes-v2/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/BUILD_DATA.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/BUILD_MODEL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/PILOT_PROPOSAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/PILOT_V1_RESULT.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/PILOT_V1_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/imagegen-shapes/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/learning-office/BUILD_DATA.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/learning-office/BUILD_PREPARE.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/learning-office/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/learning-office/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/learning-office/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/linear-theory/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/DELETE_CHANGES.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/DELETE_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/TRAIN_FIX_BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/TRAIN_FIX_FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/TRAIN_FIX_PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/TRAIN_FIX_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/object-labeling/TRAINING_COMPARISON.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/BREADBOARD_BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/BREADBOARD_PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/BREADBOARD_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/CIRCUIT_BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/CIRCUIT_PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/CIRCUIT_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/DATASET_BUILD.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/DATASET_FINDINGS.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/DATASET_PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/DATASET_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/README.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/photo-edit/VIDEO_OBSERVATIONS.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/MODEL_BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/MODEL_ROADMAP.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/PRIVACY_DRAFT.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/REQUIREMENTS.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-enforcement/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-signals/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-signals/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-signals/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-signals/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/region-signals/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/REVIEW_ROUND1.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/ALIGNMENT_REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/ALIGNMENT_V1_PROPOSAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/ALIGNMENT_V1_RESULT.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/BILLING_PROPOSAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/MARKDOWN_LEARNING.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/speech-recognition/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/timezone-unicode/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/timezone-unicode/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/timezone-unicode/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/timezone-unicode/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/timezone-unicode/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/BUILD_CORE.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/BUILD_DATA.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/EXPERIMENT.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/transformer-context/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/tw-road/SOURCES.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/tw-road/STATUS.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_collect_code_pairs_reads_0/a/one.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_collect_code_pairs_reads_0/two.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_collect_md_chat_reads_fol0/chat.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_collect_md_chat_reads_fol0/empty.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_extract_source_tree_mirro0/md/pkg/math_utils.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/.pytest-full-temp/test_extract_then_pairs_round_0/md/sort.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/BUILD_CORE.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/BUILD_DATA.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/DATASET.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/word-sense/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/BUILD_BACKEND.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/BUILD_FRONTEND.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/BUILD_LOG.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/FINAL.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/PLAN.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/RESEARCH.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |
| `team-run/xss/REVIEW.md` | 任務計畫、研究、審查與執行紀錄 | `team-run/` | `KEEP_TASK_LOCAL` |

### 模型訓練教材、架構與領域說明（4）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `tranning/context_dialogues.md` | 模型訓練教材、架構與領域說明 | `tranning/` | `KEEP_WITH_TRAINING` |
| `tranning/data/English.md` | 模型訓練教材、架構與領域說明 | `tranning/` | `KEEP_WITH_TRAINING` |
| `tranning/data/Quartus-II-鍵盤電路設計教學.md` | 模型訓練教材、架構與領域說明 | `tranning/` | `KEEP_WITH_TRAINING` |
| `tranning/RNN.md` | 模型訓練教材、架構與領域說明 | `tranning/` | `KEEP_WITH_TRAINING` |

### 網站、後端與網頁子系統文件（3）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `web/backend/REGION_SIGNALS.md` | 網站、後端與網頁子系統文件 | `web/` | `KEEP_WITH_WEB` |
| `web/image_processing/README.md` | 網站、後端與網頁子系統文件 | `web/` | `KEEP_WITH_WEB` |
| `web/SECURITY.md` | 網站、後端與網頁子系統文件 | `web/` | `KEEP_WITH_WEB` |

### 專題測試結果與相鄰圖片／資料（1）

| 目前路徑 | 用途分類 | 建議位置／歸屬 | 狀態 |
|---|---|---|---|
| `專題測試結果/旋積運算/for_loop_results/Filters與結果說明.md` | 專題測試結果與相鄰圖片／資料 | `專題測試結果/` | `KEEP_WITH_ASSETS` |

## 遷移規則

1. 不批次合併、刪除或改寫原始 Markdown；重複主題先保留各版本。
2. 治理文件、提示詞、模組 README、資料集說明、模型教材、子專案文件、歷史紀錄、任務證據與產生的測試 fixture 各自留在目前功能目錄。
3. 搬動單一文件前搜尋程式碼、README、Markdown 相對連結與圖片／資料引用；搬移後更新活躍引用並核對 SHA-256。
4. 若 Markdown 與圖片、資料或其他附件共置，優先保留整組關係；除非能一併規劃資產遷移，不單獨抽走 Markdown。
