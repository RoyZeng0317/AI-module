# 專案檔案分類整理計畫

盤點日期：2026-11-19

## 盤點結論

- 核心分層已存在：`lib/` 是介面與應用外殼、`tranning/` 放模型與訓練、`data/` 放資料、`web/` 放網站、`output/` 放產出、`memory/` 放應用記憶。
- 根目錄混有教學練習、獨立小專案、報告、模型權重、資料集壓縮檔與執行暫存檔。README 的檔案樹標示為 2026-09-21 快照，且包含已搬移的舊路徑。
- `tranning/` 有166個直接 `.py` 檔、合計216個直接項目；`data/` 有42個直接項目；`output/` 有120個直接項目，其中106個是子目錄。分類需按「模型領域／資料用途／任務輸出」建立導覽，而非只看副檔名。
- 工作樹已有大量未提交修改，包含 `lib/`、`tranning/`、`data/`、`output/`、`web/`；整理時必須保留原狀，不能把它們當成雜物清除。
- 程式與文件多處直接引用 `data/`、`tranning/`、`output/`、`CLAUDE.md`；搬動前需搜尋 Python 匯入、CLI、GUI、訓練設定、文件連結及 Git 忽略規則。

## 建議目標分類

保留現有核心目錄名稱與入口，避免一次改寫程式路徑：

```text
AI-module/
├── app/                 # 安裝器與提示詞指令內容
├── lib/                 # GUI、CLI、共用前端、推論操作外殼
├── tranning/            # 模型、訓練、評估與模型專屬工具
│   ├── language/        # 聊天、Transformer、詞義、推理
│   ├── vision/          # OCR、分類、物件／道路／電路影像
│   ├── audio/           # STT、語音合成、音訊資料處理
│   ├── domain/          # 電路、數學、PCB 等領域模型
│   └── shared/          # tokenizer、資料切分、共用工具
├── data/                # 依任務／領域分資料集；原始、草稿、切分資料分開
├── models/              # 可下載／共用權重與來源清單；訓練 checkpoint 留在 run 目錄
├── output/              # 任務輸出，沿用既有任務子目錄保存
├── docs/                # 教材、專題報告、操作說明；專案規則與歷史索引暫留根目錄
├── examples/            # 基礎練習及非正式範例程式
├── Agent/               # 歷史錯誤與已完成紀錄
├── team-run/            # 任務計畫、審查與協作證據
└── web/                 # 網站前後端與管理介面
```

`tranning/` 子目錄是長期目標，不表示本輪可直接搬動。若搬動會破壞舊匯入路徑，先新增相容匯入層或分批改引用，再移除舊路徑；模型 run/checkpoint、資料集、使用者報告及工作記錄均不刪除。

## 分階段處理

### 第 1 階段：建立分類依據（已完成）

- 新增本計畫、README 導覽及完整 Markdown 檔案清冊 `docs/MARKDOWN_CATALOG.md`。
- 不碰 `.env`、資料內容、模型權重、使用者報告或既有未提交程式。
- Office 鎖定檔、用途不明的 `.csv`、根目錄 `.pt`／`.task` 權重只列入待判定，不自行刪除或搬走。

### 第 2 階段：根目錄逐檔映射

- 確認根目錄程式、文件、模型資產與壓縮資料集的用途、Git 狀態及引用位置。
- 先列出「純文件」及「不被程式引用的教學範例」之明確目的地；保留獨立小專案原有入口。
- 路徑有相依或來源不明時保持原位，標記用途，不猜測。

### 第 3 階段：訓練與資料目錄分區

- 按 NLP／影像／音訊／領域任務建立索引或子目錄；每批搬動前更新 import、CLI、GUI、文件及資料設定路徑。
- 優先整理新檔案和沒有既有引用的資料，不先改動穩定使用的核心入口。
- 每批完成後做唯讀路徑搜尋、受影響模組匯入檢查及指定回歸驗證。

### 第 4 階段：更新專案索引

- 以實際目錄產生精簡核心樹狀索引，排除 `.git`、`.venv`、大型資料與 checkpoint 詳細清單。
- 標示路徑相容例外、獨立子專案、草稿資料與訓練成果的保存位置。

## 2026-11-19 遷移進度

- **DOCX 文件類**：10份報告、教材與商業規劃已依內容移入 `docs/reports/`、`docs/tutorials/`、`docs/business/`；SHA-256 搬移前後一致，Office 鎖定檔未動。
- **教學範例類**：`baisc/`、`martix/`、`jieba/` 與兩個根目錄範例已移入 `examples/`；CNN-MNIST 範例根目錄相對路徑已更新。檔案雜湊保持一致，僅路徑計算程式碼有必要變更。
- **Markdown 類**：先建立全檔 AI 索引，再將12份課程筆記整組移到 `docs/tutorials/重點筆記/`，部署手冊移到 `docs/deployment/`；筆記相對連結保持有效，3處原始碼說明引用已更新。
- 專題測試說明與圖片、資料、Word 報告共置，保留原位；`team-run/` 任務證據、`output/` 測試 fixture、資料集與模型說明仍依原目錄保留。
- 工作樹的其他既有修改未清理或覆寫。模型權重、資料集、模型訓練程式及 `tranning/` 扁平模組尚未遷移。

## 根目錄第一輪分類映射

以下是目的地建議，不代表已搬動。凡標記「先查引用」者，需等路徑依賴確認後再改：

| 現有項目 | 建議分類／目的地 | 整理方式 |
|---|---|---|
| `AGENTS.md`、`README.MD`、`CLAUDE.md`、`to_do_list.md`、`ErrorLog.md`、`UserPersonality.md` | 根目錄專案入口文件 | 保留在根目錄，避免 Codex、`/init`、待辦及既有連結失效；統一說明誰是權威規則與索引。 |
| `Agent/`、`team-run/` | 專案工作紀錄 | 維持分開：`Agent/` 留歷史／已完成紀錄，`team-run/` 留當前任務計畫與證據。 |
| `docs/reports/`、`docs/tutorials/`、`docs/business/` | 文件分類完成 | 10份 DOCX 依內容歸檔，檔案內容雜湊不變。 |
| `examples/python_basics/`、`examples/matrix/`、`examples/word_segmentation/`、`examples/vision/`、`examples/legacy_external_model/` | 教學與封存範例分類完成 | 原始練習依類型移動；MNIST 範例的根路徑計算已更新；外部模型示範明確標記為封存教材。 |
| `docs/tutorials/重點筆記/`、`docs/deployment/前端部屬scp.md` | Markdown 文件分類完成 | 12份相互連結課程筆記整組遷移；部署手冊遷移後更新3處有效原始碼引用。 |
| `MNIST_Dataset/`、`mnist_test/`、`sinco-emotion-training-data*` | `data/` 下具名資料集目錄（候選） | 先讀各自 README、manifest、授權／來源與忽略規則；大型資料、封存檔保持原狀直到用途清楚。 |
| `yolo11n.pt`、`yolo26n.pt`、`face_landmarker.task`、`pose_landmarker.task` | `models/`（候選） | 多處硬編碼根目錄權重；移動前需統一權重解析器並更新打包腳本、訓練器、GUI、README、遷移清單。 |
| `output/`、`runs/`、根目錄訓練 log | 任務產出／可重建紀錄 | 不混入原始資料或程式碼；先辨認是否為唯一訓練證據，再按任務歸檔，不能通用清理。 |
| `.csv`、`result.png`、`audio_step1.png`、`model_icon.ico` | 用途待核對 | 目前名稱不足以判斷是資料、結果或 UI 資產；檢查來源／引用前保持原位。 |
| `~$智慧模型-Sinco研發報告.docx`、`~WRL3179.tmp`、特殊字元檔名 PowerShell 腳本 | 暫存／來源待核對 | 不自動刪除或改名；先確認是否有程式仍在使用及檔案實際內容。 |
| `package.json`、`package-lock.json`、`pyrightconfig.json`、`.gitignore`、`.gitattributes`、`LICENSE` | 根目錄工具設定 | 暫時保留根目錄；Node 與 Python 工具會由根目錄讀設定。 |
| `check_GUI/`、`materials_recongizion/`、`猜拳/`、`shape-vision/` | 獨立小專案 | 保留各自資料夾及入口，README 索引補上用途；不併入 `lib/` 或 `tranning/`。 |

另有根目錄 `data_split.py` 為多支訓練與 GUI 共用入口，應保留現況；不要跟 `tranning/` 中同名資料切分工具因檔名相同就合併。
