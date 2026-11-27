# 一般中英訓練與 Office 研究

## Findings

信心：高（程式與原始資料直接核對）；未修改 pairs、權重或 .env，未啟動訓練。已於檔案出現後讀取本輪 PLAN。

- `data/pairs.json` 確為 742 筆、合計 prompt/reply 34,721 字元；全數具有非空字串 prompt/reply，無完全相同 prompt。此僅格式 audit，不能證明內容正確或語意無重複。
- 61 筆 prompt 含英文且不含中文字，其中 4 筆 reply 含中文：`hi`、`hello`、`bye`、`give me an example of the svoo pattern`。這是可重現的語言配對污染，宜於新衍生集修正，保留原 pairs。英文覆蓋量約 8.2%，不足以把整體一般問答當作可靠雙語能力。
- 目前唯一 `pairs*val*.json` 為 `data/pairs_imported_val.json`，僅 2 筆，與原 pairs prompt 精確交集為 0；不足以證明語意／來源群組互斥。`dataset_import.write_manifest` 只精確物件去重並逐筆 random.Random(seed) 切分；`data_split.py` 是影像用，不適用聊天。同義、翻譯、同答案與模板衍生題必須先共用 group_id 再切分。
- `transformer_chat.finetune` 接受 JSON list 的 prompt/reply；額外 metadata 不作為模型輸入（ChatSFTDataset 取 prompt/reply）。可保留 source/topic/group_id 於 sidecar manifest，或資料列額外欄位，但須測試 consumer 的嚴格程度。`build_corpus.py` 抓百科正文做 pretrain，不能取代 SFT 問答，也不要把 eval 加入 corpus。
- CLI 已支援 `finetune --data ... --val-data ... --pretrain-dir ... --out-dir ...`。有 val-data 時整份 data 用於 train；若 val-data 空白會靜默回落 train 作 val，必須在前處理拒絕空驗證集。未傳 val-data 時 random.shuffle + row-level val_split，無 group 保證，且入口未見固定 seed 參數。
- 新 run 安全做法：指向既有 pretrain 或既有 finetune 作初始化，輸出不同目錄，例如 `tranning/gpt_general_learning_runs`；要清楚記錄初始化來源，不能將 continued finetune 說成 optimizer state 完整恢復。預設 CLI freeze embeddings 與底 4 層、label smoothing 0.1、typo noise 0.3；batch 8，可配置 gpu-mem-fraction、epochs、patience、checkpoint-every。`chat --out-dir 新目錄` 可獨立驗證；正式 Web route 仍指預設 checkpoint，訓練新 run 不等於自動上線。
- Office 未見 tools/terminal_exec/command 的 docx/openpyxl/pptx 實作；父代理已確認 .venv 缺相關套件與 win32com。新增 Office 教學問答能描述操作步驟，不能產生真實操作能力。檔案方向需 local tools 建立/修改、重新讀取並驗證；桌面方向需可用原生控制工具與 Office 實際安裝，不能由模型自行聲稱完成。

## Conflicts

- 原訓練集英文 prompt 混中文答覆與使用者「全英文輸入須全英文 A2+」規則衝突。保留原檔，在衍生集明確記錄四筆替換理由，其他中文含專有名詞不應盲目翻譯。
- 742 筆格式正確不等於一般人類問答覆蓋充分。新增同義句也不能以筆數增加當知識增加；需多主題、不同推理／澄清／操作情境與真正 held-out family。
- 虛擬貨幣市場與法規會變，不宜寫固定買點、保證報酬、特定平台合法性或短期價格。來源中的美國法規只可標示其管轄，不套用台灣。
- 正式訓練授權與 Office 形式仍待使用者回答，資料可先備妥，不擅自啟動該相依工作。

## Plan Impact

1. 新增獨立原創 bilingual 資料與 audit；題組覆蓋日常、概念、學習規劃、問答澄清、crypto 投資教育與 Office 能力邊界。英文短句 A2+，繁中資料保留台灣用詞。
2. Crypto 原創內容可教：先理解區塊鏈／錢包／私鑰與保管，再學波動、流動性、費用、詐騙辨識與模擬記帳；投資決策以自身可承受損失、目標與預算為前提，不給必買標的或收益承諾。SEC 官方教育警示指出 crypto asset securities 可能有劇烈波動、投機與平台保護不足，並提醒 speculative investment 需能承受全部損失。[Investor.gov 風險教育（2023-03-23）](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/crypto-asset-securities)。[官方 Crypto Assets 入口](https://www.investor.gov/crypto-assets) 提供持續學習入口。本研究當日即時開啟核對；上述學習順序為本專案的原創教學安排。
3. 以 group_id 固定 seed 切 train/val，並保持獨立 eval 排除。輸出 hashes/source counts/language flags/group intersections；除了 exact prompt，亦對 normalized prompt、相同 answer、翻譯組與近似題人工抽查。
4. 準備新 run 指令但不執行：`python tranning/transformer_chat.py finetune --data <new_train.json> --val-data <new_val.json> --pretrain-dir tranning/gpt_chat_runs --out-dir tranning/gpt_general_learning_runs --batch-size 8 --lr 0.00003 --weight-decay 0.05 --epochs 50 --patience 8`。最終 epochs/資源上限須以使用者答覆為準，先驗證輸出目錄不存在。訓練前先做 tokenizer 截斷長度 audit，避免長 Office 教學被 block_size 截斷。
5. Office 工具需回傳 success/path/error 等結構狀態，只有實際成功且產物存在／可重讀才回「已完成」，無依賴／無權限／輸入不足則如實回報。不可讓訓練語料中的成功話術搶先於工具執行。
