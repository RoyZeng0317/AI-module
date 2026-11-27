# AI Module

## 使用者確認的進度與品質狀態（2026-10-28）

| 項目 | 狀態 |
|---|---|
| 二值化 | 尚未進行 |
| 卷積（Convolution） | 已新增純for迴圈獨立範例，測試通過；效果待驗收 |
| 直方圖／均衡化 | 已新增Y/Cb/Cr、Y直方圖與CDF均衡化範例；效果待驗收 |
| Transformer | 表現未達標，仍需改善 |
| 其餘模型／功能 | 表現待定，尚未完成效果驗收 |

以上依使用者本次確認記錄。既有程式中的門檻處理、卷積層或相關範例，不作為上述專項已完成的證據；工程測試通過也不等於實際效果已通過驗收。後續待辦見[進度清單](to_do_list.md)。

檔案核對：使用者指出的 `CNN-MNIST_Classification.py` 目前實作灰階MNIST的CNN分類，含 `Rescaling(1./255)` 正規化與 `Conv2D` 卷積層；繪製的是accuracy／loss曲線，未見直方圖均衡化運算。匯入 `cv2` 不代表已執行均衡化。該檔本次沒有修改或啟動訓練；使用者後續確認另做獨立Lenna練習。

## for迴圈影像處理（2026-10-28）

新增 `lib/opencv2/convolution.py` 及 `histogram_equalization.py`：依練習文件處理MatrixA與三種卷積核、RGB／灰階圖、Y/Cb/Cr及Y的直方圖均衡化。像素運算皆用Python迴圈，不使用NumPy或內建濾鏡。已用使用者提供的兩張WebP產生卷積結果，並對RGB圖產生均衡化前後對照；保持原尺寸，原始帶符號矩陣與顯示映射分開。20項測試及獨立審查通過，操作與限制見[影像處理說明](lib/opencv2/IMAGE_PROCESSING.md)。

## 開放式電學與節點／網目教學（2026-10-28）

共用聊天後端可用本機公式回答單一正值電阻的歐姆定律題，例如「一個100歐姆的電阻，供應20V的電壓，其電流為多少？使用歐姆定律進行計算」回覆 `I=V/R=20/100=0.2 A（200 mA）`。處理紀錄與正文會標示不是模型生成；交流、模糊接線、缺值與不支援單位不猜答。「說明節點電壓法與網目法的轉換」會展示固定接線的 KCL／KVL 及支路電流驗算，不代表可解析任意電路圖。

Transformer 原始盲答仍有偏題。42題開放式候選資料（22 train／10 dev／10 test）已經核准完成首輪3 epochs／66次更新，另存新權重；但dev與首次test均0/10，未切換正式模型。使用題目作輸入，正確步驟作評分／監督目標。操作與結果見[開放式電學說明](data/electricity_open/README.md)及[首輪報告](data/electricity_open/PILOT_V1_RESULT.md)，轉換教學見[節點與網目範例](data/electricity_open/NODAL_MESH_GUIDE.md)。此生成式方案與尚未完成審查的舊四選一草稿 `electricity_train.py` 不同；不以工具結果、工程測試或loss下降宣稱模型已學會。

## 基本電學題目與評分隔離（2026-10-27）

使用者提供的43份原檔已只讀盤點，人工核對24題無圖選擇題，按來源群切為train18／dev3／test3。`tranning/electricity_dataset.py`準備版本化資料；`electricity_practice.py attempt`只給題目、直接呼叫本機GPT並先存原始作答，另一個`grade`指令才讀評分鍵，只有train能產生不含正解的糾錯提示。

真實基準train9＋dev3共12題皆偏題／未輸出合法唯一答案，0/12；不是已學會電學。test未作答，沒有更新正式權重或改既有使用者修改。操作、來源限制與待核准下一步見[基本電學說明](data/basic_electricity/README.md)。

## 網路圖片搜尋與中文連結（2026-10-26）

CLI 可直接輸入「你尋找有關周柯宇的圖片」「搜尋周柯宇的照片」或 `find images of cats`。共用後端查公開圖片資料、下載最多八張候選，以現有 `tranning/yolo_runs/custom/weights/characters.pt` 篩選同名人物類別；每輪最多回傳三張圖片與來源，Python CLI 另開獨立圖片預覽，桌面 GUI 可點圖片附件標籤預覽。

圖片搜尋使用 360 公開資料與 Bing 備援，不使用雲端 AI、不訓練或改寫權重。搜尋供應商、圖片權限與格式可能變動，失敗會明確提示。YOLO 分數只是預測；零目標匹配、未知姓名或模型不可用時，仍可顯示明確標示「未經人物模型確認」的搜尋候選，不能當成身份辨識成功。本次真實網圖測試取得三張候選，但人物模型未匹配周柯宇。

來源網址中的 UTF-8 中文會顯示為可讀中文，例如 `https://zh.wikipedia.org/wiki/周柯宇`；點擊目標保留原始網址，不改動 `%2F`、`%26` 等保留編碼或簽名。圖片快取與預覽錯誤紀錄放在 `output/image-search/`，不會自動加入訓練資料或清理其他對話附件。既有「辨識圖片＋本機路徑／網址」功能保留。

獨立預覽目前支援 Python CLI；封裝執行檔會顯示提示並保留圖片連結。API 回覆沿用文字／Markdown 與原始圖片來源，不新增對外分享本機快取的服務。

## 上下文字詞語意（2026-10-26）

本次使用者明確允許本機中文預訓練資源作為原「從零訓練」規則的例外，沿用自建 GPT 聊天訓練，不使用雲端 AI。新增獨立的 `ContextualEncoder`：句向量採 CLS，字詞向量依明確字元區間抽取完整句子的雙向上下文表示，供分類與語意搜尋重用。預訓練向量不會自動改變原聊天模型的權重。

完整操作、資料與待核准訓練指令見 [字詞語意說明](data/word_sense/README.md)。準備選用模型時，在專案根目錄手動執行：

```powershell
.\.venv\Scripts\python.exe tranning/prepare_contextual_model.py
```

此明確下載步驟使用 [BAAI/bge-small-zh-v1.5 官方模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)，僅存入 `models/bge-small-zh-v1.5`，記錄來源 revision；推論只讀本機，沒有隱性下載。此次環境拒絕下載（WinError 10013），尚未取得真實權重或驗證消歧品質。離線隨機小模型測試只驗證程式。38 筆人工示範含蘋果、鼠與病毒的異義、否定、後置線索及資訊不足；不能代表一般中文理解。原 GPT 的 9 筆 val 原始回覆已存於 `output/word-sense/chat_val_baseline.json`，明顯偏題，尚未學會本次消歧任務。正式訓練仍依既有逐次確認約定，尚未啟動。

私人自建 AI 專案。核心規則：**禁止使用其他 AI 模型或雲端 API**（不呼叫 Claude／GPT／任何 LLM API），所有模型都自己從零設計、從零訓練，算力上限抓 **RTX 4060 8GB**，目前僅供個人使用，不考慮對外公開。完整規則見 [`CLAUDE.md`](CLAUDE.md)。

## 使用大綱

三個前端外殼共用同一套對話邏輯（`tranning/chats.py` 的 `smart_reply()`），差別只在介面：

| 前端 | 啟動指令 | 說明 |
|---|---|---|
| 桌面 GUI | `python -m lib.main`（在專案根目錄執行） | Tk 主視窗，暗色主題，支援語音、攝影機面板、斜線指令自動完成 |
| 終端機 CLI | `python lib/components/cli.py`<br>`python lib/components/cli.py --character 角色名` | 外觀仿終端機聊天工具，功能與桌面版同步 |
| Web API | 見下方「Web 版本」 | FastAPI 後端 + 靜態前端 demo |

**斜線指令**（桌面 GUI／CLI 通用，輸入 `/` 後會跳出建議清單，`Tab`／`↑↓` 選取）：

- `/model auto|sinco|code|nvidia`：切換回覆走哪個模型（預設 `auto`＝一般聊天走 sinco、程式碼請求走 code 模型；`nvidia` 需另外設定 `NVIDIA_API_KEY`，是 Rule 06 唯一的外部 API 例外）
- `/character`：另開獨立視窗（`character_browser.py`）選擇角色人格；未選角色時維持一般 sinco 模型
- `/memory`：讀寫 `memory/memory.json` 的持久記憶
- `/init`：掃描專案檔案樹，產生/更新 `CLAUDE.md`
- `/learn`、`/resume`、`/chat`、`/conversations`、`/clear`：學習樣本回饋、繼續/切換對話紀錄、清空畫面
- 其餘指令（`/rules`、`/BugFix`、`/Architecture`…）讀自 `app/command/*.md`，純粹是「讀出來當提示詞餵給模型」，不是會改變狀態的真指令

**RAG 檢索問答**（`lib/RAG.py`，目前是獨立模組，尚未接進三個前端）：

```python
from lib.RAG import ingest_paths, rag_reply

ingest_paths([Path("some_doc.txt")])   # 切塊、用自訓 embedding 模型存進本地 chromadb
rag_reply("問題內容")                    # 檢索相關片段後交給 smart_reply_traced() 回答
```

**模型訓練**：見下方「sinco 聊天模型」與「訓練骨架」章節的指令，或直接查 [`CLAUDE.md`](CLAUDE.md) 的「常用指令」。

## 目錄結構

```
AI-module/
├── tranning/          sinco 聊天模型 + 各種訓練腳本（核心）
├── data/              訓練資料（對話 pairs、程式碼 pairs）
├── app/               桌面端：Tkinter GUI + 安裝前置檢查 exe
├── web/               網頁端：FastAPI 後端 + 靜態前端 demo（攝影機物件偵測）
├── shape-vision/       獨立子專案：即時形狀/圓形偵測（教學重製）
└── CLAUDE.md          專案規則、待辦清單、錯誤紀錄、修正日誌
```

## 檔案索引（樹狀，2026-09-21 掃描；不含 .venv/.git/__pycache__）

```
AI-module/
├── .venv/                          # Python 虛擬環境（不列內容）
├── .vscode/settings.json           # VSCode 設定
├── Agent/                          # Agent 工作紀錄
│   ├── ErrorFinished/              # 已修好的錯誤紀錄 (ErrorLog.md、2026-08-13-*.md)
│   ├── ErrorLog/                   # 每日錯誤紀錄 (2026-08-13/24/27/28.md)
│   └── Index/                      # 2026-08-13-Project Advise.md
├── Rule/Basic.md                   # 基本規則
├── app/
│   ├── command/                    # 「讀出來當提示詞」的指令內容 (.md)
│   │   ├── AI_Assistant.md  Agent_Workflow.md  Architecture.md  BugFix.md
│   │   ├── CodeReview.md  PR.md  Security.md  git_workfkow.md
│   │   └── memorize.md  release.md  rules.md  system.md  task.md
│   ├── components/.cli_history     # CLI 輸入歷史
│   └── install/                    # install.msi、install_check.exe、install_check.py (環境檢查)
├── baisc/                          # Python 基礎學習練習 (ch1 / ch2 / ch9/temp)
├── check_GUI/check_errors_gui.py   # 錯誤檢查 GUI
├── data/                           # 訓練資料
│   ├── pairs.json  code_pairs.json  technical_pairs.json  schmitt_trigger_pairs.json
│   ├── pairs_*_draft.json          # 草稿資料 (context / emotion / negation_contrast)
│   ├── pairs_imported_train/val.json  character*.json  auto_learn_candidates.json
│   ├── corpus_*.txt                # 語料 (code_fullstack / code_html_css / zh_starter)
│   ├── mental.csv  mental.xlsx     # 情緒資料
│   └── dataset/{train,val}/class_a~d/   # 影像分類資料 (共 2400 張，不列)
├── install/install.ps1             # 安裝腳本
├── lib/                            # 前端殼（GUI / CLI / 影像）
│   ├── main.py                     # 桌面 GUI 入口 (python -m lib.main)
│   ├── GUI.py  img.py  img_analyize.py  AI_formula.py  basic.py
│   ├── NVIDIA.py  nvidia_api.py    # /model nvidia 用 (Rule 06 例外)
│   ├── faces_encoding.dat  logits, temperature and softmax.py
│   ├── test_NVIDIA.py  test_RAG.py  test_img.py
│   ├── components/                 # GUI / CLI 元件
│   │   ├── cli.py  command.py  conversation.py  gui.py  function.py
│   │   ├── camera.py  character_browser.py  markdown_view.py  schemdraw.py
│   │   ├── memory_store.py  conversation_store.py  session_store.py
│   │   ├── project_init.py  pcb_inspection.py  see_image.py
│   │   ├── test_*.py               # 對應上列各模組的測試
│   │   └── prompt/                 # prompt.py  prompt_traits.py  prompt_voice.py
│   ├── CVZone/                     # hand_track.py  pose.py
│   ├── RAG/                        # RAG.py  RAG_ex.py  README.md
│   ├── model/modules.py
│   ├── yolo/Instant Image Reconginizion.py
│   └── opencv2/                    # check_mouth  dnn  image  img_convert  img_to_gray  img_write  簽到 (.py) + 範例圖片
├── machine_migration/              # 換機備份 (MIGRATION_MANIFEST.md、checksums_sha256.txt、git_info.txt、requirements_freeze.txt)
├── materials_recongizion/          # 電子元件辨識 (capacitor.py、resistor.py、main.py、detect_and_read.py + test_*)
├── memory/                         # memory.json (/memory 指令)、usage.json
├── output/                         # 輸出 (三星ring2新聞消息.md、Music-Player/front/action.md)
├── shape-vision/                   # 獨立子專案：古典影像形狀辨識
│   ├── main.py  README.md  requirements.txt
│   ├── shape_vision/               # __init__  camera  gui  pipeline
│   └── tests/test_pipeline.py
├── tranning/                       # sinco 模型訓練
│   ├── chats.py                    # seq2seq 聊天/程式碼模型 + smart_reply()
│   ├── transformer_chat.py  character_model.py  reward_model.py  embedding_model.py
│   ├── train_gui.py  train_utils.py  train_grpo.py  prompt_gui.py  bpe_tokenizer.py
│   ├── tools.py  auto_learn.py  typo_augment.py  agent_workflow.py  terminal_exec.py
│   ├── build_corpus.py  build_code_corpus.py  code_retrieval.py  code_snippet_import.py
│   ├── dataset_import.py  data_split.py  filter_manifest.py  package_checkpoint.py
│   ├── pcb_dataset_prep.py  kicad_dataset_convert.py  bayesian_utils.py
│   ├── yolo_custom.py  yolo_train_utils.py  pcb_defect_train.py  road_sign_yolo_train.py
│   ├── materials_yolo_train.py  region_refer.py
│   ├── CNN.py  OCR.py  RNN.py  RNN.md  image_analysis.py  image_classifier_bnn.py  road_sign_train.py
│   ├── cv_inference_check.py  inference_check.py
│   ├── speech_to_text.py  voice_clone.py  prepare_audio.py  label_clips.py  align_lyrics.py
│   ├── download_commonvoice.py  diagnose_stt.py
│   ├── schmitt_trigger_train.py  circuit_diagram_train.py  circuit_kicad_to_schemdraw.py
│   ├── circuit_rule_check.py  circuit_schemdraw_render.py
│   ├── calculus_generator.py  calculus_solver.py  logic_reasoning_generator.py  word_problem_generator.py
│   ├── kernel_method.py  machine_learning.py  neural_network.py
│   ├── data.csv  sft_agent_data.jsonl  kernel_svm_model.pkl  linear_regression_model.pkl
│   ├── gpt_finetune_runs_log.txt  image_classifier_initial_run.log(.err)
│   ├── test_*.py                   # 每支訓練腳本對應的測試
│   ├── chat_runs/  code_runs/      # sinco 兩顆 checkpoint (config/history/vocab/encoder.pt/decoder.pt/progress.json)
│   ├── characters/                 # 周柯宇.json、character_chat_runs/、character_runs/
│   ├── MentalHealth/               # action.py  Overfitting.py  build_mental_health_data.py  emotion_runs/ ...
│   ├── image_classifier_runs/  schmitt_chat_runs/  schmitt_pretrain_runs/   # 各模型 checkpoint
│   ├── imagegen/                   # draw.py  test_draw.py  draw_runs/  demo_images/ (56 張，不列)
│   ├── data/                       # English.md、Quartus 教學、microcontroller_code_gen.py、reward_pairs.jsonl、schmitt_trigger_*
│   └── math/martix.py
├── typings/                        # js.pyi  pyscript.pyi  pyodide/{__init__,ffi}.pyi
├── web/
│   ├── backend/                    # FastAPI：app.py  detector.py  *_store.py  requirements.txt  .env  tests/
│   ├── frontend/src/components/    # 靜態 demo 頁 (index.html  script.js  style.css  python.py  google-auth-init.js)
│   ├── admin/frontend/             # 管理頁 (package.json  script.js  視覺電路圖.py  data/basic_data.sql  src/components/*)
│   └── Dockerfile  firebase.json  render.yaml  .dockerignore  .gitignore
├── 猜拳/                           # main.py  test_main.py  models/{labels.txt,model.tflite}
├── CLAUDE.md  README.MD  ErrorLog.md  to_do_list.md  UserPersonality.md   # 規則與紀錄
├── yolo11n.pt  yolo26n.pt  face_landmarker.task  pose_landmarker.task     # 預訓練權重
├── example_text_generation.py  package.json  pyrightconfig.json  前端部屬scp.md
└── 人工智慧模型-Sinco研發報告.docx  result.png  torch_install.log  .csv
```

## 核心元件

### 1. sinco 聊天模型 — `tranning/chats.py`

從零打造的字元級 seq2seq 聊天模型（不是呼叫 Claude/GPT），架構：

- **Tokenizer**：字元級（不是空白斷詞），中英文都能跑
- **Encoder**：GRU，把 prompt 讀成逐位置的 hidden state 序列
- **Decoder**：GRU + **Luong attention**，逐字生成回覆時每一步都能回頭參考 encoder 的每個位置

> 一開始 decoder 沒有 attention，短的聊天回覆背得很好，但程式碼片段一拉長（100~200+ 字元）就會全部崩潰成同一個答案——把整句話硬壓進一個固定大小的 hidden state 撐不住那麼長的序列，所以加上 attention（詳見 `chats.py` 內的架構註解）。

**兩個獨立 checkpoint**（不能混訓練，混了會互相干擾、全部崩潰）：

| Checkpoint | 資料來源 | 用途 |
|---|---|---|
| `tranning/chat_runs/` | `data/pairs.json` | 日常對話、身份問答（"你是誰"、打招呼、道謝…） |
| `tranning/code_runs/` | `data/code_pairs.json` | "寫一個 XXX 的函式" 這類程式碼片段請求 |

**即時資料查詢**（`tranning/tools.py`）：天氣（wttr.in）、簡易搜尋（DuckDuckGo Instant Answer API），都是免金鑰的純資料 API，不是 AI，最終組句仍是 sinco 自己的路由邏輯在做，符合「禁用外部 AI」的規則。

`smart_reply()` 會依訊息內容判斷要走哪一條路：即時查詢 → 程式碼模型 → 聊天模型（`smart_reply_traced()` 同時回傳走了哪一條，供 GUI 顯示「思考過程」）。

**指令**：

```bash
# 訓練聊天模型
python tranning/chats.py --data data/pairs.json --epochs 1500

# 訓練程式碼模型（序列較長，需要更大 --max-len 和獨立的 --out-dir）
python tranning/chats.py --data data/code_pairs.json --out-dir tranning/code_runs \
    --hidden-size 256 --max-len 210 --teacher-forcing-ratio 0.9 --epochs 6000

# 互動測試
python tranning/chats.py --chat
```

### 2. 桌面 GUI — `lib/main.py`

> ⚠️ 舊路徑 `app/components/`／`app/components/main.py` 已整個搬到 `lib/components/`／`lib/main.py`，此節已更新為現況（其餘尚未同步的段落見 [`CLAUDE.md`](CLAUDE.md) 說明）。

Tkinter 聊天視窗，串接 sinco 的 `smart_reply_traced()`。

- `/` 開頭會跳出指令建議清單，內建指令（`/model`、`/memory`、`/init`、`/character`…）在 `lib/components/command.py`，其餘讀自 `app/command/*.md`（讀出來當提示詞餵給模型）
- **Tab** 補全、**↑/↓** 在建議清單中移動反白、**Enter** 送出
- 送出訊息後會先顯示「思考中…」，背景執行緒跑完模型/查詢後才換成真正的回覆，避免視窗被凍住
- 「+」按鈕可以開啟本機檔案（含 `.md`）餵給模型當上下文

啟動（需在專案根目錄執行）：`python -m lib.main`

### 3. 安裝前置檢查 — `app/install/`

`install_check.py` 純標準函式庫（+ 呼叫外部 `nvidia-smi`）寫成，偵測 CPU 核心數／RAM／硬碟可用空間／NVIDIA 顯示卡 VRAM，列點對照最低與建議門檻，給出「適合安裝訓練 / 只適合跑聊天 / 不適合」的結論。用 PyInstaller 打包成 `install_check.exe`，讓連 Python 都還沒裝的使用者也能直接雙擊執行。

```bash
pyinstaller --onefile --console --name install_check app/install/install_check.py
```

### 4. Web 版本 — `web/`

- `web/backend/`：FastAPI，`/api/detect`（攝影機畫面丟進 YOLOv8n 做物件偵測）、`/api/chat`（接 sinco 的 `smart_reply`）
- `web/frontend/`：純靜態 HTML/JS/CSS demo 頁面，另有 `視覺電路圖.py`（攝影機即時電路元件辨識腳本，刻意不存截圖/錄影）
- 部署：後端 Render.com（`Dockerfile` + `render.yaml`，免信用卡的免費方案）、前端 Firebase Hosting（純靜態，`firebase.json`）
- 網頁版後端暫定使用指令啟動:
```bash
cd web/backend/
python app.py
```
### 5. 訓練骨架（等真實資料集）— `tranning/`

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `road_sign_train.py` | 台灣路牌辨識（MobileNetV2 遷移學習，凍結 backbone、不做水平/垂直翻轉增強因為路牌有方向性） | 架構完成，等真實路牌資料集 |
| `circuit_diagram_train.py` | 電路圖元件偵測（YOLO11 微調，同樣不翻轉） | 架構完成，等真實電路元件標註資料集 |
| `yolo_custom.py` | 通用 YOLO 偵測骨架（合成圓/方/三角資料驗證管線，支援背景訓練/資料增強旗標） | 合成資料已跑完，等真實資料集 |
| `pcb_defect_train.py` | PCB 缺陷偵測（YOLO，接 `pcb_dataset_prep.py --yolo-out` 的輸出，不裁圖分類） | 架構完成，等真實標註資料集 |
| `road_sign_yolo_train.py` | 路牌偵測（YOLO 版本，一張畫面多個路牌，跟上面的分類器並存） | 架構完成，等真實標註資料集 |
| `materials_yolo_train.py` | 電阻/電容定位（YOLO，接 `materials_recongizion/detect_and_read.py` 做既有古典 CV 判讀） | 架構完成，等真實標註資料集 |
| `region_refer.py` | 區域指稱（YOLO 候選框 + 自訓視覺編碼器 + 既有文字 embedding 對比學習，概念參考 VLX-Seek、乾淨室重新實作） | 合成資料已跑完，等真實圖說資料集 |
| `data_split.py` | 分層 train/val/test 切分（稀有類別也會分到三個 split） | 完成 |
| `machine_learning.py` | 線性迴歸教學範例（身高體重） | 完成，教學用 |
| `neural_network.py` | 極簡 PyTorch 全連接網路範例 | 草稿，僅定義了模型結構 |
| `OCR.py` | 文字辨識（CRNN + CTC，字元級 charset，不做翻轉增強因為文字有方向性） | 架構完成，等真實文字辨識資料集 |

每支訓練腳本都有對應的 `test_*.py`，用合成假資料驗證訓練管線本身能跑完、不會噴錯（不代表真實辨識準確度）。

### 6. shape-vision — 獨立子專案

即時形狀/圓形偵測（Canny 邊緣 → 輪廓 → 凸包 → 多邊形近似分類 + 霍夫圓檢測），教學重製並模組化。有自己的 `README.md`、`requirements.txt`、測試，跟主專案的 sinco 模型無關，純古典影像處理（無需訓練）。

## 環境安裝

專案目前用單一 venv（`.venv/`），沒有集中的 `requirements.txt`，依需要的元件分別安裝：

```bash
python -m venv .venv
.venv\Scripts\activate

# sinco 聊天模型（tranning/）
pip install torch

# 物件偵測 / 訓練骨架（tranning/、web/backend/）
pip install -r web/backend/requirements.txt

# shape-vision（獨立子專案）
pip install -r shape-vision/requirements.txt

# 打包 GUI 安裝檢查 exe（僅開發時需要）
pip install pyinstaller markdown
```

## 詳細記錄

專案規則、待辦事項、每次修正的完整脈絡都記錄在 [`CLAUDE.md`](CLAUDE.md)（Error log / 修正日誌），遇到相似問題可以先查那裡。


### 影像分類訓練：灰階輸入（2026-10-18）

在專案根目錄執行；資料夾需包含 `train/類別名稱/` 與 `val/類別名稱/`：

```powershell
.\.venv\Scripts\python.exe tranning/image_classifier_bnn.py --data-dir data/dataset --grayscale --out-dir tranning/image_classifier_gray_runs
```

`--grayscale` 會在載入時去除色彩，不修改原圖。為配合既有分類器，灰階亮度複製成三通道，並不會減少輸入通道或保證提升準確度。設定寫入輸出目錄的 `config.json`，推論自動沿用；舊 checkpoint 沒有此設定時仍使用彩色輸入。建議使用獨立輸出目錄保存灰階模型。

```powershell
.\.venv\Scripts\python.exe tranning/image_classifier_bnn.py --classify path/to/image.jpg --out-dir tranning/image_classifier_gray_runs
```

OCR 的 `tranning/OCR.py` 已內建單通道灰階載入，不需要新增選項。此變更不修改生圖、修圖或去模糊模型。


分類器訓練結束後會自動產生 `--out-dir` 下的 `training_curves.png`，包含訓練／驗證 loss 與 accuracy 曲線。需安裝 `matplotlib`。若已有 `history.json`，可直接產圖，不必重新訓練：

```powershell
.\.venv\Scripts\python.exe tranning/image_classifier_bnn.py --plot-history --out-dir tranning/image_classifier_runs
```

灰階模型請將 `--out-dir` 改成 `tranning/image_classifier_gray_runs`；圖表只代表該目錄實際記錄的訓練結果。


### 灰階特徵學習（2026-10-18）

`image_classifier_bnn.py` 新訓練使用自建三層 CNN（16/32/64 通道），所有卷積層從隨機權重學習，不下載預訓練模型。灰階仍複製成三通道，保留既有前處理介面；`--unfreeze-backbone` 僅保留舊指令相容性，自建 CNN 一律訓練全部特徵層。GUI 的圖像分類訓練亦提供灰階勾選。

建議使用新目錄，避免覆蓋舊模型：

```powershell
.\.venv\Scripts\python.exe tranning/image_classifier_bnn.py --data-dir data/dataset --grayscale --out-dir tranning/image_classifier_feature_runs --epochs 30
.\.venv\Scripts\python.exe tranning/image_classifier_bnn.py --features data/dataset/val/class_d/class_d_0018.jpg --out-dir tranning/image_classifier_feature_runs
```

第一行會訓練並輸出 loss/acc 曲線；第二行使用訓練好的模型輸出 `feature_maps.png`（三個階段、每階段前四個通道）。特徵圖顯示卷積反應，不代表模型一定理解物件或具有真實泛化能力。沒有 `architecture` 欄位的舊 checkpoint 仍以原 MobileNetV2 結構讀取，但初始化不再下載權重；舊模型不會自動變成自建 CNN，必須重新訓練。

`--features` 必須指定實際存在的圖片；`圖片路徑.jpg` 是佔位範例，不是專案檔案。相對路徑以執行指令時的目錄為準。


### 問答、上網與網頁介面（2026-10-18）

三個前端仍共用 `tranning/chats.py`。新增 `chat_quality.py` 的精確常用中英文對話輔助與有限輸出檢查，修正故事誤進程式模式、能力詢問誤查行情、定義題誤查天氣。這些規則不是模型已學會理解；未見問法仍可能偏題，角色模式維持既有流程。

`搜尋 關鍵字` 或 `search topic` 走網路索引。Serper 若有 process 環境變數 `SERPER_API_KEY` 則沿用；沒有時使用公開 DuckDuckGo HTML／Bing RSS，不自動讀取 `.env`。外站可能驗證、限流或查不到，會回報失敗。只過濾明顯詞面不符結果，不保證語意正確。搜尋結果保留來源，候選資料仍需 `/learn` 人工審核；不自動訓練。

RAG／交叉熵的精確定義題有中英文已審核本機詞條，附來源與查核日期，明確標示非即時搜尋、非模型生成。詞條參考 RAG 原始論文（https://arxiv.org/abs/2005.11401）與 PyTorch CrossEntropyLoss 文件（https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html）。其他領域的 RAG 縮寫仍需補充語境。

實際網頁在 `web/admin/frontend/src/components`：深色響應式介面、提問卡、手機功能選單、回答處理紀錄、安全來源連結、組字 Enter 保護與防重複送出。`/api/chat` 新增 `trace` 欄位；現有 `reply` 與 `conversation_id` 保留。重新啟動後端再整理網頁才會載入新的後端流程。

#### 中英文品質驗收

`data/chat_eval_cases.json` 含 24 筆日常、知識、主體、否定及追問案例，獨立於訓練 pairs。記錄輸出供人工審查，不產生虛假的語意準確率：

```powershell
.\.venv\Scripts\python.exe tranning/evaluate_chat.py --run --output output/chat_eval_offline.jsonl
```

預設停用工具並略過網路案例，適合單獨觀察聊天生成；加入 `--allow-network` 才會執行完整工具路由與查詢。此指令會進行模型推論，不會訓練或修改 checkpoint。完整路由結果不能直接當作純模型能力。`team-run/chat_eval_offline.jsonl` 是此次實測，未見中文／自然英文改寫仍多有不通順與不相關回覆；目前**未達一般人類或英文母語對話品質，更未證實與 Claude／GPT 同等能力**。後續需擴充經審核雙語語料、多輪格式、重訓並人工比較，不能只靠更多固定回覆或 loss 下降判定完成。


### 照片方向、灰階特徵與文字修改示範（2026-10-20）

獨立前端為 `lib/photo_edit_demo.py`，訓練與推論核心為 `tranning/photo_edit.py`。
不呼叫外部 AI、不載入預訓練權重；既有聊天、分類器及 `lib/img.py` 不變。

```powershell
.\.venv\Scripts\python.exe lib/photo_edit_demo.py
```

預設使用專案內的企鵝實拍照片，寫入新目錄 `output/photo_edit_demo`：原圖、灰階、邊緣、角點、旋轉結果、彩色修改結果、四方向候選及 JSON 報告。輸出目錄已存在時，請改用 `--output output/photo_edit_demo_2`，避免覆寫。

```powershell
.\.venv\Scripts\python.exe lib/photo_edit_demo.py --input lib/opencv2/penguins.jpg --output output/my_photo_demo --instruction "亮度 15；銳化 0.5；添加圓形 40 40 15 #ff5500"
```

以保留原圖色度的方式合成灰階亮度修改；新增彩色圖形屬明確操作。沒有方向 checkpoint 時保持原方向，不會用隨機權重自動旋轉；手動 `--angle 90` 為逆時針。四方向自建模型可用方向正確的照片資料夾訓練；灰階文字殘差模型可用每組 `source.png`、`target.png`、`instruction.txt` 訓練。

**能力界線**：邊緣與角點不等同語意理解；立即示範僅接受文件列出的精確文字語法。學習式編輯目前只改亮度，不能學會替換顏色，也未完成任意指令移除人物或增加真實物件的能力。尚無真實獨立照片集證明方向與語意編輯品質。完整語法、資料夾格式與訓練指令見 [示範說明](team-run/photo-edit/README.md)。


照片示範已擴充電路圖邊緣模式：`lib/img_edge.py --mode circuit --input output/circuit_edges_teaching.png --output output/circuit_edges_try`。使用專案 `.venv/Scripts/python.exe` 執行；輸出三組邊緣門檻比較與保留線寬的二值圖。完整管線 `lib/photo_edit_demo.py` 也支援 `--mode circuit`、`--edge-low`、`--edge-high`；電路預設不修改亮度。此為白底深線的影像前處理，尚非元件辨識或電路連接分析，詳見上述示範說明。


實拍麵包板可使用 `--mode breadboard`（兩個入口皆支援）：例如 `.venv/Scripts/python.exe lib/img_edge.py --mode breadboard --input output/breadboard_example/source.jpg --output output/breadboard_edges_try`。新增保邊去噪、保留原色與邊緣疊圖；實拍範例來自 Adafruit https://www.adafruit.com/product/4203 ，來源紀錄見該資料夾 source.json。色彩遮罩只是像素前處理，尚未辨識杜邦線／電阻／孔位或接線，完整層級仍待使用者確認；不將單張實拍流程通過當作模型已理解真實電路。


### 地區封禁規格、隱私草案與模型改進計畫（2026-10-20）

[需求與待確認項](team-run/region-enforcement/REQUIREMENTS.md)、[隱私協議草案](team-run/region-enforcement/PRIVACY_DRAFT.md)、[模型能力差距與16週作業表](team-run/region-enforcement/MODEL_ROADMAP.md)。四地限制、拒絕定位仍可使用與僅人工解除已記錄；現有後端仍為 IP 識別與 observe_only，未啟用帳戶封禁。隱私草案保留待填資訊，不是已生效協議；時程為附投入假設的工作估算，並非達到 Claude/GPT 整體能力的承諾。


使用者現已確認目標為「接線理解」，並提供自有來源 `C:/Users/roy/Downloads/電路資料集/`（先前專題測試結果為誤傳，排除）。新增唯讀整理工具 `lib/breadboard_dataset.py`／`tranning/breadboard_dataset.py`，實跑結果15來源、29資產、0讀取錯誤，含全來源SHA256、抽幀、空白端點／連線／孔列與分組模板，見 [資料說明](output/user_circuit_dataset/README.md) 與 [盤點結果](team-run/photo-edit/DATASET_FINDINGS.md)。未訓練或產生已驗證netlist；預期電路／三腳元件腳序仍待補充。


## conda 環境
建置程式:
```bash
conda create -n AI-module python=3.11
```
啟動程式:
```bash
conda activate AI-module
```

## 自訂物品標註與即時偵測

人物模型需獨立使用時，執行 `.\.venv\Scripts\python.exe -m lib.yolo.character_labeler`；使用獨立人物資料及 `characters.pt`，見[人物操作說明](lib/yolo/CHARACTERS.md)。

執行 `.\.venv\Scripts\python.exe -m lib.yolo.object_labeler`，可匯入照片或擷取攝影機影像，人工框選橡皮擦等自訂類別、儲存標註、微調本地 YOLO 權重，再使用訓練出的 `best.pt` 即時偵測。完整步驟見 [操作說明](lib/yolo/README.md)。這次已明確授權物品偵測使用預訓練權重微調；尚未收集真實標註或完成真實模型訓練，不代表已能準確辨識橡皮擦。
