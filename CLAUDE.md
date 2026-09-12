# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI-module.
## 目錄(超連結)
01. [To do list](to_do_list.md)
02. [Error Log](ErrorLog.md)
## Rules
01. 不可離開與本專案其他的資料夾內容，本專案資料夾為: "C:\Users\roy\Documents\GitHub\AI-module" 以外的內容進行任意修改，除非我本人同意授權後才可以進行
02. 我會使用繁體中文與英文進行交流討論，當我輸入中文就用中文回應，英文就用英文回應，不可以問 A 答 B ，以免 token 用量的浪費
03. 有任何錯誤就輸出到 error log 當中，紀錄錯誤點好讓我可以檢查修正
04. 我會給予修正日誌，當修正好的結果確認功能沒問題後會讓你寫入到修正日誌當中，以後未來有遇到相似的問題可以快速排查
05. 不可以觸碰到 .env 檔案，這個由我自行手動進行輸入
06. 整個專案全部禁用其他 AI 模型或是 API ，而是我要自行設計 AI 模型，不管需要多高的算力等問題，要求就是可以在 RTX 4060 8G 為上限的算力能力進行處理，目前都是屬於私人使用，不考量對外公布等任何因素問題
07. 已變更 to do list 與 error log 的紀錄，變更於目錄(超連結)當中，可以看到有 markdown 檔案
## 需求
01. [x] 指令需要加上: /model, /init, /memory 等，並且需要選項框顯示，一樣與 home_screen.py GUI 一樣可以 tab 快速鍵入 — 見 to_do_list.md #17
02. [x] 角色的部分我要求能夠另外在/character的指令部分進入這個功能(另外的 GUI 介面，不是與 home_screen.py 同一個 GUI) — 見 to_do_list.md #17
03. [x] 一般情況下都是以一般的 AI 模型進行對話 — /model 預設值、/character 未選角色時都是 sinco 一般模型，維持這個預設行為
04. [x] 對於餵養的模型學習(machine learning)部分要的是使用一般的檔案，如音訊檔案、影片檔案、文字檔案等，我手頭上沒有足夠的 json 檔案可以給予模型進行訓練，所以 train_gui.py 需要改正適用檔案格式 — 見 to_do_list.md #17（tranning/dataset_import.py）

## 指令規則（由需求 #01/#02 落實為固定規範）
01. `/model`、`/memory`、`/init` 是真正的 Python 端指令（會改變狀態/寫檔），不是 app/command/*.md 那種「讀出來塞給模型當提示詞」的指令；新增這類指令時要沿用這個區分，不要混在一起
02. 指令的引數選項（例如 `/model` 的 auto/sinco/code/nvidia）要能在輸入「/指令 」之後跳出選項框、用 Tab／上下鍵操作，跟指令名稱自動完成同一套機制（`lib/components/command.py` 的 `ARG_SUGGESTIONS`；CLI 前端對應 `lib/components/cli.py` 的 `_arg_suggestions()`）
03. `/character` 永遠開獨立的 Toplevel 視窗（`lib/components/character_browser.py`），不可以把角色清單改成塞進桌面 GUI 聊天視窗本身的分頁或面板

## 架構總覽

三個前端（桌面 GUI / 終端機 CLI / 網頁 API）都只是殼，實際的對話邏輯只有一份，在
`tranning/chats.py` 的 `smart_reply()` / `smart_reply_traced()`：

- **桌面 GUI**：入口是 `lib/main.py`（Tk 主視窗，暗色主題、圓角輸入膠囊，外觀刻意
  仿照 Claude.ai 但配色/命名仍待去除品牌字樣，見待辦），透過 `lib/components/conversation.py`
  的 `Conversation` 類別在背景執行緒呼叫 `smart_reply_traced()`，避免卡住 Tk 主迴圈；
  `lib/components/command.py` 的 `CommandPalette` 負責斜線指令自動完成與 `/model`、
  `/memory`、`/init`、`/character`、`/learn` 這幾個「真指令」；`lib/components/camera.py`、
  `character_browser.py` 各自開獨立視窗
- **終端機 CLI**：`lib/components/cli.py`，外觀仿 Claude Code CLI（橫幅、斜線指令、
  「思考中」狀態列），後端跟桌面版共用同一個 `smart_reply_traced()`，不新增任何模型
  或路由邏輯
- **網頁**：`web/backend/app.py`（FastAPI），`/api/chat` 接 `smart_reply`、`/api/detect`
  丟 YOLOv8/11 做攝影機物件偵測；`web/frontend/` 是純靜態 demo 頁面
- **共用小工具**：`lib/components/function.py`（純函式，讀檔/去 Markdown 標籤，故意
  不依賴 Tkinter，command.py 跟 conversation.py 都從這裡 import）、
  `lib/components/memory_store.py`（`/memory` 指令的 JSON 持久記憶，存在 `memory/memory.json`，
  跟 `app/command/*.md` 那種「整份讀出來塞給模型當提示詞」是不同機制）、
  `lib/components/project_init.py`（`/init` 指令：deterministic 檔案樹掃描寫 CLAUDE.md，
  不是模型生成——sinco 目前的 seq2seq 還沒有讀懂任意程式碼庫的能力）

**指令內容 vs 指令邏輯是分開兩個資料夾**：`app/command/*.md` 只放「讀出來當提示詞」的
指令內容（`rules.md`、`BugFix.md`、`Architecture.md`…），指令的 UI/執行邏輯（自動完成、
`/model` 等真指令）在 `lib/components/`。

**外部 API 是 Rule 06 的顯式例外，不是預設路徑**：`lib/NVIDIA.py` 提供 `/model nvidia`
手動切換用的 NVIDIA 雲端模型（需要環境變數 `NVIDIA_API_KEY`，讀取 `lib/.env`），只有
使用者手動選擇該模式才會被呼叫；預設（`auto`/`sinco`/`code`）三種模式完全不碰外部 API。

**`lib/img.py` 是「訓練邏輯留在 tranning/、lib/ 只放前端殼」這條既定分工的刻意例外**
（2026-09-12 你明確選擇直接寫進 lib/img.py，不是誤放）：文字生圖／修圖的 CVAE 訓練與
推論程式碼都直接寫在 `lib/img.py`，重用 `tranning/imagegen/draw.py` 的 `Encoder`/
`Decoder`/`vae_loss`（動態 `sys.path.insert` 後 `import draw`，跟 `lib/components/
conversation.py` import `chats`/`speech_to_text` 同一套慣例），沒有另外搬進
`tranning/imagegen/`。之後如果新增其他「訓練邏輯」性質的檔案，預設仍照舊規矩放
`tranning/`，除非你再次明確要求例外。

### sinco 核心模型（`tranning/`）

字元級 seq2seq（GRU encoder + Luong attention GRU decoder），兩個獨立 checkpoint 不能混
訓練：`tranning/chat_runs/`（`data/pairs.json`，日常對話）、`tranning/code_runs/`（
`data/code_pairs.json`，程式碼片段請求）。`smart_reply()` 依訊息內容判斷走即時查詢
（`tranning/tools.py`：天氣/DuckDuckGo，純資料 API 不是 AI）→ 程式碼模型 → 聊天模型。
其餘 `tranning/*.py`（`transformer_chat.py`、`character_model.py`、`RNN.py`、`CNN.py`、
`OCR.py`、`speech_to_text.py`、`road_sign_train.py`…）是同一顆「全部自己從零訓練」原則
下的各種模型/骨架，多數還在等真實資料集（見 to_do_list.md 逐項狀態）。`tranning/imagegen/
draw.py` 是無條件式生圖 VAE（給圖片資料夾學畫類似風格，不吃文字）；`lib/img.py`（見上
一節的分工例外）在它的架構上加文字條件，做文字生圖與修圖兩個骨架，同樣還在等真實
圖片資料集（見 to_do_list.md #31）。`lib/img.py` 的第三個功能「模糊變清晰」
（`DeblurNet`／`train_deblur()`／`deblur()`）不用 `draw.py` 的 Encoder/Decoder（那個
128 維 latent 瓶頸會把銳化任務需要的細節壓沒），改用有 skip connection 的小型 U-Net，
且**不需要另外準備配對資料集**——給任何一個清晰圖片資料夾，訓練時自動隨機高斯模糊
合成模糊/清晰配對；也可以另外給 `[{"blurry":...,"sharp":...}]` manifest 用真實模糊照片
（見 to_do_list.md #32）。

### 目錄現況（⚠️ 與 README.MD 描述不同步，以此為準）

`app/components/` 已整個搬到 `lib/components/`、`app/components/main.py` 搬到並改名為
`lib/main.py`（這次搬移尚未 commit，git status 會顯示 `app/components/*` 為 deleted、
`lib/` 為 untracked）。`app/` 現在只剩：`app/install/install_check.py`（環境檢查，
PyInstaller 打包成 exe）與 `app/command/*.md`（指令提示詞內容）。README.MD 裡「桌面
GUI 入口是 `app/command/home_screen.py`」等描述已經過期，实际入口是 `lib/main.py`。

## 常用指令

```bash
# 訓練 sinco 聊天/程式碼模型
python tranning/chats.py --data data/pairs.json --epochs 1500
python tranning/chats.py --data data/code_pairs.json --out-dir tranning/code_runs \
    --hidden-size 256 --max-len 210 --teacher-forcing-ratio 0.9 --epochs 6000
python tranning/chats.py --chat   # 互動測試

# 啟動桌面 GUI（在專案根目錄執行，讓 lib.components.* 這種絕對匯入找得到）
python -m lib.main

# 啟動終端機 CLI
python lib/components/cli.py
python lib/components/cli.py --character 周柯宇

# 文字生圖／修圖骨架（尚無真實資料集，見 to_do_list.md #31）
python lib/img.py --train-t2i captions.json --epochs 100
python lib/img.py --train-edit edits.json --epochs 100
python lib/img.py --generate "a red circle" "a blue square"
python lib/img.py --edit before.png "make it blue"

# 模糊變清晰（只需要一個清晰圖片資料夾，自動合成模糊配對，見 to_do_list.md #32）
python lib/img.py --train-deblur path/to/sharp_photos/ --epochs 100
python lib/img.py --deblur blurry.jpg

# 跑測試（各元件分開跑，彼此 requirements 不共用；沒有集中的 pytest.ini/conftest.py）
pytest tranning/
pytest lib/components/
pytest lib/test_img.py
pytest web/backend/tests/
pytest shape-vision/tests/

# 打包安裝前置檢查 exe
pyinstaller --onefile --console --name install_check app/install/install_check.py
```

已知環境缺口（見 ErrorLog.md 第 8-10 條）：缺套件會讓整個目錄在 pytest collect 階段
就失敗，不是單一測試失敗——`tranning/` 需要 `beautifulsoup4`、`fitz`(PyMuPDF)；
`lib/components/` 的 `cli.py`/`test_cli.py` 需要 `prompt_toolkit`。另外專案沒有集中
`requirements.txt`，依元件分別 `pip install`（`torch`、`web/backend/requirements.txt`、
`shape-vision/requirements.txt`）。

**兩份 Python 環境並存（見 ErrorLog.md 第 11 條）**：`pytest` 用專案的 `.venv`；GUI
過去曾用另一個 conda/micromamba 環境（`cuda` env）手動啟動測試，兩邊安裝的套件不完全
一致，是已知現況，換成統一環境前不要假設兩邊套件相同。

## 專案規則

- 私人自建 AI 專案，禁止呼叫其他雲端 LLM API（Rule 06），唯一顯式例外是 `/model nvidia`
  手動切換（見上）
- 每支訓練腳本都有對應 `test_*.py`，用合成假資料驗證訓練管線本身能跑完、不代表真實
  準確度
- `shape-vision/` 是獨立子專案（純古典影像處理，Canny 邊緣→輪廓→凸包→多邊形分類+
  霍夫圓檢測），跟 sinco 模型無關，有自己的 README/requirements/tests
- 完整規則、待辦清單、每次修正的除錯脈絡見 [to_do_list.md](to_do_list.md) 與
  [ErrorLog.md](ErrorLog.md)——遇到相似問題先查那裡
