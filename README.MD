# AI Module

私人自建 AI 專案。核心規則：**禁止使用其他 AI 模型或雲端 API**（不呼叫 Claude／GPT／任何 LLM API），所有模型都自己從零設計、從零訓練，算力上限抓 **RTX 4060 8GB**，目前僅供個人使用，不考慮對外公開。完整規則見 [`CLAUDE.md`](CLAUDE.md)。

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

### 2. 桌面 GUI — `app/command/home_screen.py`

Tkinter 聊天視窗，串接 sinco 的 `smart_reply_traced()`。

- `/` 開頭會跳出指令建議清單，指令內容取自同資料夾內的 `*.md` 檔（讀出來當提示詞餵給模型）
- **Tab** 補全、**↑/↓** 在建議清單中移動反白、**Enter** 送出
- 送出訊息後會先顯示「思考中…」，背景執行緒跑完模型/查詢後才換成真正的回覆，避免視窗被凍住
- 「+」按鈕可以開啟本機檔案（含 `.md`）餵給模型當上下文

啟動：`python app/command/home_screen.py`

### 3. 安裝前置檢查 — `app/install/`

`install_check.py` 純標準函式庫（+ 呼叫外部 `nvidia-smi`）寫成，偵測 CPU 核心數／RAM／硬碟可用空間／NVIDIA 顯示卡 VRAM，列點對照最低與建議門檻，給出「適合安裝訓練 / 只適合跑聊天 / 不適合」的結論。用 PyInstaller 打包成 `install_check.exe`，讓連 Python 都還沒裝的使用者也能直接雙擊執行。

```bash
pyinstaller --onefile --console --name install_check app/install/install_check.py
```

### 4. Web 版本 — `web/`

- `web/backend/`：FastAPI，`/api/detect`（攝影機畫面丟進 YOLOv8n 做物件偵測）、`/api/chat`（接 sinco 的 `smart_reply`）
- `web/frontend/`：純靜態 HTML/JS/CSS demo 頁面，另有 `視覺電路圖.py`（攝影機即時電路元件辨識腳本，刻意不存截圖/錄影）
- 部署：後端 Render.com（`Dockerfile` + `render.yaml`，免信用卡的免費方案）、前端 Firebase Hosting（純靜態，`firebase.json`）

### 5. 訓練骨架（等真實資料集）— `tranning/`

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `road_sign_train.py` | 台灣路牌辨識（MobileNetV2 遷移學習，凍結 backbone、不做水平/垂直翻轉增強因為路牌有方向性） | 架構完成，等真實路牌資料集 |
| `circuit_diagram_train.py` | 電路圖元件偵測（YOLOv8 微調，同樣不翻轉） | 架構完成，等真實電路元件標註資料集 |
| `data_split.py` | 分層 train/val/test 切分（稀有類別也會分到三個 split） | 完成 |
| `machine_learning.py` | 線性迴歸教學範例（身高體重） | 完成，教學用 |
| `neural_network.py` | 極簡 PyTorch 全連接網路範例 | 草稿，僅定義了模型結構 |
| `OCR.py` | 文字辨識 | 空檔，尚未實作 |

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
