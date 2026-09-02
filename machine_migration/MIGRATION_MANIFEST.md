# 換機交接清單

建立時間：2026-08-23
用途：記錄目前這台電腦上 AI-Module 專案的模型版本、環境與所有換到新電腦時需要注意的資訊，避免升級硬體後遺失東西或環境對不上。

本資料夾內容為**唯讀快照**，不會改動專案原本任何檔案。

---

## 1. 結論先講：模型本體已經安全

檢查結果：目前 `main` 分支與 `origin/main`（GitHub: RoyZeng0317/AI-module）**完全同步**，沒有任何未 commit 或未 push 的變更。

也就是說：**下列所有模型 checkpoint 都已經在 Git 版控＋GitHub remote 裡，換電腦只要 `git clone` 就會自動全部拿回來，不需要額外複製這些大檔案：**

| 模型 | 路徑 | 說明 |
|---|---|---|
| sinco 聊天模型 | `tranning/chat_runs/{encoder,decoder}.pt` | 字元級 seq2seq（GRU+attention），日常對話 |
| sinco 程式碼模型 | `tranning/code_runs/{encoder,decoder}.pt` | 同架構，程式碼片段請求 |
| 角色對話模型 | `tranning/characters/character_chat_runs/{encoder,decoder}.pt` | 周柯宇角色對話 |
| 角色分類模型 | `tranning/characters/character_runs/best_model.pt` | 角色特徵分類 |
| 圖像生成 VAE | `tranning/imagegen/draw_runs/vae.pt` | draw.py 用 |
| 迴歸模型 | `tranning/linear_regression_model.pkl` | machine_learning.py 用 |
| YOLO 偵測模型 | `yolo11n.pt`（根目錄） | 攝影機物件偵測，Ultralytics 官方預訓練權重 |

當前 commit：`99c5ab032ffdb215b95bb4faa6bc40f29ce6abc5`（詳見同資料夾 `git_info.txt`）。

**`checksums_sha256.txt`** 已計算上述每個模型檔案的 SHA256。到新電腦 `git clone` 後，用同樣指令再算一次，兩邊數值要完全一致，才能確認檔案在搬運過程沒有損毀：

```powershell
Get-FileHash yolo11n.pt -Algorithm SHA256
Get-FileHash tranning\chat_runs\encoder.pt -Algorithm SHA256
# ...其餘依 checksums_sha256.txt 清單逐一比對
```

---

## 2. Git 沒有記錄到、需要「自己手動」處理的東西

以下檔案/資料夾在 `.gitignore` 中被排除，**不會**跟著 `git clone` 一起過去，而且我依規則（專案 Rule 05：不可觸碰 `.env`）不會去讀取或複製它們的內容，需要你自己決定怎麼搬：

| 項目 | 路徑 | 重要性 | 建議 |
|---|---|---|---|
| API 金鑰 | `lib/.env` | ⚠️ 高（`/model nvidia` 需要 `NVIDIA_API_KEY`） | 你自己手動複製這個檔案到新電腦相同路徑，我不會碰它 |
| 對話記憶 | `memory/session.json`（54KB） | 中（`/resume` 的歷史對話紀錄，不可重新產生） | 建議手動複製；若不需要保留歷史對話可略過 |

以下是**可重新產生 / 不重要**的東西，換電腦不用管：

- `.venv/`、`.pytest_cache/`、`__pycache__/`、`.cli_history` — 環境快取，新電腦重裝即可
- `nova.exe`、`build/` — PyInstaller 打包產物，用 `pyinstaller --onefile --console --name install_check app/install/install_check.py` 可重新產生
- `web/backend/yolo11n.pt`、`web/backend/yolov8n.pt` — Ultralytics 官方預訓練權重，重新執行會自動下載
- `tranning/commonvoice_zhTW/`、`gpt_chat_runs/`、`gpt_pretrain_runs/`、`speech_runs/`、`speech_data/`、`Tranning/` — `.gitignore` 註明「可重新下載/重新產生的訓練資料」

---

## 3. 環境資訊（這台電腦目前狀態）

- Python：3.14.6
- PyTorch：**2.13.0+cpu** — ⚠️ 注意：目前這台電腦裝的是 **CPU-only 版本**，沒有吃到 GPU 加速
- 偵測到的 GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，driver 591.91（但因為 torch 是 cpu 版，`torch.cuda.is_available()` 回傳 `False`，目前訓練/推論其實都在跑 CPU）
- 完整套件清單已存成 `requirements_freeze.txt`（`pip freeze` 全量輸出，86 個套件，內容不含機密資訊，可安心保留）

**換到效能更好的新電腦時建議做的事**：

1. `git clone https://github.com/RoyZeng0317/AI-module.git`
2. 建新的 `.venv`，用 `requirements_freeze.txt` 還原套件版本作為參考起點：
   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -r machine_migration\requirements_freeze.txt
   ```
   （注意：`requirements_freeze.txt` 是舊電腦裝的 CPU 版 torch，如果新電腦想要吃到 GPU 加速，torch 這行建議改用官方指令依新 GPU/CUDA 版本重裝，而不是照抄舊版本 — 參考 https://pytorch.org/get-started/locally/ ）
3. 依 CLAUDE.md 補裝其餘分散的 requirements：`web/backend/requirements.txt`、`shape-vision/requirements.txt`，以及 `beautifulsoup4`、`fitz`(PyMuPDF)、`prompt_toolkit`
4. 手動複製 `lib/.env` 到新電腦同一路徑
5. （選用）手動複製 `memory/session.json` 以保留對話記憶
6. 跑 `pytest tranning/`、`pytest lib/components/` 確認環境正常
7. 用第 1 節的 SHA256 checksum 比對模型檔案完整性

---

## 4. 本資料夾檔案清單

- `MIGRATION_MANIFEST.md` — 本檔案
- `requirements_freeze.txt` — `pip freeze` 完整快照
- `checksums_sha256.txt` — 核心模型檔案 SHA256（用來換機後驗證檔案沒壞）
- `git_info.txt` — 目前 branch / commit / remote 同步狀態 / 近期 commit 紀錄

> 這個資料夾目前**還沒有加進 `.gitignore`，也還沒有 commit**（我依規則不會主動改動 `.gitignore` 或幫你下 git commit）。裡面沒有機密內容（不含 `.env`、不含 `session.json`），你可以自行決定：
> - 直接 commit 進 repo 當作交接文件，或
> - 加進 `.gitignore` 只留在本機參考，或
> - 手動複製整個 `machine_migration/` 資料夾到隨身碟/雲端硬碟另外保存
