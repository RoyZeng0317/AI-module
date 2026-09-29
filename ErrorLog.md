# Error Log

> **2026-09-14 補記**：`to_do_list.md`／Claude memory 裡多處引用「ErrorLog #22」
> 「ErrorLog #23」「ErrorLog #29」，代表根目錄這份 `ErrorLog.md` 過去確實累積到
> 至少 29 號。但今天要新增紀錄時發現**這個檔案在磁碟上完全不存在**：`git log --all`
> 對它唯一的紀錄是 2026-08-17 commit `3d232f4`「刪除」（當時改成
> `Agent/ErrorLog/YYYY-MM-DD.md` 分日期檔案），但 memory 顯示 2026-09-10 那次
> 對話確實有寫入過「entry 22」——代表這份檔案後來被重建過，但那次重建從未被
> `git add`，之後不知道在什麼情況下又整份消失，也沒有留下任何 git 紀錄可查。
> **結論：1~29 號的完整內容目前已經遺失**，只能從 `to_do_list.md`（#21/#22/#23/#27/#29
> 一帶）跟 Claude memory 裡的摘要片段拼湊大概內容，原始細節找不回來了。這份檔案
> 從第 30 號重新開始記錄；建議之後每次修改都順手 `git add ErrorLog.md`，避免
> 未追蹤檔案再度整份遺失。

## 30. sinco 一般聊天（Transformer 分支）對英文輸入給出不合理的短回覆

**回報**：2026-09-14，使用者用英文跟 sinco 對話：
- 「hello, how are you?」→ 回「I'm thanking」
- 「what are you doing now?」→ 回「I'm!」

兩則回覆都文法不通、答非所問，使用者判定為「模型回應不合理」。

**現況確認**：一般聊天分支目前預設呼叫 `tranning/transformer_chat.py` 的
`reply()`（`chats.py` 的 `smart_reply_traced()` 自 2026-09-10 起改成呼叫這支，
GRU `chat_runs` 不再是預設路徑，見 to_do_list.md #27）。

**根因分析**（未實際重訓，純檢查現有 checkpoint／資料）：
1. `data/pairs.json`（609 筆）裡確實有 `"how are you" -> "I'm good, thanks for
   asking"` 這筆一模一樣的訓練資料，但沒有任何一筆涵蓋
   「what are you doing now」這類問法——英文語料本來就比中文語料稀疏。
2. `tranning/gpt_chat_runs/history.json` 最後一個 epoch（41）：
   `train_loss = 0.247`、`val_loss = 2.156`，兩者差距非常大，是典型的
   **過擬合**——8 層／384 維的 Transformer 對只有 609 筆的 finetune 資料集來說
   偏大，模型記得住見過的句子，但對「幾乎一樣但不完全一樣」（如 "how are you"
   實際上有練過，卻仍生成走樣的 "I'm thanking"）或完全沒見過的英文問法
   （"what are you doing now"）就很容易生成不連貫的短句。
3. 生成本身是取樣（`temperature=0.7`／`top_k=30`／`top_p=0.85`，非貪婪解碼），
   在分布外（out-of-distribution）輸入上取樣容易放大不連貫程度，跟
   ErrorLog #22（同一顆 checkpoint 對「我今天心情不好」有 37.5% 機率把負面
   情緒誤判成正面）是同一個「17M 參數模型的容量天花板」問題的另一種病徵，
   不是新的獨立 bug。

**尚未修正**。可能的修法（都需要 GPU 訓練時間，依
`feedback_no_unattended_long_training` 記憶規則，要先跟使用者確認要採哪個
方向、什麼時候做，不會自己擅自開一輪重訓）：
- 補更多英文日常對話的 pairs（尤其是閒聊類問法的變體），再重新 finetune；
- 降低模型容量或加強正則化（更高 weight decay／更早 early stop）減少過擬合；
- 純調整解碼參數（比照 2026-09-10 那次靠 `repetition_penalty` 不重訓就改善
  重複迴圈的做法）——但這次的病徵是「短且無意義」而非「重複迴圈」，不確定
  單靠調參能不能解決，需要先用同樣的測試句實際比較。

**相關**：ErrorLog #22（同顆 checkpoint 的負面情緒誤判，37.5% 機率）、
to_do_list.md #27（Transformer 分支整段訓練與上線過程）。

## 31. `web/backend/app.py` 的 sys.path 順序讓 `import conversation_store` 一直載到桌面版那份

**發現於**：2026-09-14，實作 to_do_list.md #36（對話紀錄依 Google 帳號分開）
時，替 `web/backend/conversation_store.py` 加上必填的 `owner` 參數後，
`pytest web/backend/tests/test_app.py` 直接炸出
`AttributeError: 'str' object has no attribute 'exists'`，traceback 指向
`lib\components\conversation_store.py:66`——不是我改的那支檔案。

**根因**：`app.py` 開頭依序
`sys.path.insert(0, BACKEND_DIR)` → `TRAINING_DIR` → `FRONTEND_DIR` →
`COMPONENTS_DIR`，每次 `insert(0, ...)` 都會把新路徑推到最前面，結果最後
`sys.path[0]` 是 `COMPONENTS_DIR`（`lib/components/`）。這個資料夾底下也有一支
同名的 `conversation_store.py`（桌面 GUI/CLI 版，函式簽名是
`create_conversation(path=None)`，沒有 `owner` 參數），Python 找模組時
`COMPONENTS_DIR` 排在 `BACKEND_DIR` 前面，`import conversation_store as
convo_store` 這幾年來實際上一直載到桌面版那份，`web/backend/
conversation_store.py` 形同沒被用到。過去兩份檔案的 API 一直保持同步
（docstring 裡也明講「改一邊要記得同步改另一邊」），所以行為看起來完全正常，
沒人發現真正被 import 進來的是哪一支。

**影響範圍**：只有這次新增的「對話紀錄依 Google 帳號隔離」功能會踩到——舊的
兩份 API 只要維持完全同步，這個 shadowing 不會造成任何可觀察的行為差異；
但這次刻意只在 `web/backend/conversation_store.py` 加 `owner` 欄位（桌面版
沒有 Google 登入概念，不需要），兩份 API 第一次真的產生分歧，才讓這個
排序問題浮出水面。

**修正**：把 `sys.path.insert(0, str(BACKEND_DIR))` 移到四行的最後一行執行，
讓 `BACKEND_DIR` 排在 `sys.path[0]`，`import conversation_store` 真的載到
這支後端自己維護的檔案。已確認 `web/backend`、`lib/components`、`tranning/`
三個資料夾之間沒有其他同名 `.py` 檔案會被這個排序調整意外影響到。
`pytest web/backend/`（41 項）、`pytest lib/components/
test_conversation_store.py`（14 項）都全過。

**相關**：to_do_list.md #36。

## 32. 長時間訓練跟著 VSCode 關閉一起中斷，中途沒有 checkpoint 可以續跑

**回報**：2026-09-14，`tranning/code_runs`（`chats.py --data code_pairs.json`）的訓練跑到一半，使用者不慎關掉 VSCode。

**確認**：
- `tasklist` / `Get-CimInstance Win32_Process` 查無任何 `chats.py` 相關 python 行程仍在執行（僅剩兩個跟本專案無關的 `AutoPush\lib\GUI.py` 行程）。
- `tranning/code_runs/progress.json` 停在 `"epoch": 2488, "epochs": 3000, "percent": 82.9, "done": false`，最後寫入時間 21:04:13，跟發現當下（21:05:24）只差約 1 分鐘，確認是被砍掉當下、不是訓練自己跑完收尾。
- 但 `encoder.pt`/`decoder.pt` 最後寫入時間是 18:40:58，比 `progress.json` 停下的時間早了 2 小時 24 分——`chats.py` 目前只有在**整個訓練跑完**時才寫一次模型權重，`progress.json`/`history.json` 才是逐 epoch 更新。等於這 2488 個 epoch（約 2.4 小時）的訓練成果完全沒有 checkpoint 可續跑，只能整個重來。

**根因（兩個疊加）**：
1. 訓練行程掛在 VSCode/Claude Code 這次工作階段底下執行，VSCode 關閉時子行程被一併終止，沒有用能獨立於編輯器存活的方式（背景服務／`nohup`＋`disown`／獨立視窗）啟動。
2. `chats.py`／`transformer_chat.py` 的權重存檔時機只在訓練全部結束時寫一次，沒有像 `progress.json` 一樣逐 epoch（或每隔 N epoch／每隔固定時間）存一次中途 checkpoint。

**原則（往後所有長時間訓練都要遵守，不只這次）**：
1. 啟動任何預期跑超過幾分鐘的訓練前，先跟使用者確認要不要現在跑、要用什麼方式背景執行——沿用既有 `feedback_no_unattended_long_training` 記憶規則，不擅自開跑。
2. 訓練腳本要能中途存 checkpoint（例如每隔 N epoch 存一次權重，取代「全部跑完才存一次」），這樣中斷後才能從最近的 checkpoint 續跑而不是整個重來。**這項程式碼修改本身還沒做**，列入 to_do_list.md #37 待辦。
3. 在確認第 2 點修好之前，任何長時間訓練都要明確告知使用者「中途中斷=整段重來」這個風險，並提醒不要在訓練跑著的時候關閉 VSCode／編輯器視窗。

**現況**：本次 `code_runs` 訓練需要從頭重跑（無中途 checkpoint 可續），使用者選擇明天再繼續，這次先不重啟。

**相關**：`feedback_no_unattended_long_training` 記憶規則、to_do_list.md #37。

## 33. 網頁前端英文對話中途被回覆成不相關的中文短句

**回報**：2026-09-16，使用者在網頁前端（web/frontend 或 web/admin）跟 sinco 對話：
- 「how are you?」→ 回「I'm good, thanks for asking」（正常）
- 接著「ok, got it.」→ 回「可以幫你看」（答非所問，且語言從英文突然跳成中文）

**確認**：`web/backend/app.py:510-549` 的 `/api/chat` 直接把使用者原文丟給
`smart_reply_traced()`，沒有任何語言偵測／過濾。一般聊天分支會走到
`tranning/chats.py:687` 呼叫 `tranning/transformer_chat.py` 的 `reply()`；
這支是純自迴歸取樣（`transformer_chat.py:624-643`，
`temperature=0.7`／`top_k=30`／`top_p=0.85`／`repetition_penalty=1.3`），
聊天分支**沒有**像 `sinco-code` 分支那樣對 `data/pairs.json` 做相似度檢索
（`code_retrieval.retrieve()` 只用在 `chats.py:658-668` 的程式碼分支），
所以「可以幫你看」不是比對錯到不相關的訓練資料，是模型對分布外輸入直接
生成出來的。

**根因**：`data/pairs.json`（609 筆）裡只有約 31 筆（≈5%）是英文提問，且
沒有任何一筆接近「ok」「got it」這種簡短應答；`pretrain()` 用的
`data/corpus_zh_starter.txt`（zh-wiki 語料）幾乎 100% 是中文。模型整體
token 統計嚴重偏中文，遇到訓練時完全沒見過的英文簡短句型，取樣時就會
偏向高機率的中文字詞組合，生成出一句語法通順但完全不相關的中文——跟
ErrorLog #30（同一顆 17M 參數 checkpoint 對分布外英文輸入生成不連貫短句）
是同一個容量天花板問題，這次多了「連語言都會跳掉」這個新病徵，之前沒記錄過。

**尚未修正**。修法方向與 #30 相同（補英文日常對話 pairs 再重訓／調整解碼
參數／降低模型容量或加強正則化），是否要現在動手、選哪個方向，需要先問
使用者，不擅自開訓練（`feedback_no_unattended_long_training` 記憶規則）。

**相關**：ErrorLog #30（同顆 checkpoint 對分布外英文輸入的病徵）、
to_do_list.md #27。

## 34. lib/opencv2/dnn.py 攝影機物件偵測只顯示最後一幀、縮排錯誤

**症狀（程式碼審視發現，尚未實機執行）**：`imshow`／`waitKey(0)` 寫在 `while` 迴圈外，只會顯示最後一幀；`net = cv2.dnn.readNet(...)` 與 `VideoCapture(0)` 縮排在 `with open(...)` 區塊內；沒有 `cap.release()`；設定檔副檔名寫成 `protext.txt`（Caffe 設定檔應為 `prototxt`）。

**修正（2026-09-21）**：`imshow` 移進迴圈並用 `waitKey(1)`，按 q／ESC 離開；縮排修正；補 `release()`；模型路徑改為相對腳本所在的 `lib/opencv2/models/`，缺檔時明確丟 `FileNotFoundError`；類別名稱改 `splitlines()`，索引越界不再 IndexError；移除每幀 `print`。

**尚未驗證**：`lib/opencv2/models/` 內沒有 `MobileNetSSD_deploy.caffemodel`／`.prototxt.txt`／`MobileNetSSD_labels.txt`，需使用者自行放入後實機測試。副檔名 `protext`→`prototxt` 是推測，若使用者檔案本來就叫 `protext` 需改回。MobileNetSSD 為現成預訓練模型，與 Rule 06 有衝突，是否保留待使用者決定。

## 35. sinco 一般聊天模型過擬合追蹤：finetune 已早停、pretrain v2 中途停在 17/25 epoch 無行程可查

**回報**：2026-09-29，`Agent/ErrorLog/2026-09-29.md` 記錄使用者要求對「一般聊天大量口吃／過擬合」做深度研究。這題本身已經是 to_do_list.md #42／#43 正在追的問題，本次只做現況診斷（`AskUserQuestion` 問過使用者，這輪範圍限定聊天模型、只診斷不訓練）。

**確認**：
1. `tranning/gpt_chat_runs_v2/history.json`（#42 的 finetune，`patience=10`）：`val_loss` 在 epoch 7 見底（3.8526），之後緩慢爬升到 epoch 17 的 3.9823，`train_loss` 同期從 2.447 一路壓到 1.112——是教科書等級的過擬合曲線；紀錄停在 epoch 17（7+10=17，`patience` 剛好打滿），檔案最後寫入時間 2026-09-22，判斷已依 early stop 自然結束，不是被中斷。
2. `tranning/gpt_pretrain_runs_v2/history.json`（#43-5，已套用「語料含 `pairs.json` 口語內容」的 BPE 修法）：`val_loss` 在 epoch 11 見底（4.4209），epoch 17 回升到 4.4808，`train_loss` 同期仍持續下降（3.52→3.01），同一種過擬合訊號在 pretrain 階段也已出現。
3. `history.json`／`model.pt` 最後寫入 2026-09-28 21:35:11，`checkpoint/` 底下最新一筆是 epoch 15（21:30:46）；`tasklist`／`Get-CimInstance Win32_Process` 查無任何跟本專案相關的 python 行程（僅兩個 `AutoPush\lib\GUI.py`，無關）。代表這個原本規劃跑 25 epoch 的 pretrain，在 epoch 17 之後、離目標還差 8 epoch 時就沒有再繼續，且發現當下（2026-09-29）已經超過 24 小時沒有動靜——研判是行程中途被中止（跟 ErrorLog #32 同一種「訓練行程跟編輯器/終端機一起被關掉」模式），不是正常跑完。好消息是這次有 `checkpoint_every=5` 安全網（#42 補的功能），只丟失 epoch 15→17 這 2 epoch 的進度，能從 `checkpoint/` 續跑，不用整個重來。

**根因**：跟 to_do_list.md #43 第 301 行已下的診斷一致——8 層／384 維／約 1746 萬參數的 Transformer，相對 `data/pairs.json` 目前 742 筆的資料量仍然偏大，不管是 finetune 還是（這次已擴充語料的）pretrain 階段，都在總 epoch 數還沒跑完前 `val_loss` 就見底反彈。目前的診斷是資料量仍是主要瓶頸，架構是否要縮小，依 #43 既定順序（資料量→斷詞→embedding→架構）要等前面步驟都試過才輪到。

**尚未修正**。這次只診斷、不訓練（使用者本輪明確選擇）。待使用者決定的選項：
- pretrain v2 要不要從 `checkpoint/`（epoch 15）續跑到 25 epoch，還是就此用 epoch 11 附近的權重當 pretrain 基底往下走 finetune；
- finetune（`gpt_chat_runs_v2`）已經因為 `patience` 自然早停，其 val_loss 最低點（epoch 7，3.8526）本身也還沒優於舊版 `gpt_chat_runs` 的最終數字（ErrorLog #30：`val_loss=2.156`），需要先用同一批測試句實測比較才能判斷是否真的優於舊版，不能只看數字；
- 是否要現在就啟動任何一段訓練，依 `feedback_no_unattended_long_training` 規則，個別詢問。

**相關**：to_do_list.md #42／#43、ErrorLog #30／#32／#33、memory `feedback_no_unattended_long_training`、`feedback_training_checkpoint_and_detach`。

## 36. `git push` 一直失敗：`RPC failed; HTTP 500` / `unexpected disconnect while reading sideband packet`

**回報**：2026-09-29，使用者要求把本地領先 origin/main 的 commit push 上 GitHub，push 直接失敗。

**根因**：待推送的 3 個 commit（`模型訓練PCB`、`資料訓練擴增與報告編寫已知錯誤`、`股票預測數據資料集`）裡，前兩個一次性新增了約 **7.3GB** 的內容：
- `data/pcb/archive/PKU-Market-PCB(...)` 原始資料集（見 `project_pcb_defect_dataset_found` 記憶），數千張圖片，每張約 5-6MB；
- 6 份幾乎完全相同的 `model.pt` checkpoint（`gpt_chat_runs_v2`、`gpt_chat_runs_v3`、`gpt_pretrain_runs_v2` 各自的頂層與 `checkpoint/` 子目錄下各一份，每份約 72MB）。

`.gitignore` 本來就有排除舊版 `tranning/gpt_chat_runs/`、`tranning/gpt_pretrain_runs/` 整個目錄（可重新產生/太大），但改版後新增的 `_v2`／`_v3` 目錄名稱沒有同步補上規則，才會被意外整包 commit 進去。單次 push 7GB+ 遠超過 GitHub HTTPS 服務穩定處理的範圍，才會回 HTTP 500。

**修正過程**（依使用者要求：最多 5 次修正）：
1. `git config http.postBuffer 1048576000` + `http.version HTTP/1.1` 後重試 → 失敗，仍是同樣錯誤。
2. 單純重試（排除暫時性問題）→ 失敗，確認不是網路瞬斷。
3. 用 `git diff-tree`／`git cat-file --batch-check` 精確定位是哪個 commit、哪些檔案造成 7.3GB，向使用者說明兩條可行路線（從 commit 中移除大檔案 / 改用 Git LFS / 先不處理只記錄），使用者選擇「從尚未推送的 commit 中移除大檔案」。
4. 先建立備份分支 `backup-before-cleanup-20260929`（指向重寫前的原始 3 個 commit），再用 `git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch ...' -- origin/main..HEAD` 只重寫這 3 個本地尚未推送的 commit，把 `data/pcb/archive` 與 6 份 `model.pt` 從歷史中移除（origin 上沒有這些 commit，重寫不影響任何共享歷史）。
   - **副作用**：`filter-branch` 重寫後會用新 HEAD 覆蓋工作目錄，導致這些大檔案連硬碟上的實體檔案都被砍掉，不只是取消追蹤——不是預期行為，發現後立刻用 `git checkout backup-before-cleanup-20260929 -- <paths>` 把檔案救回硬碟（再 `git reset` 取消暫存，維持未追蹤狀態），沒有實際遺失資料，但這是這次操作裡最大的風險點，下次做同類 `filter-branch` 清理前應該先預期這個副作用。
   - 在 `.gitignore` 補上 6 個 `model.pt` 精確路徑與 `data/pcb/archive/` 整個目錄，避免以後又不小心重新加回版控。
5. 重新 `git push origin main` → 待推送內容降到約 18MB，push 成功（`e5fa380a..73071297`）。

**已修正**。備份分支 `backup-before-cleanup-20260929` 目前仍保留在本地（未刪除），供之後需要回頭核對原始歷史時使用；它只存在本機，不會被 push 上 GitHub。

**相關**：`project_pcb_defect_dataset_found`（PCB 資料集來源）、`project_transformer_chat_retrain_v2` / `project_nlp_strengthen_plan`（`gpt_chat_runs_v2`／`gpt_pretrain_runs_v2` 這兩顆 checkpoint 的訓練脈絡）。
