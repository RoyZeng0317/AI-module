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
