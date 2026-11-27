# 訓練資料準備工具

新增 `tranning/prepare_chat_data.py` 與 `tranning/test_prepare_chat_data.py`。不修改原始資料、checkpoint、`.env`，不執行模型訓練。

工具以固定 seed 將明確 group_id、正規化相同問題及相同答案做可傳遞分組，再以群組切分。重複記錄的 group_id 亦保留連結，避免去重後破壞群組。任意語意近似不能由此工具證明，仍需人工審查；大量共用答案可能產生大群組，audit 會列出最大群大小。

輸出新目錄內 `train.json`、`val.json`（僅 prompt/reply）、`manifest.json`（metadata、來源位置、分割群組）、`audit.json`（來源 SHA256、語言／分割數量、問題及答案互斥檢查、每筆排除原因和原始記錄）。拒絕覆寫既有目錄、專案外路徑、不合法比例或空驗證分割。可用 `--overrides` 指定既有 prompt 替換，替換原因留在 manifest，原始記錄留在 audit；不存在的覆寫問題會拒絕。

`--eval` 可重複指定多個檔案，僅讀保護用 prompt，正規化重疊資料隔離，不將 eval 答案納入訓練；既有 `data/chat_eval_cases.json` 如存在預設亦受保護。不得把 eval 同時指定為 base/extra/overrides。英文問題配中文字答案隔離，明示翻譯題例外。

審查修正：隔離前先對所有有效來源記錄建立可傳遞 component；eval prompt 命中或 eval 明確 group_id 命中時隔離整個 component，包括同組改寫、翻譯、相同答案、重複記錄所連接的其他群組。避免先刪除命中列／重複列而遺失橋樑。這仍只保證明確 metadata 與正規化相同問答的分組，不代表自動理解所有語意近似。

指令（新增資料檔完成後由整合者執行）：

```powershell
.\.venv\Scripts\python.exe tranning/prepare_chat_data.py --base data/pairs.json --extra data/general_learning_v1.json --overrides data/general_learning_overrides.json --eval data/chat_eval_cases.json --eval data/chat_learning_eval_v1.json --out-dir data/general_learning_prepared_v1 --seed 42 --val-ratio 0.15
```

驗證：`python -m pytest tranning/test_prepare_chat_data.py -q`。測試涵蓋衝突隔離、eval 隔離、英文混中文、去重、可傳遞群組、重複 metadata 群組連結、來源不變、覆寫拒絕、錯誤 seed/ratio、eval 當來源拒絕、找不到的 override、單群組無法切分。沒有長訓練或聲稱模型能力改善。新增資料檔尚未出現時未執行正式資料準備，由整合者待資料完成後執行。
