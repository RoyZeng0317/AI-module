# Sinco 情緒訓練資料交付

PR：https://github.com/RoyZeng0317/AI-module/pull/7

本 ZIP 包含已產生的訓練／驗證／測試資料、來源／授權、匯入與驗證報告、匯入程式及測試。沒有模型權重。原始下載封存檔不包含在 ZIP，但來源 manifest 可重建。

合併 PR #7 後，把 ZIP 中 data/emotion_datasets 的資料放到專案同名資料夾，即可依其 README 的 --data／--val-data 指令準備微調。原本 data/pairs.json 不需覆蓋。

資料數：回覆 train 27,934、val 3,664、test 3,766。CPED train 83,964；GoEmotions train 43,166。後兩者為獨立分類資料，不能直接給原 action.py 的分數迴歸模型。

新增回覆主要是英文原文，沒有翻成中文。訓練資料已準備，但尚未執行模型訓練或宣稱真實中文情緒回覆改善。

匯入程式10項測試通過，來源與資料隔離檢查通過；全部回覆資料未超過目前512-token block。

所有來源授權見 data/emotion_datasets/licenses 與 README；請保留來源及研究／非商業用途限制。
