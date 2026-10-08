# 情緒資料加入訓練集（2026-10-08）

使用者指示將找到的多個資料集加入訓練資料。已實際下載 ESConv、EmpatheticDialogues、CPED、GoEmotions 與 ConvLab 整理的 DailyDialog，固定來源版本與 SHA256。

新回覆資料：data/emotion_datasets/pairs_train.json（27,934）、pairs_val.json（3,664）、pairs_test.json（3,766），符合現有 Transformer ChatSFTDataset 格式。包含 ESConv 的關心策略、EmpatheticDialogues 初次自述與第一個回覆、DailyDialog 的情緒句對及符合規則的既有 Sinco 對話。原本 data/pairs.json 不覆蓋；選取新 --data 與 --val-data 即可用於後續微調。

CPED 的 comfort 台詞有些仍是敷衍／責備，故不直接匯入安慰回覆目標。它轉成繁中、保留原細粒度標籤，與 GoEmotions 分別產生分類 train/val/test JSONL。這兩組分類資料還需對應分類頭與 loader 才能訓練，不直接給原本 action.py 的 0~5 分數模型。

已驗證所有回覆／分類 splits 的完全相同正規化文字與對話 ID 無跨集合重疊；所有回覆對現有 BPE 的 512-token block 無超長。10 項匯入測試通過，並實際抽各 split 的四種來源資料跑既有 ChatSFTDataset／DataLoader，資料可被既有微調管線載入、target 含可評分 token。

重要語言限制：新增回覆資料是英文原文，沒有自動翻成中文；新 train 約 98% 英文，適合英文／雙語實驗，不能把它宣稱為繁中情緒回覆已改善。中文主集保持可用。實際英文生成／繁中表現仍需本機權重與獨立評估，沒有執行訓練。

詳見 data/emotion_datasets/README.md；来源／授權文件與匯入程式可重建全部資料。實際語料另提供 ZIP，不把大量研究資料推進公開 Git 歷史。此 PR 與 #5（評估）、#6（明確情緒短句）可獨立合併。
