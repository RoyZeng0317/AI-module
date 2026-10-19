# Review — 一般學習資料準備

狀態：第二輪待整合總審。管線 MAJOR 已解除，目前零 MAJOR／BLOCKER；完整資料與正式準備產物尚在建置中，未作最終 PASS。信心：高（管線）；中（暫存資料抽查）。

## 已驗證

- 已讀 AGENTS.md、PLAN、RESEARCH、BUILD_PREPARE 與新增管線、測試、overrides、獨立 eval。
- `.venv/Scripts/python.exe -m pytest tranning/test_prepare_chat_data.py -q`：4 passed。
- 初審 eval 同義群組洩漏已修正：完整來源先建立可傳遞 component，再隔離命中 eval prompt 或 eval group_id 的整組。新增測試涵蓋同組改寫、重複記錄橋接、相同答案傳遞與 eval group_id 直接命中；反例不再進訓練。
- 管線拒絕既有輸出目錄、專案外路徑與直接將 eval 路徑當來源；只輸出 prompt/reply 給模型，保留 metadata 與 audit。
- 原始 pairs 的四筆語言修正採衍生 override；Office eval 正確要求真實工具結果，沒有將教學文字當實際操作能力。
- 抽查寫入中的 32 題雙語暫存內容（crypto 16、Office 16）：百分比損失恢復例題正確，crypto 涵蓋損失、詐騙、私鑰、平台、保管、費用與流動性，未見保證收益。Office 內容為教學及能力邊界，沒有假稱已修改檔案。

## 待最終複審

主資料仍由 builder 分批寫入，不將未完成狀態視為缺件錯誤。待最終 handoff 後審核完整 JSON、全部 topic/group、英文品質、crypto 官方來源、eval 語意隔離、正式 train/val/manifest/audit 及 token audit。

本審查僅涵蓋工程準備，不能證明模型已學會或 Office 功能已實作。正式訓練與 Office 方向仍待使用者回答；本審查未啟動訓練。
