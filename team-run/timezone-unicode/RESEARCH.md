# RESEARCH
查閱日期：2026-10-18（Asia/Taipei）。只使用 Anthropic 官方來源。

- [High] Claude Help Center 的可用地區清單列出 Taiwan，沒有 China／中國大陸。來源：https://support.claude.com/en/articles/8461763-where-can-i-access-claude （頁面標示 March 16, 2026）。支持地區限制的結論，不支持以國籍或輸入文字直接封禁。
- [High] 官方 Safeguards warnings and appeals 說明，可能封禁的原因包括重複違反使用政策、從不支援地區建立帳號、違反服務條款；認為誤封者可登入被封帳號使用申訴表。來源：https://support.claude.com/en/articles/8241253-safeguards-warnings-and-appeals
- [High] 2025-09-04 官方公告也談到不支援地區（以 China 舉例）控制之企業或組織的銷售限制，含直接或間接超過 50% 所有權；這是組織與服務政策，不是 Unicode 檢測規則。來源：https://www.anthropic.com/news/updating-restrictions-of-sales-to-unsupported-regions
- [High] 另讀 https://www.anthropic.com/legal/supported-regions 時頁面標示 Effective November 12, 2026，晚於查閱日期；不能當作截至 2026-10-18 已生效的新條款。最終答案以日期較早的 Help Center 與 2025 公告為主。
- [Medium] 本次查閱的官方來源未說明以 UTC 時差、瀏覽器時區或 Unicode 字元作為自動封禁條件；未找到不等於證明內部完全不使用，不能聲稱已知其未公開風控實作。

技術解讀：[High] offset 與 timezone label 是裝置/瀏覽器設定，不能認定實際位置；多個時區可共享 UTC+8。Unicode 漢字碼位跨語言地區共用，不能從字元編碼認定使用者國籍。學習工具只顯示資料，不推斷或封禁。
