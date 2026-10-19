# PLAN
## Restatement & Success Criteria
使用者要求全部三項：Python 本機時區與 Unicode 查碼練習、既有網頁顯示與字元分析、Claude 官方地區限制與封禁政策查證。只顯示本機可讀設定與字元資料，不進行國籍/實際所在地推斷、不建立封禁機制、不儲存或上傳分析資料。[High]
## Open Questions for the User
none；採預設不儲存紀錄。UTC 時差與時區名稱分別顯示，無法取得 IANA 名稱時誠實顯示平台名稱。
## Sub-tasks
1. Python 標準庫 CLI：讀取系統 UTC 與 local aware datetime、UTC offset、zone label；支援 --text；逐 code point 的 U+、Unicode name、category；標示結合字與 emoji 可能由多個碼位構成。限制顯示 200 code points。
2. 網頁：在現有 tools details 中增加入口，開獨立 modal；使用 Intl API 讀瀏覽器時區與當下 local offset；輸入 Unicode 分析使用 Python unicodedata，textContent 安全輸出，限制 200 code points；無 fetch/localStorage。保留既有全部註解與未提交修改。
3. 主代理查證官方來源並在聊天提供結果。禁止猜測 Anthropic 的未公開封禁演算法。
4. Reviewer 獨立核對，前端 pytest 及 CLI smoke，主代理瀏覽器模擬時區與 Unicode 實測。
## Approach & Verification per Sub-task
新增 baisc/timezone_unicode.py；網頁沿用 action.py 的 modal 與 style.css，以避免新增靜態暴露檔案。測 Asia/Taipei、Asia/Shanghai 時區皆為 UTC+8、emoji+combining+HTML 字元輸入，關閉 modal、無資料請求。
## Unknowns for Research
Claude 支援地區、封禁及申訴規則由官方網站查證；不影響學習工具實作。
## Stop Conditions
功能與驗證完成、Reviewer 無重大問題後交付。禁止修改 .env，僅專案內修改，不啟動訓練或外部 LLM API，不自行寫驗收完成日誌。

## 使用者需求修正（2026-10-19）
使用者明確要求「這個評判是要求不顯示在網頁前端」。撤下本輪新增的前端工具入口、視窗及相關程式/CSS；保留獨立 Python 範例與官方政策查證。前端截圖及瀏覽器驗證僅為已撤下版本的歷史證據，不代表目前交付功能。內部判讀要做觀察／人工審核或執行帳戶限制的規則尚待使用者回覆；未授權猜測觸發條件，不新增封禁實作。
