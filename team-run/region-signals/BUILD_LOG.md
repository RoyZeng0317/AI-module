# BUILD_LOG

新增 `web/backend/region_signals.py`、`region_rules.json`、`REGION_SIGNALS.md` 與核心/API/前端 helper 測試。整合 `ChatRequest.client_timezone`（可選、128字上限）及 `/api/chat` 的內部評估。前端只靜默取得 Intl 時區；失敗送 null。沒有評判 UI、沒有公開評判 API、沒有模型提示詞或回覆/trace 注入、沒有封禁。MY IP 與北京時區衝突保留獨立狀態。最近200筆無原始訊息/IP、重啟清空。加入 maxminddb 依賴與 DB-IP 署名連結，文件說明資料授權及代理來源可信邊界。

驗證：`.venv/Scripts/python.exe -m pytest web/backend/tests/test_region_signals.py web/backend/tests/test_app.py -q`，45 passed（新增共用繁體字反例後由主代理重跑）。現有兩則第三方棄用警告，無測試功能失敗。

開發中曾發生測試括號語法錯誤、patch 匹配通知函式而非聊天函式、套件與裸模組 mock 對象不同；均已修正並通過測試，不影響最終通知流程。資料庫由主代理取得，本Builder未觸碰 .env，未修改專案外檔案。未宣稱分類準確率，未啟動訓練。

## 主代理整合驗證
- 三檔最終合跑：test_region_signals.py + test_app.py + test_frontend_chat.py，共50 passed，2項既有deprecation warnings。
- browser-check.cjs：真實PyScript DOM在Shanghai/Kuala_Lumpur/Taipei靜默傳送client_timezone、無評判UI、正常聊天，PASS。API mock僅用於前端驗證；後端整合另由TestClient驗證。
- 真實本機MMDB分類smoke：cn_timezone_and_ip／malaysia_ip／malaysia_timezone_conflict／other_country_ip／unknown，詳real-database-results.json；非真實人員位置驗證。
- 供應商原始資料、SHA256及來源見database-download.json。產品既有追蹤檔git diff --check通過。
