# PLAN
## Restatement & Success Criteria
使用者原文：首先需要根據以下進行判定／只要時區是北京時區就進行初步判定／再有就是IP地址是中國大陸就二次判定／文字內容就是簡體中文是其一，並且逗號的UNICODE碼: \\uff0c／與一些北京話等，如: "三七旮旯话"，”那咋啦“，"如何呢？又能怎？"等一系列進行判定／但是需要注意並非簡體中文就判定一定是中國大陸地區用戶／同時要確定可能是馬來西亞的用戶進行使用。

系統後端內部規則，無前端顯示。分離時區、IP 國別及弱文字訊號，不推斷國籍，不把規則分數宣稱為機率。帳戶限制行為未指定，本輪只產生內部結果，不自動封禁。[High]
## Open Questions for the User
none。只做內部訊號判讀；未規定的帳戶處置不實作。
## Sub-tasks
1. 新增 web/backend/region_signals.py 與可維護 region_rules.json：具體北京時區別名初判，不以 UTC+8 當北京；country==CN 二次 IP 訊號；文字只輔助、不能確認地區；MY IP 必須有獨立狀態，與北京時區衝突不得判為 CN 確認。TW/SG/unknown 同理不因文字判CN。
2. 使用本機 DB-IP country MMDB，maxminddb 讀取。主代理負責取得供應商公開資料並署名來源。缺庫/壞庫/私有IP/無效IP回 unknown，故障不阻塞聊天；不信任請求body的country/IP或未驗證X-Forwarded-For。ASGI request.client 為既有信任邊界，反向代理需正確可信代理設定。
3. ChatRequest 增加可選 client_timezone（長度限128）；前端僅將 Intl 時區名稱附加聊天payload，不增加UI或顯示內部結果。Intl 失敗不阻塞送出。後端評估原始message，結果留 request.state 與 app.state 有界最近結果（最多200筆），不回傳到reply/trace/commands等任何public API、不傳入LLM提示詞。不記錄原始訊息/IP到額外檔案。
4. 測試訊號組合：Shanghai+CN、Shanghai+MY、Kuala_Lumpur+MY簡體、Taipei+TW全形逗號、UTC+8未知IP、CN缺時區、簡體/逗號/用語單獨、偽造country/header、缺/壞資料庫、IPv4/IPv6/私有/無效IP、前端Intl失敗。
## Approach & Verification per Sub-task
單一Builder修改核心與API/前端靜默蒐集，主代理處理資料下載；Reviewer獨立驗證契約及API不洩露。使用pytest假IP國別注入隔離測試及真實MMDB smoke；不以合成測試宣稱所在地準確率。
## Unknowns for Research
已查官方Unicode U+FF0C為FULLWIDTH COMMA，MaxMind說明IP VPN只能定位出口；DB-IP免費country Lite每月更新，CC BY4.0需署名。主代理寫RESEARCH.md。
## Stop Conditions
可用內部判讀、資料庫真實smoke、回歸與獨立審查PASS。不改.env、不覆蓋使用者其他修改、不啟動模型訓練。不新增前端判定UI、封禁/解除流程。必要供應商署名連結僅為資料授權，不顯示評判。
